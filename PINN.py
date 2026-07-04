##############################
# NEURAL NETWORK DEFINITION
##############################
import configparser as cp
import math
import os
import torch
import torch.nn as nn

from logger import Logger


# Configuration reading
conf = cp.ConfigParser()
conf.read("config.ini")

# Instantiate log class
base_path = os.path.dirname( os.path.abspath(__file__) )
log_path = os.path.sep.join([base_path, 'log'])
logger = Logger(filepath=log_path)
separator = '-' * 50


class PINN(nn.Module):

    def __init__(self, k_func, x_interval, t_interval):
        super(PINN, self).__init__()

        # Save parameters
        self.k_func = k_func
        self.x_interval = x_interval
        self.t_interval = t_interval

        # Read from configuration
        self.class_name = 'PINN'
        self.read_configuration()

        # Define the activation function
        #self.activation = nn.Tanh()
        self.activation = SineActivation(omega_0=self.omega_0)

        # Define the loss function
        self.criterion = nn.MSELoss()

        # Parameters reading from configuration
        neurons_list = []
        for layer in conf['PINN.structure']:
            neuron_no = conf['PINN.structure'][layer]

            # Check if there is a number
            if neuron_no:
                neuron_no = int(neuron_no)
                neurons_list.append(neuron_no)
        
        if len(neurons_list) == 0:
            logger.error(self.class_name, 'No layers found on configuration file!')
        
        logger.info(self.class_name, 'Found {} layers on configuration file'.format(len(neurons_list)))

        # Building the model structure
        self.model = nn.Sequential(
            nn.Linear(2, neurons_list[0]),
            self.activation
        )

        # Add a layer to the model for each neuron_layer on configuration
        for i in range(1, len(neurons_list)):
            self.model.append( nn.Linear(neurons_list[i-1], neurons_list[i]) )
            self.model.append( self.activation )
        
        # Add the output layer
        self.model.append( nn.Linear(neurons_list[-1], 1) )

        # Set the SIREN parameters
        with torch.no_grad():
            is_first = True
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    num_input = layer.in_features
                    if is_first:
                        # First layer, weights must be between -1/in_features and 1/in_features
                        layer.weight.uniform_(-1.0 / num_input, 1.0 / num_input)
                        is_first = False
                    else:
                        # Internal layer and output scaled according to omega_0
                        b = math.sqrt(6.0 / num_input) / self.activation.omega_0
                        layer.weight.uniform_(-b, b)
                    
                    # Inizializziamo i bias a zero per stabilità iniziale
                    if layer.bias is not None:
                        layer.bias.zero_()

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)

        logger.info(self.class_name, self.model)
    
    def read_configuration(self):
        logger.info(self.class_name, separator)
        self.omega_0 = float( conf["PINN.params"]["omega_0"] )
        self.w_pde, self.w_pde2 = float( conf["PINN.params"]["w_pde"] ), float( conf["PINN.params"]["w_pde2"] )
        self.w_ic, self.w_ic2 = float( conf["PINN.params"]["w_ic"] ), float( conf["PINN.params"]["w_ic2"] )
        self.w_bc, self.w_bc2 = float( conf["PINN.params"]["w_bc"] ), float( conf["PINN.params"]["w_bc2"] )
        self.ftype = conf["PINN.params"]['function_type']

        logger.info(self.class_name, 'omega_0 for SIREN:                        {:.2f}'.format(self.omega_0))
        logger.info(self.class_name, 'w_pde for a first training portion:       {:.3f}'.format(self.w_pde))
        logger.info(self.class_name, 'w_pde for a fisecondrst training portion: {:.3f}'.format(self.w_pde2))
        logger.info(self.class_name, 'w_ic for a first training portion:        {:.3f}'.format(self.w_ic))
        logger.info(self.class_name, 'w_ic for a second training portion:       {:.3f}'.format(self.w_ic2))
        logger.info(self.class_name, 'w_bc for a first training portion:        {:.3f}'.format(self.w_bc))
        logger.info(self.class_name, 'w_bc for a second training portion:       {:.3f}'.format(self.w_bc2))
        logger.info(self.class_name, separator)

    def forward(self, x, t):
        x_min, x_max = self.x_interval[0], self.x_interval[1]
        t_min, t_max = self.t_interval[0], self.t_interval[1]

        # Scale the inputs
        x = (x - x_min) / (x_max - x_min)
        t = (t - t_min) / (t_max - t_min)

        output = torch.cat([x, t], dim=1)
        return self.model(output)


    def diff_eq(self, x, t):
        t = t.detach().requires_grad_(True)
        x = x.detach().requires_grad_(True)

        # Predict the function u
        u = self.forward(x, t)

        # Compute temporal partial derivative (du/dt)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute spatial partial derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute the flux and its partial derivative
        k = self.k_func(x, ftype=self.ftype)
        flux = k * u_x 
        flux_x = torch.autograd.grad(flux, x, grad_outputs=torch.ones_like(flux), create_graph=True)[0]

        # Return the implicit function for the differential equation
        f = u_t - flux_x
        return f


    def train_step(self, x_coll, t_coll,
                  x_ic, t_ic, u_ic,
                  x_bc0, t_bc0, x_bc1, t_bc1,
                  epoch, n_epochs):

        # Weights definition
        if epoch < (n_epochs // 3):
            w_pde = self.w_pde
            w_ic = self.w_ic
            w_bc = self.w_bc
        else:
            w_pde = self.w_pde2
            w_ic = self.w_ic2
            w_bc = self.w_bc2

        self.optimizer.zero_grad()

        # Estimate the PDE loss
        f_out = self.diff_eq(x_coll, t_coll)
        loss_pde = self.criterion(f_out, torch.zeros_like(f_out))

        # Estimate the IC loss
        u_ic_pred = self.forward(x_ic, t_ic)
        loss_ic = self.criterion(u_ic_pred, u_ic)

        # Estimate the boundary loss
        u_bc0_pred = self.forward(x_bc0, t_bc0)
        loss_bc0 = self.criterion(u_bc0_pred, torch.zeros_like(u_bc0_pred))

        u_bc1_pred = self.forward(x_bc1, t_bc1)
        loss_bc1 = self.criterion(u_bc1_pred, torch.zeros_like(u_bc1_pred))

        # Estimate the total loss
        loss_total = (w_pde * loss_pde) + (w_ic * loss_ic) + (w_bc * loss_bc0) + (w_bc * loss_bc1)
        loss_total.backward()
        self.optimizer.step()

        return {
          'loss_total': loss_total.item(),
          'loss_pde': loss_pde.item(),
          'loss_ic': loss_ic.item(),
          'loss_bc0': loss_bc0.item(),
          'loss_bc1': loss_bc1.item()
      }


########################################
# ACTIVATION FUNCTION FOR SIREN
########################################
class SineActivation(nn.Module):

    def __init__(self, omega_0:float):
        super(SineActivation, self).__init__()
        self.omega_0 = omega_0
    
    def forward(self, x):
        return torch.sin(self.omega_0 * x)