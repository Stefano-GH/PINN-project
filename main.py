###########################################################
# Author: Stefano Tumino                                  #
# Course: AI models for physics                           #
#                                                         #
# Diffusion equation with variable coefficients           #
#                                                         #
# We want to solve the equation                           #
# \partial u / \partial t - \nabla \cdot (k \nabla u) = 0 #
# Dirichlet conditions: u(0,t) = u(L,t) = 0               #
###########################################################
import configparser as cp
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from logger import Logger
from PINN import PINN
from utils import generate_data, plot_dataset, plot_1d_results, plot_heatmap, plot_2d_curve

# Instantiate log class
base_path = os.path.dirname( os.path.abspath(__file__) )
log_path = os.path.sep.join([base_path, 'log'])
logger = Logger(filepath=log_path)
separator = '-' * 50

# First print for beginning the session
logger.info('', r"   __ _ _        __        __    __ _ _   ")
logger.info('', r"  / / / /       /  \      |  |   \ \ \ \  ")
logger.info('', r" / / / /       /    \     |  |    \ \ \ \ ")
logger.info('', r"( ( ( (       /  /\  \    |  |     ) ) ) )")
logger.info('', r" \ \ \ \     /  ____  \   |  |    / / / / ")
logger.info('', r"  \_\_\_\===/__/====\__\==|__|===/_/_/_/  ")
logger.info('', 'New session has started!')

# Import the configurations
conf = cp.ConfigParser()
conf.read("config.ini")

# Establish the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info('Device Management', 'Device:\t{}'.format(device))


###################################################
# CONFIGURATION FUNCTIONS
###################################################
def k_function(x, ftype:str = 'linear'):
    '''
    params:
    x
    type - Must be in ('linear', 'piecewise', 'quadratic'). Type of function to be used for coefficient
    '''

    if ftype == 'linear':
        return 1 + x
    elif ftype == 'piecewise':
        return torch.where(x < 1.0, torch.tensor(0.1, device=x.device), torch.tensor(1.0, device=x.device))
    elif ftype == 'quadratic':
        return x**2


def u_value(x, t):
    # TODO: dimostrare la soluzione
    return torch.exp(- torch.pi**2 * t) * torch.sin(torch.pi * x)



###################################################
# CONFIGURATION READING
###################################################
def configuration_reading():
    global n_coll, n_ic, n_bc, data_filename
    global n_epochs, x_min, x_max, t_min, t_max, significance_to_print
    global test_data_size
    global k_func_type

    class_name = 'Configuration'
    logger.info(class_name, separator)

    # Parameters for generating the physical system
    n_coll = int( conf['data.point.generation']['collocation_points_no'] )
    n_ic = int( conf['data.point.generation']['initial_condition_points_no'] )
    n_bc = int( conf['data.point.generation']['boundary_condition_points_no'] )
    data_filename = conf['data.point.generation']['data_filename']

    logger.info(class_name, 'Number of collocation points:               {}'.format(n_coll))
    logger.info(class_name, 'Number of initial condition points:         {}'.format(n_ic))
    logger.info(class_name, 'Number of boundary condition points:        {}'.format(n_bc))
    logger.info(class_name, 'Filename for the dataset:                   {}'.format(data_filename))

    # Parameters for training the model
    n_epochs = int( conf['model.train']['n_epochs'] )
    x_min, x_max = float( conf['model.train']['x_min'] ), float( conf['model.train']['x_max'] )
    t_min, t_max = float( conf['model.train']['t_min'] ), float( conf['model.train']['t_max'] )
    significance_to_print = float( conf['model.train']['significance_to_print'] )

    logger.info(class_name, 'Number of epochs:                           {}'.format(n_epochs))
    logger.info(class_name, 'Interval for x:                             [{}, {}]'.format(x_min, x_max))
    logger.info(class_name, 'Interval for t:                             [{}, {}]'.format(t_min, t_max))
    logger.info(class_name, 'Significance to print info during training: {}%'.format(significance_to_print*100))

    # Parameters for testing
    test_data_size = int( conf['model.test']['test_data_size'] )
    logger.info(class_name, 'Size of test data:                          {}'.format(test_data_size))

    # Parameters of the model system
    k_func_type = conf["PINN.params"]["function_type"]
    logger.info(class_name, 'Function type:                              {}'.format(k_func_type))


###################################################
# DATA POINT GENERATION
###################################################
def dataset_generation(data_path, class_name, x_interval, t_interval):
    
    # If the dataset does not exists, then make it
    if not os.path.isfile(data_path):
        data = generate_data(n_coll, n_ic, n_bc, x_interval, t_interval)
        plot_dataset(data, img_path)
        np.savez(data_path, **data)

        logger.info(class_name, 'Dataset (and related image) correctly generated.\n\tPath: {}'.format(data_path))

    else:
        logger.info(class_name, 'Dataset already exists...')


###################################################
# DATA LOADING
###################################################
def dataset_loading(data_path):
    global x_coll, t_coll, x_ic, t_ic, x_bc0, t_bc0, x_bc1, t_bc1

    data = np.load(data_path)
    x_coll = torch.tensor(data['x_coll'], dtype=torch.float32).to(device)
    t_coll = torch.tensor(data['t_coll'], dtype=torch.float32).to(device)
    x_ic   = torch.tensor(data['x_ic'], dtype=torch.float32).to(device)
    t_ic   = torch.tensor(data['t_ic'], dtype=torch.float32).to(device)
    x_bc0  = torch.tensor(data['x_bc0'], dtype=torch.float32).to(device)
    t_bc0  = torch.tensor(data['t_bc0'], dtype=torch.float32).to(device)
    x_bc1  = torch.tensor(data['x_bc1'], dtype=torch.float32).to(device)
    t_bc1  = torch.tensor(data['t_bc1'], dtype=torch.float32).to(device)


###################################################
# MAIN
###################################################
def main(verbose=True):

    # Read the configuration file
    configuration_reading()


    # Dataset generation, IF not exist
    class_name = 'Data Generation'
    logger.info(class_name, separator)

    data_path = os.path.sep.join([base_path, data_filename])
    logger.info(class_name, 'Check if the data file is available...')

    x_interval = (x_min, x_max)
    t_interval = (t_min, t_max)
    dataset_generation(data_path, class_name, x_interval, t_interval)


    # Dataset loading
    class_name = 'Data Loading'
    logger.info(class_name, separator)

    try:
        dataset_loading(data_path)
        logger.info(class_name, 'Data correctly loaded!')
    except Exception as e:
        logger.error(class_name, 'Error in loading the dataset: {}'.format(str(e)))
        return
    

    ###################################################
    # TRAIN THE MODEL
    ###################################################
    class_name = 'Model Training'
    logger.info(class_name, separator)

    # Prepare the PINN parameters
    x_interval = (x_min, x_max)
    t_interval = (t_min, t_max)
    
    # Instantiate the model
    model = PINN(k_function, x_interval, t_interval).to(device)
    loss_evolution = []

    # Start the training
    logger.info(class_name, 'Training...')
    epochs = range(1, n_epochs+1)

    # Differentiate the equation
    u_ic = u_value(x_ic, t_ic).to(device)

    model.train()
    for epoch in epochs:

        # Train step
        loss = model.train_step(x_coll, t_coll, x_ic, t_ic, u_ic, x_bc0, t_bc0, x_bc1, t_bc1, epoch, n_epochs)
        loss_evolution.append(loss['loss_total'])

        # Check if the new loss has a difference of at least 10% from the previous value
        if len(loss_evolution) == 1 and verbose:
            logger.info(class_name, 'Epoch: {:>6} | Total loss: {:.5f} | PDE loss: {:.5f} | IC loss: {:.5f} | BC x=0 loss: {:.5f} | BC x=1 loss: {:.5f}'
                        .format(epoch, loss['loss_total'], loss['loss_pde'], loss['loss_ic'], loss['loss_bc0'], loss['loss_bc1']))

        elif (len(loss_evolution) > 2) and verbose:
            denominator = max(loss_evolution[-1], 1e-8)
            delta = abs(loss_evolution[-1] - loss_evolution[-2]) / denominator
            if (delta >= significance_to_print) or (n_epochs % 100 == 0):
                logger.info(class_name, 'Epoch: {:>6} | Total loss: {:.5f} | PDE loss: {:.5f} | IC loss: {:.5f} | BC x=0 loss: {:.5f} | BC x=1 loss: {:.5f}'
                            .format(epoch, loss['loss_total'], loss['loss_pde'], loss['loss_ic'], loss['loss_bc0'], loss['loss_bc1']))
        
    # plot the training loss
    plot_1d_results(img_path, epochs, loss_evolution, title=f"Training loss - {k_func_type} varying coefficient")
    logger.info(class_name, 'Training loss plot saved on the path\t{}'.format(img_path))


    ###################################################
    # TEST THE MODEL
    ###################################################
    class_name = 'Model Testing'
    logger.info(class_name, separator)

    # Test points
    x_1d = np.linspace(x_min, x_max, test_data_size)
    t_1d = np.linspace(t_min, t_max, test_data_size)

    X, T = np.meshgrid(x_1d, t_1d)
    x_test = torch.tensor(X.reshape(-1, 1), dtype=torch.float32).to(device)
    t_test = torch.tensor(T.reshape(-1, 1), dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_test, t_test).cpu().numpy().reshape(test_data_size, test_data_size)
        u_true = u_value(torch.tensor(X, dtype=torch.float32),
                        torch.tensor(T, dtype=torch.float32))
        u_true = u_true.cpu().numpy().reshape(test_data_size, test_data_size)

        u_pred_2d, u_true_2d = u_pred.reshape(test_data_size, test_data_size), u_true.reshape(test_data_size, test_data_size)

        error_l1 = np.abs(u_true - u_pred)
        error_l2 = np.sqrt( np.mean((u_true - u_pred)**2) )

        logger.info(class_name, 'Error L1: {:.3f}'.format(np.mean(error_l1)))
        logger.info(class_name, 'Error L2: {:.3f}'.format(error_l2))

        # Graphs
        logger.info(class_name, 'Making 1D plot...')
        plot_1d_results(img_path=img_path, x=t_1d, y=u_pred_2d[-1], y_true=u_true_2d[-1],
                        title=f'1D solution - {k_func_type} varying coefficient', x_label='t', y_label='u(x,t)')

        logger.info(class_name, 'Making heatmap...')
        extent = [x_min, x_max, t_min, t_max]
        plot_heatmap(img_path=img_path, y_pred=u_pred, y_true=u_true, error=error_l1, extent=extent, k_func_type=k_func_type)

        logger.info(class_name, 'Making 2D plot...')
        plot_2d_curve(img_path=img_path, x=X, t=T, u_pred=u_pred, u_true=u_true, k_func_type=k_func_type)


###################################################
# RUN THE PROGRAM
###################################################
if __name__ == '__main__':

    # Directory for images
    img_path = os.path.sep.join([base_path, 'images'])
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    
    main(verbose=True)
    logger.close_file()