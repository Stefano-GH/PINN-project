import matplotlib.pyplot as plt
import numpy as np
import os


###################################################
# DATA GENERATION
###################################################
def generate_data(n_coll:int, n_ic, n_bc, x_interval=(0.0, 1.0), t_interval=(0.0, 1.0)):
    '''
    :n_coll:     - Number of collocation points
    :k_func:     - Function defining k(x,t)
    :n_ic:       - Number of border points. They have the initial condition constraint 
    :x_interval: - Domain for x-variable delimited by the two extreme points
    :t_interval: - Domain for t-variable delimited by the two extreme points
    '''
    # Pick minimum and maximum for both the variables
    x_min, x_max = x_interval[0], x_interval[1]
    t_min, t_max = t_interval[0], t_interval[1]

    np.random.seed(7)

    # Generate the collocation data points
    x_coll = np.random.rand(n_coll, 1) * (x_max - x_min) + x_min
    t_coll = np.random.rand(n_coll, 1) * (t_max - t_min) + t_min

    # Generate the initial condition points
    x_ic = np.random.rand(n_ic, 1) * (x_max - x_min) + x_min
    t_ic = np.full((n_ic, 1), t_min)

    # Generate border points
    x_bc0 = np.full((n_bc, 1), x_min)
    t_bc0 = np.random.rand(n_bc, 1) * (t_max - t_min) + t_min
    x_bc1 = np.full((n_bc, 1), x_max)
    t_bc1 = np.random.rand(n_bc, 1) * (t_max - t_min) + t_min

    return {
        'x_coll': x_coll,
        't_coll': t_coll,
        'x_ic': x_ic,
        't_ic': t_ic,
        'x_bc0': x_bc0,
        't_bc0': t_bc0,
        'x_bc1': x_bc1,
        't_bc1': t_bc1
    }


###################################################
# PLOT THE DATASET
###################################################
def plot_dataset(data, img_path):
    x_coll, t_coll = data['x_coll'], data['t_coll']
    x_ic, t_ic = data['x_ic'], data['t_ic']
    x_bc0, t_bc0, x_bc1, t_bc1 = data['x_bc0'], data['t_bc0'], data['x_bc1'], data['t_bc1']
    path = os.path.sep.join([img_path, 'original data.png'])

    plt.figure(figsize=(8, 6))
    plt.scatter(x_coll, t_coll, s=2, label='Collocation', alpha=.5)
    plt.scatter(x_ic, t_ic, s=15, label='Initial condition')
    plt.scatter(x_bc0, t_bc0, s=15, label='Boundary x={:.2f}'.format(x_coll.min()))
    plt.scatter(x_bc1, t_bc1, s=15, label='Boundary x={:.2f}'.format(x_coll.max()))
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.savefig(path, dpi=150)
    #plt.show()


###################################################
# PLOT THE RESULTS
###################################################
def plot_1d_results(img_path:str, x:np.ndarray, y:np.ndarray, y_true:np.ndarray=None, title:str="Training loss", x_label:str="t", y_label:str="L(t)"):
    '''
    :path: directory where you want the image to be saved
    :param x: x-values for the main plot (model prediction)
    :param y: y-values for the main plot (model prediction)
    :param y_true: y-values for the ground truth (optional)
    '''
    plt.figure(figsize=(10, 5))
    plt.title(title, fontsize=20)
    plt.plot(x, y, color='orange', lw=3, label='model')
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)

    path = os.path.sep.join([img_path, '1D plot - {}.png'.format(title)])

    # Optional elements
    if y_true is not None:
        plt.plot(x, y_true, color='blue', lw=3, ls='--', label='Ground truth')
        
    plt.legend()
    plt.grid(True)
    plt.savefig(path, dpi=150)
    #plt.show()


def plot_heatmap(img_path:str, y_pred:np.ndarray, y_true:np.ndarray, error:np.ndarray, extent:np.ndarray, k_func_type:str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(y_pred, extent=extent, origin='lower', aspect='auto')
    axes[0].set_title('Model')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(y_true, extent=extent, origin='lower', aspect='auto')
    axes[1].set_title('Ground truth')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error, extent=extent, origin='lower', aspect='auto')
    axes[2].set_title('Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('t')
    fig.colorbar(im2, ax=axes[2])

    plt.suptitle(f'Heatmap - {k_func_type} varying coefficient')
    plt.tight_layout()

    path = os.path.sep.join([img_path, 'heatmap.png'])
    plt.savefig(path, dpi=150)
    #plt.show()


def plot_2d_curve(img_path:str, x:np.ndarray, t:np.ndarray, u_pred:np.ndarray, u_true:np.ndarray, k_func_type:str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection':'3d'})

    surf1 = axes[0].plot_surface(x, t, u_pred, cmap='viridis')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_zlabel('u(x,t)')
    axes[0].set_title('Model')
    fig.colorbar(surf1, ax=axes[0])

    surf2 = axes[1].plot_surface(x, t, u_true, cmap='viridis')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_zlabel('u(x,t)')
    axes[1].set_title('Ground truth')
    fig.colorbar(surf2, ax=axes[1])

    fig.suptitle(f'3D curve - {k_func_type} varying coefficient')
    plt.tight_layout()
    
    path = os.path.sep.join([img_path, '3D plot.png'])
    plt.savefig(path, dpi=150)
    #plt.show()