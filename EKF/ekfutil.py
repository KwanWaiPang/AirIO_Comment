import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import torch


def plot_bias_subplots(bias, title='Bias of Three Axes', save_path=None):
    """
    Plots the bias of three axes in subplots arranged vertically.

    Parameters:
    - bias_x: array-like, bias data for the X-axis.
    - bias_y: array-like, bias data for the Y-axis.
    - bias_z: array-like, bias data for the Z-axis.
    - title: str, optional, the main title of the plot.
    - save_path: str, optional, file path to save the plot image.
    """

    # Ensure all bias arrays are of the same length
    bias_x, bias_y, bias_z = bias[:, 0], bias[:, 1], bias[:, 2]
    indices = range(len(bias_x))
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)

    # Plot bias_x in the first subplot
    axs[0].plot(indices, bias_x, label='Bias X-axis', color='r')
    axs[0].set_ylabel('Bias X')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Plot bias_y in the second subplot
    axs[1].plot(indices, bias_y, label='Bias Y-axis', color='g')
    axs[1].set_ylabel('Bias Y')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # Plot bias_z in the third subplot
    axs[2].plot(indices, bias_z, label='Bias Z-axis', color='b')
    axs[2].set_xlabel('Sample Index')
    axs[2].set_ylabel('Bias Z')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)

    # Add a main title to the figure
    fig.suptitle(title, fontsize=16)


    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        # Display the p
        plt.plot()
        
        
def interp_xyz(time, opt_time, xyz):

    intep_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
    intep_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
    intep_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])

    inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
    return torch.tensor(inte_xyz)