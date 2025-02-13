import os

import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import torch

def visualize_body_vel(save_prefix, save_folder, gt_vel, vel, label="AirIO"):
    # Split the ground truth and predicted velocities into x, y, z components, downsampling by a factor of 50
    v_gt_x, v_gt_y, v_gt_z = torch.split(gt_vel.cpu(), 1, dim=1)
    airVel_x, airVel_y, airVel_z = torch.split(vel.cpu(), 1, dim=1)
    
    # Set up figure and subplots
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 1, fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Plot x-component of velocities
    ax1.plot(airVel_x, label=label)
    ax1.plot(v_gt_x, label="Ground Truth")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Velocity X')
    ax1.legend()
    
    # Plot y-component of velocities
    ax2.plot(airVel_y, label=label)
    ax2.plot(v_gt_y, label="Ground Truth")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity Y')
    ax2.legend()
    
    # Plot z-component of velocities
    ax3.plot(airVel_z, label=label)
    ax3.plot(v_gt_z, label="Ground Truth")
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Velocity Z')
    ax3.legend()
    ax3.set_ylim(-4,1)
    
    # Save the plot
    save_path = os.path.join(save_folder, f"{save_prefix}_body_velocity.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    

def visualize_motion_cov(save_prefix, save_folder,state_dist, cov, label="AirIO"):
    ### visualize cov and error
    vel_error = state_dist.pow(2)
    cov = cov[0]
    vel_error_norm = vel_error.norm(dim=-1)
    cov_norm = cov.norm(dim=-1)
    err_x, err_y, err_z = torch.split(vel_error.cpu(), 1, dim=1)
    cov_x, cov_y, cov_z = torch.split(cov.cpu(), 1, dim=1)
    
    fig = plt.figure(figsize = (12,6))
    gs = GridSpec(3,2)
    
    ax11 = fig.add_subplot(gs[0:,0])
    ax11.plot(cov_norm, label = "covariance")
    ax11.plot(vel_error_norm, label = "velocity loss")
    ax11.set_xlabel('time(s)')
    ax11.legend()
    ax11.grid(True)
    
    ax21 = fig.add_subplot(gs[0:1,1])
    ax22 = fig.add_subplot(gs[1:2,1])
    ax23 = fig.add_subplot(gs[2:3,1])
  
    ax21.plot(cov_x, label = "covariance")
    ax22.plot(cov_y, label = "covariance")
    ax23.plot(cov_z, label = "covariance")
      
    ax21.plot(err_x, label = "velocity loss")
    ax22.plot(err_y, label = "velocity loss")
    ax23.plot(err_z, label = "velocity loss")
    
    ax21.legend()
    ax22.legend()
    ax23.legend()
    
    ax21.set_xlabel('time(s)')
    ax22.set_xlabel('time(s)')
    ax23.set_xlabel('time(s)')

    ax21.grid(True)
    ax22.grid(True)
    ax23.grid(True)
    
    save_prefix += "_cov_loss.png"
    plt.savefig(os.path.join(save_folder, save_prefix), dpi = 300)
    plt.close()
    

    
def visualize_motion(save_prefix, save_folder, outstate,infstate,label="AirIO"):
    ### visualize gt&netoutput velocity, 2d trajectory. 
    gt_x, gt_y, gt_z                = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)
    
    v_gt_x, v_gt_y, v_gt_z       = torch.split(outstate['vel_gt'][0][::50,:].cpu(), 1, dim=1)
    airVel_x, airVel_y, airVel_z = torch.split(infstate['net_vel'][0][::50,:].cpu(), 1, dim=1)
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 2) 

    ax1 = fig.add_subplot(gs[:, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[1, 1]) 
    ax4 = fig.add_subplot(gs[2, 1]) 
   
    #visualize traj 
    ax1.plot(airTraj_x, airTraj_y, label=label)
    ax1.plot(gt_x     , gt_y     , label="Ground Truth")
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.legend()
    
    #visualize vel
    ax2.plot(airVel_x,label=label)
    ax2.plot(v_gt_x,label="Ground Truth")
    
    ax3.plot(airVel_y,label=label)
    ax3.plot(v_gt_y,label="Ground Truth")
    
    ax4.plot(airVel_z,label=label)
    ax4.plot(v_gt_z,label="Ground Truth")
    
    ax2.set_xlabel('time')
    ax2.set_ylabel('velocity')
    ax2.legend()
    ax3.legend()
    ax4.legend()
    save_prefix += "_state.png"
    plt.savefig(os.path.join(save_folder, save_prefix), dpi = 300)
    plt.close()

def visualize_3d_trajectory(save_prefix, save_folder, outstate,infstate,label="AirIO"):
    gt_x, gt_y, gt_z                = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)
    
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot3D(airTraj_x, airTraj_y,airTraj_z, label=label)
    ax.plot3D(gt_x     , gt_y     , gt_z, label="Ground Truth")
    ax.set_zlabel('Z axis')
    save_prefix += "_trajectory_xyz.png"
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.legend()
    plt.savefig(os.path.join(save_folder, save_prefix), dpi = 300)
    plt.close()


def visualize_all_state_error(
    save_prefix,
    relative_outstate,
    relative_infstate,
    bias_relative_infstate,
    save_folder=None,
    mask=None,
    file_name="state_error_compare.png",
):
    if mask is None:
        outstate_pos_err = relative_outstate["pos_dist"][0]
        outstate_vel_err = relative_outstate["vel_dist"][0]
        outstate_rot_err = relative_outstate["rot_dist"][0]

        infstate_pos_err = relative_infstate["pos_dist"][0]
        infstate_vel_err = relative_infstate["vel_dist"][0]
        infstate_rot_err = relative_infstate["rot_dist"][0]

        bias_infstate_pos_err = bias_relative_infstate["pos_dist"][0]
        bias_infstate_vel_err = bias_relative_infstate["vel_dist"][0]
        bias_infstate_rot_err = bias_relative_infstate["rot_dist"][0]
    else:
        outstate_pos_err = relative_outstate["pos_dist"][0, mask]
        outstate_vel_err = relative_outstate["vel_dist"][0, mask]
        outstate_rot_err = relative_outstate["rot_dist"][0, mask]

        infstate_pos_err = relative_infstate["pos_dist"][0, mask]
        infstate_vel_err = relative_infstate["vel_dist"][0, mask]
        infstate_rot_err = relative_infstate["rot_dist"][0, mask]

        bias_infstate_pos_err = bias_relative_infstate["pos_dist"][0, mask]
        bias_infstate_vel_err = bias_relative_infstate["vel_dist"][0, mask]
        bias_infstate_rot_err = bias_relative_infstate["rot_dist"][0, mask]

    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("Integration vs AirIMU vs Adapt Integration error")

    axs[0].plot(outstate_pos_err, color="black", linewidth=1)
    axs[0].plot(infstate_pos_err, color="red", linewidth=1)
    axs[0].plot(bias_infstate_pos_err, color="blue", linewidth=1)
    axs[0].legend(["integration_pos_error", "AirIMU_pos_error", "Adapt_pos_error"])
    axs[0].grid(True)

    axs[1].plot(outstate_vel_err, color="black", linewidth=1)
    axs[1].plot(infstate_vel_err, color="red", linewidth=1)
    axs[1].plot(bias_infstate_vel_err, color="blue", linewidth=1)
    axs[1].legend(["integration_vel_error", "AirIMU_vel_error", "Adapt_vel_error"])
    axs[1].grid(True)

    axs[2].plot(outstate_rot_err, color="black", linewidth=1)
    axs[2].plot(infstate_rot_err, color="red", linewidth=1)
    axs[2].plot(bias_infstate_rot_err, color="blue", linewidth=1)
    axs[2].legend(["integration_rot_error", "AirIMU_rot_error", "Adapt_rot_error"])
    axs[2].grid(True)

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, save_prefix + file_name), dpi=300)
    plt.show()
    plt.close()


def visualize_state_error(
    save_prefix,
    relative_outstate,
    relative_infstate,
    save_folder=None,
    mask=None,
    file_name="state_error_compare.png",
):
    if mask is None:
        outstate_pos_err = relative_outstate["pos_dist"][0]
        outstate_vel_err = relative_outstate["vel_dist"][0]
        outstate_rot_err = relative_outstate["rot_dist"][0]

        infstate_pos_err = relative_infstate["pos_dist"][0]
        infstate_vel_err = relative_infstate["vel_dist"][0]
        infstate_rot_err = relative_infstate["rot_dist"][0]
    else:
        outstate_pos_err = relative_outstate["pos_dist"][0, mask]
        outstate_vel_err = relative_outstate["vel_dist"][0, mask]
        outstate_rot_err = relative_outstate["rot_dist"][0, mask]

        infstate_pos_err = relative_infstate["pos_dist"][0, mask]
        infstate_vel_err = relative_infstate["vel_dist"][0, mask]
        infstate_rot_err = relative_infstate["rot_dist"][0, mask]

    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("Integration error vs AirIMU Integration error")

    axs[0].plot(outstate_pos_err, color="b", linewidth=1)
    axs[0].plot(infstate_pos_err, color="red", linewidth=1)
    axs[0].legend(["integration_pos_error", "AirIMU_pos_error"])
    axs[0].grid(True)

    axs[1].plot(outstate_vel_err, color="b", linewidth=1)
    axs[1].plot(infstate_vel_err, color="red", linewidth=1)
    axs[1].legend(["integration_vel_error", "AirIMU_vel_error"])
    axs[1].grid(True)

    axs[2].plot(outstate_rot_err, color="b", linewidth=1)
    axs[2].plot(infstate_rot_err, color="red", linewidth=1)
    axs[2].legend(["integration_rot_error", "AirIMU_rot_error"])
    axs[2].grid(True)

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(os.path.join(save_folder, save_prefix + file_name), dpi=300)
    plt.show()
    plt.close()


def visualize_rotations(save_prefix, gt_rot, out_rot, inf_rot=None, save_folder=None):
    gt_euler = 180.0 / np.pi * pp.SO3(gt_rot).euler()
    outstate_euler = 180.0 / np.pi * pp.SO3(out_rot).euler()

    # legend_list = ["roll","pitch", "yaw"]
    # legend_list = ["yaw","pitch", "roll"]
    legend_list = ["pitch", "row", "yaw"]
    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("integrated orientation")
    for i in range(3):
        axs[i].plot(outstate_euler[:, i], color="b", linewidth=0.9)
        axs[i].plot(gt_euler[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["Integrated_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)

    if inf_rot is not None:
        infstate_euler = 180.0 / np.pi * pp.SO3(inf_rot).euler()
        print(infstate_euler.shape)
        for i in range(3):
            axs[i].plot(infstate_euler[:, i], color="red", linewidth=0.9)
            axs[i].legend(
                [
                    "Integrated_" + legend_list[i],
                    "gt_" + legend_list[i],
                    "AirIMU_" + legend_list[i],
                ]
            )
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + "_orientation_compare.png"), dpi=300
        )
    plt.show()
    plt.close()


def visualize_trajectory(save_prefix, save_folder, outstate, infstate):
    gt_x, gt_y, gt_z = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    rawTraj_x, rawTraj_y, rawTraj_z = torch.split(outstate["poses"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)

    fig, ax = plt.subplots()
    ax.plot(rawTraj_x, rawTraj_y, label="Raw")
    ax.plot(airTraj_x, airTraj_y, label="AirIMU")
    ax.plot(gt_x, gt_y, label="Ground Truth")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    plt.savefig(os.path.join(save_folder, save_prefix + "_trajectory_xy.png"), dpi=300)
    plt.show()
    plt.close()

    ###########################################################

    fig, ax = plt.subplots()
    ax.plot(rawTraj_x, rawTraj_z, label="Raw")
    ax.plot(airTraj_x, airTraj_z, label="AirIMU")
    ax.plot(gt_x, gt_z, label="Ground Truth")

    ax.set_xlabel("X axis")
    ax.set_ylabel("Z axis")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.savefig(os.path.join(save_folder, save_prefix + "_trajectory_xz.png"), dpi=300)
    plt.show()
    plt.close()

    ###########################################################

    fig, ax = plt.subplots()
    ax.plot(rawTraj_y, rawTraj_z, label="Raw")
    ax.plot(airTraj_y, airTraj_z, label="AirIMU")
    ax.plot(gt_y, gt_z, label="Ground Truth")

    ax.set_xlabel("Y axis")
    ax.set_ylabel("Z axis")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.savefig(os.path.join(save_folder, save_prefix + "_trajectory_yz.png"), dpi=300)
    plt.show()
    plt.close()

    ###########################################################

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    elevation_angle = 20  # Change the elevation angle (view from above/below)
    azimuthal_angle = 30  # Change the azimuthal angle (rotate around z-axis)

    ax.view_init(elevation_angle, azimuthal_angle)  # Set the view
    gt_x, gt_y, gt_z = gt_x.squeeze(), gt_y.squeeze(), gt_z.squeeze()
    print(gt_x)
    rawTraj_x, rawTraj_y, rawTraj_z = (
        rawTraj_x.squeeze(),
        rawTraj_y.squeeze(),
        rawTraj_z.squeeze(),
    )
    airTraj_x, airTraj_y, airTraj_z = (
        airTraj_x.squeeze(),
        airTraj_y.squeeze(),
        airTraj_z.squeeze(),
    )
    # Plotting the ground truth and inferred poses
    ax.plot(rawTraj_x.numpy(), rawTraj_y.numpy(), rawTraj_z.numpy(), label="Raw")
    ax.plot(airTraj_x.numpy(), airTraj_y.numpy(), airTraj_z.numpy(), label="AirIMU")
    ax.plot(gt_x.numpy(), gt_y.numpy(), gt_z.numpy(), label="Ground Truth")

    # Adding labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.legend()

    plt.savefig(os.path.join(save_folder, save_prefix + "_trajectory_3d.png"), dpi=300)
    plt.close()


def box_plot_wrapper(ax, data, edge_color, fill_color, **kwargs):
    bp = ax.boxplot(data, **kwargs)

    for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
        plt.setp(bp[element], color=edge_color)

    for patch in bp["boxes"]:
        patch.set(facecolor=fill_color)

    return bp


def plot_boxes(folder, input_data, metrics, show_metrics):
    fig, ax = plt.subplots(dpi=300)
    raw_ticks = [_ - 0.12 for _ in range(1, len(metrics) + 1)]
    air_ticks = [_ + 0.12 for _ in range(1, len(metrics) + 1)]
    label_ticks = [_ for _ in range(1, len(metrics) + 1)]

    raw_data = [input_data[metric + "(raw)"] for metric in metrics]
    air_data = [input_data[metric + "(AirIMU)"] for metric in metrics]

    # ax.boxplot(data, patch_artist=True, positions=ticks, widths=.2)
    box_plot_wrapper(
        ax,
        raw_data,
        edge_color="black",
        fill_color="royalblue",
        positions=raw_ticks,
        patch_artist=True,
        widths=0.2,
    )
    box_plot_wrapper(
        ax,
        air_data,
        edge_color="black",
        fill_color="gold",
        positions=air_ticks,
        patch_artist=True,
        widths=0.2,
    )
    ax.set_xticks(label_ticks)
    ax.set_xticklabels(show_metrics)

    # Create color patches for legend
    gold_patch = mpatches.Patch(color="gold", label="AirIMU")
    royalblue_patch = mpatches.Patch(color="royalblue", label="Raw")
    ax.legend(handles=[gold_patch, royalblue_patch])

    plt.savefig(os.path.join(folder, "Metrics.png"), dpi=300)
    plt.close()


def visualize_velocity(save_prefix, gtstate, outstate, refstate=None, save_folder=None):
    legend_list = ["x", "y", "z"]
    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("integrated orientation")
    for i in range(3):
        axs[i].plot(outstate[:, i], color="b", linewidth=0.9)
        axs[i].plot(gtstate[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["AirIO_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)
    
    if refstate is not None:
        for i in range(3):
            axs[i].plot(refstate[:, i], color="red", linewidth=0.9)
            axs[i].legend(
                [
                "AirIO_" + legend_list[i], 
                "gt_" + legend_list[i],
                "IOnet" + legend_list[i],
                ]
            )

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + ".png"), dpi=300
        )
    plt.show()
    plt.close()