"""
Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/master/src/learning/data_management/prepare_datasets/blackbird.py
"""
import os

import numpy as np
import pypose as pp
import torch
import copy
from utils import lookAt, qinterp, Gaussian_noise
from .dataset import Sequence
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
import pickle
import json

class BlackBird(Sequence):
    def __init__(
        self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.81007, 
        remove_g=False,
        **kwargs
    ):
        super(BlackBird, self).__init__()
        (
            self.data_root,
            self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(), None, None, None, None, None)
        
        self.g_vector = torch.tensor([0, 0, gravity],dtype=torch.double)
        data_path = os.path.join(data_root, data_name)
        self.load_imu(data_path, data_name)
        self.load_gt(data_path)
        self.refer_IMO()
        
        # when evaluation: load airimu or integrated orientation:
        self.set_orientation(rot_path, data_name, rot_type)
        
        # transform to global/body frame:
        self.update_coordinate(coordinate, mode)
        
        # remove gravity term
        self.remove_gravity(remove_g)

    def refer_IMO(self):
        # the provided ground truth is the drone body in the NED vicon frame
        # rotate to have z upwards
        R_w_ned = np.array([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.]])
        t_w_ned = np.array([0., 0., 0.])

        # rotate from body to imu frame
        R_b_i = np.array([
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 1.]])
        t_b_i = np.array([0., 0., 0.])
        
        raw_imu = np.asarray(self.imu_data)
        thrusts = np.asarray(self.thrusts)

        data_tmp = self.gt_data


        data = []
        for data_i in data_tmp:
            ts_i = data_i[0] / 1e6
            
            t_i = data_i[1:4]
            R_i = Rotation.from_quat(
                np.array([data_i[5], data_i[6], data_i[7], data_i[4]])).as_matrix()

            # transform to world frame
            R_it = R_w_ned @ R_i
            t_it = t_w_ned + R_w_ned @ t_i

            # transform to imu frame
            t_it = t_it + R_it @ t_b_i
            R_it = R_it @ R_b_i

            q_it = Rotation.from_matrix(R_it).as_quat()
            d = np.array([
                ts_i,
                t_it[0], t_it[1], t_it[2],
                q_it[0], q_it[1], q_it[2], q_it[3]
            ])
            data.append(d)
        data = np.asarray(data)

        # include velocities
        gt_times = data[:, 0] 
        gt_pos = data[:, 1:4]

        # compute velocity
        v_start = ((gt_pos[1] - gt_pos[0]) / (gt_times[1] - gt_times[0])).reshape((1, 3))
        gt_vel_raw = (gt_pos[1:] - gt_pos[:-1]) / (gt_times[1:] - gt_times[:-1])[:, None]
        gt_vel_raw = np.concatenate((v_start, gt_vel_raw), axis=0)
        # filter
        gt_vel_x = np.convolve(gt_vel_raw[:, 0], np.ones(5) / 5, mode='same')
        gt_vel_x = gt_vel_x.reshape((-1, 1))
        gt_vel_y = np.convolve(gt_vel_raw[:, 1], np.ones(5) / 5, mode='same')
        gt_vel_y = gt_vel_y.reshape((-1, 1))
        gt_vel_z = np.convolve(gt_vel_raw[:, 2], np.ones(5) / 5, mode='same')
        gt_vel_z = gt_vel_z.reshape((-1, 1))
        gt_vel = np.concatenate((gt_vel_x, gt_vel_y, gt_vel_z), axis=1)

        gt_traj_tmp = np.concatenate((data, gt_vel), axis=1)  # [ts x y z qx qy qz qw vx vy vz]

        # In Blackbird dataset, the sensors measurements are at:
        # 100 Hz IMU meas.
        # 180 Hz RPM meas.
        # 360 Hz Vicon meas.
        # resample imu at exactly 100 Hz
        t_curr = raw_imu[0, 0]
        dt = 0.01
        new_times_imu = [t_curr]
        while t_curr < raw_imu[-1, 0] - dt - 0.001:
            t_curr = t_curr + dt
            new_times_imu.append(t_curr)
        new_times_imu = np.asarray(new_times_imu)
        gyro_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 1:4], axis=0)(new_times_imu)
        accel_tmp = interp1d(raw_imu[:, 0], raw_imu[:, 4:7], axis=0)(new_times_imu)
        raw_imu = np.concatenate((new_times_imu.reshape((-1, 1)), gyro_tmp, accel_tmp), axis=1)

        # We down sample to IMU rate
        times_imu = raw_imu[:, 0]
        # get initial and final times for interpolations
        idx_s = 0
       
        for ts in times_imu:
            if ts > gt_traj_tmp[0, 0] and ts > thrusts[0, 0]:
                break
            else:
                idx_s = idx_s + 1
        assert idx_s < len(times_imu)

        idx_e = len(times_imu) - 1
        for ts in reversed(times_imu):
            if ts < gt_traj_tmp[-1, 0] and ts < thrusts[-1, 0]:
                break
            else:
                idx_e = idx_e - 1
        assert idx_e > 0

        idx_e = len(times_imu) - 1
        for ts in reversed(times_imu):
            if ts < gt_traj_tmp[-1, 0]:
                break
            else:
                idx_e = idx_e - 1
        assert idx_e > 0

        times_imu = times_imu[idx_s:idx_e + 1]
 
 
        raw_imu = raw_imu[idx_s:idx_e + 1]
        self.data["gyro"] = torch.tensor(raw_imu[:, 1:4])
        self.data["acc"] = torch.tensor(raw_imu[:, 4:])

        # interpolate ground-truth samples at thrust times
        self.data["gt_translation"]  = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 1:4], axis=0)(times_imu)
        self.data["gt_orientation"] = Slerp(gt_traj_tmp[:, 0], Rotation.from_quat(gt_traj_tmp[:, 4:8]))(times_imu).as_quat()
        self.data["velocity"] = interp1d(gt_traj_tmp[:, 0], gt_traj_tmp[:, 8:11], axis=0)(times_imu)
        
        
        self.data["gt_translation"]  = torch.tensor(self.data["gt_translation"])
        self.data["gt_orientation"] = pp.SO3(torch.tensor(self.data["gt_orientation"]))
        self.data["velocity"] = torch.tensor(self.data["velocity"])
        self.data["gt_time"] = torch.tensor(times_imu)
        self.data["time"] = torch.tensor(times_imu)
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

    def get_length(self):
        return self.data["time"].shape[0]
    

    def load_imu(self, folder,data_name=None):
        imu_data = np.loadtxt(
            os.path.join(folder, "imu_data.csv"), dtype=float, delimiter=","
        )
        thrusts = np.loadtxt(
            os.path.join(folder, "thrust_data.csv"), dtype=float, delimiter=","
        )
        self.imu_data = copy.deepcopy(imu_data)
        self.thrusts = copy.deepcopy(thrusts)



    def load_gt(self, folder):
        gt_data = np.loadtxt(
            os.path.join(folder, "groundTruthPoses.csv"),
            dtype=float,
            delimiter=",",
        )
        self.gt_data = copy.deepcopy(gt_data)
    

    def update_coordinate(self, coordinate, mode):
        """
        Updates the data (imu / velocity) based on the required mode.
        :param coordinate: The target coordinate system ('glob_coord' or 'body_coord').
        :param mode: The dataset mode, only rotating the velocity during training. 
        """
        if coordinate is None:
            print("No coordinate system provided. Skipping update.")
            return
        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode != "infevaluate" and mode != "inference":
                    self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            print("An error occurred while updating coordinates:", e)
            raise e

    def set_orientation(self, exp_path, data_name, rotation_type):
        """
        Sets the ground truth orientation based on the provided rotation.
        :param exp_path: Path to the pickle file containing orientation data.
        :param rotation_type: The type of rotation within the pickle file (AirIMU corrected orientation / raw imu Pre-integration).
        """
        if rotation_type is None or rotation_type == "None" or rotation_type.lower() == "gtrot":
            return
        try:
            with open(exp_path, 'rb') as file:
                loaded_data = pickle.load(file)
            state = loaded_data[data_name]
            

            if rotation_type.lower() == "airimu":
                self.data["gt_orientation"] = state['airimu_rot']
            elif rotation_type.lower() == "integration":
                self.data["gt_orientation"] = state['inte_rot']
            else:
                print(f"Unsupported rotation type: {rotation_type}")
                raise ValueError(f"Unsupported rotation type: {rotation_type}")
        except FileNotFoundError:
            print(f"The file {experiment_path} was not found.")
            raise
    
    def remove_gravity(self,remove_g):
        if remove_g is True:
                print("gravity has been removed")
                self.data["acc"] -= self.g_vector