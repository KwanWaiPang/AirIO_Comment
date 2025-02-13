# output the trajctory in the world frame for visualization and evaluation
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import argparse
import numpy as np
import pypose as pp
import tqdm

import torch
import torch.utils.data as Data
from utils import move_to

from pyhocon import ConfigFactory
from datasets import imu_seq_collate,SeqDataset, SeqInfDataset
 
from utils import CPU_Unpickler, integrate, interp_xyz
from utils.velocity_integrator import Velocity_Integrator, integrate_pos

from utils.visualize_state import visualize_velocity, visualize_rotations

from ekf import IMUEKF
from IMUstate import IMUstate
from matplotlib import pyplot as plt
from ekfutil import plot_bias_subplots, interp_xyz
import pdb

class SingleIMU(IMUstate):
    '''
    Take the
    state: R, V, P, bg, ba
    propogate(Ri, hat(Vi), Pi) - Pj;  hat(Vj) - Vj;
   
    observation: V
    '''
    def __init__(self):
        super().__init__()

    def state_transition(self, state: torch.Tensor, input: torch.Tensor, dt: torch.Tensor, t: torch.Tensor=None):
        '''
        Don't add noise in this function, as it will be used for automatically
        linearizing the system by the parent class ``pp.module.NLS``.
        '''
        # breakpoint()
        init_rot = pp.so3(state[..., :3]).Exp()
        bg = state[..., 9:12]
        ba = state[..., 12:15]

        w = (input[..., 0:3]-bg)
        a = (input[..., 3:6] - init_rot.Inv() @ self.gravity.double() -  ba)
        Dr = pp.so3(w * dt).Exp()
        Dv = Dr @ a * dt
        Dp = Dv * dt + Dr @ a * 0.5 * dt**2
        R = (init_rot @ Dr).Log() # torch.mul(state[..., :3], input[..., 0:3])
        V = state[..., 3:6] + init_rot @ Dv # R imu->world 
        P = state[..., 6:9] + state[..., 3:6] * dt + init_rot @ Dp

        return torch.cat([R, V, P, ba, bg], dim=-1).tensor()

    def observation(self, state, input, dt, t=None):
        '''
        state: k
        '''
        nstate = self.state_transition(state, input, dt)
        rot = pp.so3(nstate[..., :3]).Exp()
        velo = rot.Inv() @ nstate[..., 3:6]
        return velo
    
    def error(self, state, input, obs, dt):
        '''
        state: k
        '''
        return (self.observation(state, input, dt) - obs).norm()

class EKF_runner():
    def __init__(self, model, 
                 q = torch.ones(12, dtype=torch.float64) * 0.01, 
                 r = torch.ones(3, dtype=torch.float64) * 0.01, 
                 p = torch.zeros(15, dtype=torch.float64)):
        self.state = torch.zeros(15, dtype=torch.float64)
        self.model = model

        self.current_idx = 0
        self.state_lengeth = 9
        self.window = 1

        self.r = r
        self.q = q
        self.p = p

        Q = torch.eye(12, dtype=torch.float64) * self.q ** 2
        R = torch.eye(3, dtype=torch.float64) * self.r ** 2
        self.model = model
        self.filter = IMUEKF(self.model, Q=Q, R=R).double()

        self.P = torch.eye(15, dtype=torch.float64)*self.p**2
        self.P_list = [self.P]
        self.est_history = []
        self.gravity = torch.tensor([0., 0., 9.81007])
        self.data = None
    
    def get_result(self):
        return torch.stack(self.est_history), torch.stack(self.P_list)

    def propogate_update(self, imu_data, observation, Q=None, R=None):
        d_bias_gyro = self.state[9:12] # changes of the bias, it is zero in general
        d_bias_acc = self.state[12:15]
        input = torch.cat([imu_data["gyro"], imu_data["acc"], d_bias_gyro, d_bias_acc], dim=-1)
        error_before = self.model.error(self.state, input, observation, imu_data["dt"])
        self.state, self.P = self.filter(state = self.state, obs=observation, input = input, 
                                         P=self.P, dt=imu_data["dt"], Q = Q, R = R)
        error_after = self.model.error(self.state, input, observation, imu_data["dt"])
        # print("error_before", error_before, "error_after", error_after)
        
    def propogate_state(self, imu_data, Q=None):
        d_bias_gyro = self.state[9:12] # changes of the bias, it is zero in general
        d_bias_acc = self.state[12:15]
        input = torch.cat([imu_data["gyro"], imu_data["acc"], d_bias_gyro, d_bias_acc], dim=-1)
        self.state, self.P = self.filter.state_propogate(state = self.state, input = input, 
                                                         P=self.P, dt=imu_data["dt"], Q = Q)

    def run(self, imu_data, observation = None, Q=None, R=None):
        if observation is not None:
            self.propogate_update(imu_data, observation, Q=Q, R=R)
        else:
            self.propogate_state(imu_data, Q=Q)
            
        self.est_history.append(self.state.clone())
        self.P_list.append(self.P.clone())
        self.current_idx+=1

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--exp", type=str, default="experiments/euroc/motion_body", help="Path for AirIO netoutput")
    parser.add_argument("--airimu_exp", type=str, default=None, help="Path for AirIMU netoutput")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_body.conf", help="the configuration of the dataset")
    parser.add_argument("--savedir",type=str,default = "./EKFresult/loss_result",help = "Directory where the results wiil be saved")
    parser.add_argument("--usegtrot", action="store_true", help="Use ground truth rotation for gravity compensation")
    
    args = parser.parse_args(); 
    print(("\n"*3) + str(args) + ("\n"*3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference
    print(dataset_conf.keys())
    
    if args.exp is not None:
        net_result_path = os.path.join(args.exp, 'net_output.pickle')
        if os.path.isfile(net_result_path):
            with open(net_result_path, 'rb') as handle:
                inference_state_load = CPU_Unpickler(handle).load()
        else:
            raise Exception(f"Unable to load the network result: {net_result_path}")

    if args.airimu_exp is not None:
        airimu_ori_path = os.path.join(args.airimu_exp, 'net_output.pickle')
        if os.path.isfile(airimu_ori_path):
            with open(airimu_ori_path, 'rb') as handle:
                airimu_ori_load = CPU_Unpickler(handle).load()
        else:
            raise Exception(f"Unable to load the network result: {airimu_ori_path}")
        
        
    folder = args.savedir
    os.makedirs(folder, exist_ok=True)
    
    for data_conf in dataset_conf.data_list:
        print(data_conf)
        for data_name in data_conf.data_drive:
            print(data_conf.data_root, data_name)
            inference_state = airimu_ori_load[data_name] 
            dataset_inf = SeqInfDataset(data_conf.data_root, data_name, inference_state, device = args.device, name = data_conf.name,duration=1, step_size=1, drop_last=False, conf = dataset_conf)
            infloader = Data.DataLoader(dataset=dataset_inf, batch_size=1, 
                                            collate_fn=imu_seq_collate, 
                                            shuffle=False, drop_last=True)

            # dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=1, step_size=1, drop_last=False, conf = dataset_conf)
            
            sequential_dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=1, step_size=1, drop_last=False, conf = dataset_conf)
            
            init = dataset_inf.get_init_value()
            gravity = dataset_inf.get_gravity()
            integrator_outstate = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity=gravity,
                reset=False
            ).to(args.device).double()
            
            # dict_keys(['cov', 'net_vel', 'ts'])
            io_result = inference_state_load[data_name]
            # dict_keys(['acc_cov', 'gyro_cov', 'correction_acc', 'correction_gyro', 'corrected_acc', 'corrected_gyro', 'rot', 'dt'])
            
            model = SingleIMU().double()
            ekf = EKF_runner(model)
            
            # STEP 1 state initialization
            initial_state = torch.zeros(15, dtype=torch.float64)
            initial_state[:3] = init["rot"].Log()
            initial_state[3:6] = init["vel"]
            initial_state[6:9] = init["pos"]
            ekf.state = initial_state
            
            # STEO 2 covariance initialization
            # the uncertainty of the initial state
            io_index = 0
            air_imu_index = 0 
            gt_state = {"pos": [], "vel": [], "rot": []}
            
            t_range = tqdm.tqdm(dataset_inf)
            for data in t_range:
                # add the measurement of the airimu 
                imu_data = {"gyro": data["gyro"][0], "acc": data["acc"][0], "dt": data["dt"][0]}
                io_stamp = io_result["ts"][io_index]
                
                # STEP 2 add the learned uncertainty
                r = torch.ones(3, dtype=torch.float64) * 0.001
                q = torch.ones(12, dtype=torch.float64) *  0.01
                q[:3] = data["gyro_cov"][0]
                q[3:6] = data["acc_cov"][0]
                
                if io_stamp - data["timestamp"].abs() < 0.001:
                    observation = io_result["net_vel"][io_index]
                    r[:3] = io_result["cov"][0][io_index] * 0.1 ## Modify the scaling factor to balance uncertainties
                    io_index+=1
                else:
                    observation = None

                Q = torch.eye(12, dtype=torch.float64) * q 
                R = torch.eye(3, dtype=torch.float64) * r
                
                ekf.run(imu_data, observation=observation, Q=Q, R=R)
                gt_state["pos"].append(data["gt_pos"][0])
                gt_state["vel"].append(data["gt_vel"][0])
                gt_state["rot"].append(data["gt_rot"][0])
                
                # t_range.set_description(f"io_index: {io_index}, {io_stamp - data['timestamp']}")
                
            gtpos = torch.stack(gt_state["pos"])
            gtrot = torch.stack(gt_state["rot"])
            gtvel = torch.stack(gt_state["vel"])
            ekf_result, ekf_cov = ekf.get_result()
            ekf_result = ekf_result.numpy()
            
            if data_conf.name == "BlackBird":
                data_name = os.path.dirname(data_name).split('/')[1]
            else:
                data_name = data_name
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            np.save(os.path.join(folder, f"{data_name}_ekf_result.npy"), ekf_result)
            ekf_poses = np.zeros((ekf_result.shape[0], 8))
            ekf_poses[:, 1:4] = ekf_result[:, 6:9]
            ekf_poses[:, 4:] = pp.so3(ekf_result[:, :3]).Exp().numpy()
            np.save(os.path.join(folder, f"{data_name}_ekf_poses.npy"), ekf_poses)
            
            plt.figure()
            plt.plot(ekf_result[:, 6], ekf_result[:, 7], label="EKF")
            plt.plot(gtpos[:, 0], gtpos[:, 1], label="GT")
            
            plt.savefig(os.path.join(folder, f"{data_name}_ekf_result.png"))
            
            # visualize the net velocity
            io_ts = io_result["ts"][:,0]
            net_vel = io_result["net_vel"]
            interp_net_vel = interp_xyz(dataset_inf.data["time"], io_ts, net_vel.numpy())
            interp_net_vel = dataset_inf.data["gt_orientation"] @ interp_net_vel
            interp_net_vel = interp_net_vel[:len(ekf_result)].numpy()
            
            plot_bias_subplots(ekf_result[:, 9:12], title="EKF Bias", save_path=os.path.join(folder, f"{data_name}_bias.png"))
            visualize_rotations(f"EKF_rot_{data_name}", gtrot, pp.so3(ekf_result[:, :3]).Exp(), save_folder=folder)
            visualize_velocity(f"EKF_vel_{data_name}", gtvel, ekf_result[:, 3:6], interp_net_vel, save_folder=folder)
