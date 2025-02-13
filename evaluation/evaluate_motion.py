# output the trajctory in the world frame for visualization and evaluation
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import json
import argparse
import numpy as np
import pypose as pp
import tqdm

import torch
import torch.utils.data as Data
from utils import move_to

from pyhocon import ConfigFactory
from datasets import imu_seq_collate,SeqDataset
 
from utils import CPU_Unpickler, integrate, interp_xyz
from utils.velocity_integrator import Velocity_Integrator, integrate_pos

from utils.visualize_state import visualize_motion
 
def calculate_rte(outstate,duration, step_size):
    poses, poses_gt = outstate['poses'],outstate['poses_gt'][1:,:]

    dp = poses[:, duration-1:] - poses[:, :-duration+1]
    dp_gt = poses_gt[duration-1:] - poses_gt[:-duration+1]
    rte = (dp - dp_gt).norm(dim=-1)  
    return rte
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--exp", type=str, default="experiments/euroc/motion_body", help="Path for AirIO netoutput")
    parser.add_argument("--seqlen", type=int, default="1000", help="the length of the segment")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_body.conf", help="the configuration of the dataset")
    parser.add_argument("--savedir",type=str,default = "./result/loss_result",help = "Directory where the results wiil be saved")
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
    
    folder = args.savedir
    os.makedirs(folder, exist_ok=True)

    AllResults = []
    net_out_result = {}

    for data_conf in dataset_conf.data_list:
        print(data_conf)
        for data_name in data_conf.data_drive:
            print(data_conf.data_root, data_name)
            print("data_conf.dataroot", data_conf.data_root)
            print("data_name", data_name)
            print("data_conf.name", data_conf.name)

            dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
            loader = Data.DataLoader(dataset=dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False, drop_last=False)
            init = dataset.get_init_value()
            gravity = dataset.get_gravity()
            integrator_outstate = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity=gravity,
                reset=False
            ).to(args.device).double()
            
            integrator_reset = pp.module.IMUPreintegrator(
                init['pos'], init['rot'], init['vel'],gravity = gravity,
                reset=True
            ).to(args.device).double()
            
            outstate = integrate(
                integrator_outstate, loader, init, 
                device=args.device, gtinit=False, save_full_traj=True,
                use_gt_rot=args.usegtrot
            )
            relative_outstate = integrate(
                integrator_reset, loader, init, 
                device=args.device, gtinit=True,
                use_gt_rot=args.usegtrot
            )
            
            if args.exp is not None:
                motion_dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
                motion_loader = Data.DataLoader(dataset=motion_dataset, batch_size=1, collate_fn=imu_seq_collate, shuffle=False, drop_last=False)
            
                inference_state = inference_state_load[data_name] 
                gt_ts =  motion_dataset.data['time']
                vel_ts = inference_state['ts']
                indices = torch.cat([torch.where(gt_ts == item)[0] for item in vel_ts[:,0]]).to(torch.int32)

                if "coordinate" in dataset_conf.keys():
                    print("*************",dataset_conf["coordinate"],"*************")
                    if dataset_conf["coordinate"] == "body_coord":
                        rotation = motion_dataset.data['gt_orientation'] # Set data['gt_orientation'] using AirIMU or gt-truth in the dataconf

                        vel_dist = rotation[indices,:] * inference_state['net_vel'] - motion_dataset.data['velocity'][indices,:]  
                        net_vel = interp_xyz(gt_ts, vel_ts[:,0],  inference_state['net_vel'])
                        net_vel = rotation * net_vel
                    
                    if dataset_conf["coordinate"] == "glob_coord":
                        vel_dist = inference_state['net_vel'] - motion_dataset.data['velocity'][indices,:]    
                        net_vel = interp_xyz(gt_ts, vel_ts[:,0], inference_state['net_vel'])

                if data_conf.name == "BlackBird":
                    save_prefix = os.path.dirname(data_name).split('/')[1]
                else:
                    save_prefix = data_name
               
                dt = gt_ts[1:] - gt_ts[:-1]
                data_inte = {"vel":net_vel,'dt':dt}
                
                integrator_vel =Velocity_Integrator(
                    init['pos']).to(args.device).double()
                
                inf_outstate =integrate_pos(
                    integrator_vel, data_inte, init, motion_dataset,
                    device=args.device
                )
                inf_rte = calculate_rte(inf_outstate, args.seqlen,args.seqlen)

                #save loss result
                result_dic = {
                    'name': data_name,      
                    'ATE':torch.sqrt((inf_outstate['pos_dist']**2).mean()).item(),
                    'AVE':inf_outstate['vel_dist'].mean().item(),
                    'RP_RMSE': np.sqrt((inf_rte**2).mean()).numpy().item(),
                    }
                
                AllResults.append(result_dic)
                
                print("==============Integration==============")
                print("outstate:")
                print("pos_err: ", outstate['pos_dist'].mean())
                print("rte",relative_outstate['vel_dist'].mean())
                
                print("==============AirIO==============")
                print("infstate:")
                print("pos_err: ", inf_outstate['pos_dist'].mean())
                print("rte",inf_rte.mean())

            visualize_motion(save_prefix, folder,outstate,inf_outstate)

        file_path = os.path.join(folder, "result.json")
        with open(file_path, 'w') as f: 
            json.dump(AllResults, f, indent=4)
