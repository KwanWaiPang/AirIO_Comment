# output the trajctory in the world frame for visualization and evaluation
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import os
import json
import argparse
import numpy as np
import pypose as pp
import tqdm
from matplotlib.gridspec import GridSpec
import torch
import torch.utils.data as Data
from utils import move_to

from pyhocon import ConfigFactory
from datasets import imu_seq_collate,SeqDataset
 
from utils import CPU_Unpickler, integrate

import matplotlib.pyplot as plt

def calculate_rte(outstate,duration, step_size):
    poses, poses_gt = outstate['poses'],outstate['poses_gt'][1:,:]

    dp = poses[duration-1:] - poses[:-duration+1]
    dp_gt = poses_gt[duration-1:] - poses_gt[:-duration+1]
    
    rte = (dp - dp_gt).norm(dim=-1)  
    return rte
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
    parser.add_argument("--seqlen", type=int, default="1000", help="the length of the segment")
    parser.add_argument("--dataconf", type=str, default="configs/datasets/EuRoC/Euroc_global.conf", help="Path to the dataset configuration file. ")
    parser.add_argument("--savedir",type=str,default = "./result/loss_result",help = "Directory where the results wiil be saved")
    parser.add_argument("--exp", type=str, default="./EKFresult/loss_result", help="Path for the EKF results")
    args = parser.parse_args(); 
    print(("\n"*3) + str(args) + ("\n"*3))
    config = ConfigFactory.parse_file(args.dataconf)
    dataset_conf = config.inference
    print(dataset_conf.keys())


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

            motion_dataset = SeqDataset(data_conf.data_root, data_name, args.device, name = data_conf.name, duration=args.seqlen, step_size=args.seqlen, drop_last=False, conf = dataset_conf)
            
            if args.exp is not None:
                
                ekf_result = np.load(os.path.join(args.exp, data_name+"_ekf_result.npy"))
                ekf_pos =torch.tensor(ekf_result[:,6:9])
                
                inf_state = {}
                inf_state['poses'] = ekf_pos
                inf_state['poses_gt'] = motion_dataset.data['gt_translation']
               
                pos_dist = (inf_state['poses'] -inf_state['poses_gt'][1:,:]).norm(dim=-1)
                inf_rte = calculate_rte(inf_state, args.seqlen,args.seqlen)

                #save loss result
                result_dic = {
                    'name': data_name,
                    'ATE(EKF)':torch.sqrt((pos_dist**2).mean()).item(),
                    'RTE(EKF)':inf_rte.mean().item(),
                    'RP_RMSE(EKF)': np.sqrt((inf_rte**2).mean()).numpy().item(),
                    }
                
                AllResults.append(result_dic)

                print("==============EKF==============")
                print("infstate:")
                print("pos_err: ", pos_dist.mean())
                print("rte",inf_rte.mean())

        file_path = os.path.join(folder, "ekf_result.json")
        with open(file_path, 'w') as f: 
            json.dump(AllResults, f, indent=4)
