import torch
from .loss_func import loss_fc_list, diag_ln_cov_loss
from utils import report_hasNan
import numpy as np

def motion_loss_(fc, pred, targ):
    dist = pred - targ
    loss = fc(dist)
    return loss, dist

def get_motion_loss(inte_state, label, confs):
    ## The state loss for evaluation
    loss, cov_loss = 0, {}
    loss_fc = loss_fc_list[confs.loss]
    
    vel_loss, vel_dist = motion_loss_(loss_fc, inte_state['net_vel'],label)

    # Apply the covariance loss
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()

        if "covaug" in confs and confs["covaug"] is True:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist, cov)
        else:
            vel_loss += confs.cov_weight * diag_ln_cov_loss(vel_dist.detach(), cov)
    loss += confs.weight * vel_loss
    return {'loss':loss, 'cov_loss':cov_loss}


def get_motion_RMSE(inte_state, label, confs):
    '''
    get the RMSE of the last state in one segment
    '''
    def _RMSE(x):
        return torch.sqrt((x.norm(dim=-1)**2).mean())
    cov_loss = 0
    dist = (inte_state['net_vel'] - label)
    dist = torch.mean(dist,dim=-2)
    loss = _RMSE(dist)[None,...]
    
    if confs.propcov:
        #velocity covariance.
        cov = inte_state['cov']
        cov_loss = cov.mean()
    
    return {'loss': loss, 
            'dist': dist.norm(dim=-1).mean(),
            'cov_loss': cov_loss}
