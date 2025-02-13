import torch
import math
def imu_seq_collate(data):
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    gt_rot = torch.stack([d["gt_rot"] for d in data])
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    init_rot = torch.stack([d["init_rot"] for d in data])
    init_vel = torch.stack([d["init_vel"] for d in data])

    dt = torch.stack([d["dt"] for d in data])

    return {
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "gt_rot": gt_rot,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_rot": init_rot,
    }


def custom_collate(data):
    dt = torch.stack([d["dt"] for d in data])
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])
    rot = torch.stack([d["rot"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    gt_rot = torch.stack([d["gt_rot"] for d in data])
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    init_rot = torch.stack([d["init_rot"] for d in data])
    init_vel = torch.stack([d["init_vel"] for d in data])

    return (
        {
            "dt": dt,
            "acc": acc,
            "gyro": gyro,
            "rot": rot,
        },
        {
            "pos": init_pos,
            "vel": init_vel,
            "rot": init_rot,
        },
        {
            "gt_pos": gt_pos,
            "gt_vel": gt_vel,
            "gt_rot": gt_rot,
        },
    )

def motion_collate_data(data):
    timestamp = None
    timestamp = [d['timestamp'] for d in data if 'timestamp' in d]
    if timestamp:
        timestamp = torch.stack(timestamp)
    acc = torch.stack([d['acc'] for d in data])
    gyro = torch.stack([d['gyro'] for d in data])
    rot = torch.stack([d['rot'] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    gt_rot = torch.stack([d["gt_rot"] for d in data])
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    init_rot = torch.stack([d["init_rot"] for d in data])
    init_vel = torch.stack([d["init_vel"] for d in data])

    dt = torch.stack([d['dt'] for d in data])

    return (
        {
            'ts': timestamp,
            "dt": dt,
            "acc": acc,
            "gyro": gyro,
            "rot": rot,
        },
        {
            "pos": init_pos,
            "vel": init_vel,
            "rot": init_rot,
        },
        {
            "gt_pos": gt_pos,
            "gt_vel": gt_vel,
            "gt_rot": gt_rot,
        },
    )
    
def motion_collate(data, **kwargs):
    input_data, init_state, label = motion_collate_data(data)
    if len(kwargs) > 0:
        # TODO: Implement data augmentation if needed
        pass  
    return input_data, init_state, label

    
collate_fcs = {
    "base": custom_collate,
    'motion': motion_collate,
}
