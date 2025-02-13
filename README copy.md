# <div align="center">AirIO: Learning Inertial Odometry with Enhanced<br/> IMU Feature Observability<br/>[![Homepage](https://img.shields.io/badge/Homepage-4385f4?style=flat&logo=googlehome&logoColor=white)](https://air-io.github.io/) [![arXiv](https://img.shields.io/badge/arXiv-b31b1b?style=flat&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2501.15659) [![YouTube](https://img.shields.io/badge/YouTube-ff0000?style=flat&logo=youtube&logoColor=white)]()
![Introduction](img/blackbird_github.gif)

## 📦 Installation & Dataset Setup  
### Environment Installation
 See `requirements.txt` for environment requirements. 
 
This work is based on pypose. For installation and advanced usage, follow the official PyPose instructions [here](https://github.com/pypose/pypose).

### Download Datasets
Euroc dataset: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

Blackbird dataset:  
- Download from the official dataset website: http://blackbird-dataset.mit.edu/BlackbirdDatasetData/
- Download the specific sequences used in our work: [here](https://github.com/Air-IO/Air-IO/releases/download/datasets/Blackbird.zip)

Our custom simulation Pegasus datasets: [here](https://github.com/Air-IO/Air-IO/releases/download/datasets/PegasusDataset.zip)

### Get Pre-trained models & results

> [!NOTE]
> To run the EKF, you need to download the AirIMU results for IMU preintegration to provide integration and uncertainty. 
> We also provide a conventional mode that uses ground truth orientation as input, which can be used if you just want to test the network.

| Datasets               | AirIO Pre-trained Models & Results| AirIMU Pre-trained Models & Results | 
| :----------------: | :------: | :----: |
| EuRoC       |  [AirIO Model](https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_EuRoC.zip)   | [AirIMU](https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_EuRoC.zip)|
| Blackbird           |   [AirIO Model](https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_Blackbird.zip)  | [AirIMU](https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_blackbird.zip)|
| Pegasus    |  [AirIO Model](https://github.com/Air-IO/Air-IO/releases/download/AirIO/AirIO_Pegasus.zip)   | [AirIMU](https://github.com/Air-IO/Air-IO/releases/download/AirIMU/AirIMU_pegasus.zip)|

Each AirIMU Results pickle file contains raw IMU correction; Each Orientations pickle file contains two critical keys: `airimu_rot` for AirIMU-corrected orientation, `inte_rot` for raw IMU integrated orientation.

If you want to train a your own model for AirIMU please check [https://airimu.github.io](https://airimu.github.io/)

## 🚀 Quick Start: Run with Default Configurations
Immediately test the pipeline using pre-defined settings. 
### Training

To start training with the default configuration file:
```
python train_motion.py --config configs/EuRoC/motion_body_rot.conf
```
Remember to reset the `data_root` in `configs/datasets/EuRoC/Euroc_body.conf` to your EuRoC dataset directory.

### Run Inference
To perform inference using the default configuration file:
```
python inference_motion.py --config configs/EuRoC/motion_body_rot.conf
```
Network predictions will be saved as a `net_output.pickle` file
> [!NOTE]
> AirIO network supports three orientation input modes: \
> **Default**: using Ground-truth orientation (no setup required) \
> **Switch modes**: modify the ```rot_type``` and ```rot_path``` in the dataset config file ```configs/datasets/EuRoC/Euroc_body.conf```. You can use AirIMU-corrected Orientation (```rot_type: airimu```)  or raw IMU preintegration orientation (```rot_type: integration```). Download precomputed rotation files (e.g. orientation_output.pickle) from the Download Datasets section and update the ```rot_path```.


### Evaluate & Visualize Network Predictions
Run the following command to assess the network's motion estimation and plot the trajectories:
```
python evaluation/evaluate_motion.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp experiments/euroc/motion_body_rot \
    --seqlen 1000
```
`--seq` is the segment length (in frames) for RTE calculation

### EKF example
```
python EKF/IMUofflinerunner.py \
      --dataconf configs/datasets/EuRoC/Euroc_body.conf \
      --exp experiments/euroc/motion_body_rot\
      --airimu_exp ${AirIMU_RESULT_PATH}
```
EKF results will be saved as `${SEQUENCE_NAME}_ekf_poses.npy` and `${SEQUENCE_NAME}_ekf_results.npy`
> [!NOTE]  
> Replace ${AirIMU_RESULT_PATH} with the directory containing AirIMU result pickle files.
### Evaluate & Visualize EKF Results
```
python evaluation/evaluate_ekf.py \
    --dataconf configs/datasets/EuRoC/Euroc_body.conf \
    --exp EKFresult/loss_result \
    --seqlen 1000
```

## 🛠️ Custom Configuration Guide    
- Modifying Configuration files
- Adding custom datasets  
- Modifying new network architectures

### Modifying Configuration files
Setting two types of configuration files before training: the config file for datasets and the training parameters.

You can place your custom dataset config file in the ```configs/datasets/YOUR_CUSTOM_DATASET/``` directory and the training config file in the ```configs/YOUR_CUSTOM_DATASET/``` directory.
```plaintext
└── configs/
    ├── datasets # Dataset configurations
    |   ├── EuRoC
    |   |    ├── EuRoC_body.conf # body-frame representation
    |   |    ├── EuRoC_global.conf 
    |   |    └── EuRoC_global_rmg.conf # global-frame representation without gravity term
    |   ├── Blackbird
    |   ├── Pegasus
    |   └── YOUR_CUSTOM_DATASET
    |   |    └── ${YOUR_CUSTOM_DATASET_SETTING}.conf 
    |   |    
    ├── EuRoC # Training configurations for EuRoC
    |   ├──motion_body_rot.conf # training with body-frame representation + Attitude Encoding
        └──motion_body.conf     # training with body-frame representation
    ├── Blackbird
    ├── Pegasus
    └── YOUR_CUSTOM_DATASET
        └── ${YOUR_CUSTOM_TRAINING_SETING}.conf 
```

**1. Dataset Configuration files** 

```configs/datasets/YOUR_CUSTOM_DATASET/${YOUR_CUSTOM_DATASET_SETTING}.conf```

Below is a snippet of the training part demonstrating some of the key parameters:
```yaml
train:
{    
    mode: train              # The current mode is "train". modes can be: test (for validation), evaluate (for evaluation), inference/infevaluate (for inference)
    coordinate: body_coord   # The network will process body-frame IMU as input and predict the body-frame velocity.
                             # You can also use  "glob_coord" to process IMU data in the global frame.
    remove_g:False           # Whether to remove the gravity component from IMU data. The default is False.
    rot_type: airimu         # Only using in the inference phase, indicating the rotation data comes from the AirIMU network.
                             # You can also use "integration" to use the raw IMU preintegration rotation. The default is None, using ground-truth rotation.
    rot_path: PATH_TO_AIRIMU_NETOUTPUT # Path to the AirIMU corrected rotation or raw IMU preintegration rotation. The default is None, using ground-truth rotation.
    data_list:
    [{
        name: YourCustomDataset    #  Corresponding to the dataset class. Create your own dataset class in datasets/_dataset.py
        window_size: 1000          #  The size of the data window is 1000 frames.
        step_size: 10              #  The window slides forward by 10 frames.
        data_root: DATA_ROOT       #  Path to dataset (REPLACE THIS WITH YOUR DATASET DIRECTION)
        data_drive: [sequence_1, sequence_name_2] #  The dataset sequences to use for training. 
        },
    ]
    gravity: 9.81007 # the gravity value used in the dataset
}

```
> [!NOTE]
> For testing, evaluation, or inference, make similar changes for the corresponding sections (```test:```, ```eval:```, ```inference:```) in the same config file.

**2. Training Configuration files** 

```configs/YOUR_CUSTOM_DATASET/${YOUR_CUSTOM_TRAINING_SETING}.conf```
Below is a snippet demonstrating some of the key parameters you need to change:
```yaml
# Include Dataset Configuration:
dataset:
{
   include "../datasets/YOUR_CUSTOM_DATASET/${YOUR_CUSTOM_DATASET_SETTING}.conf" # Path to the dataset configuration file
   collate: {type: motion}
}
# Training Parameters
train:
{
   network: codewithrot  # Specify the network architecture.
                         # - "codewithrot": For attitude encoding.
                         # - "codenetmotion": For training without attitude encoding.
   lr: 1e-3              # Initial learning rate
   min_lr: 1e-5          # Minimum learning rate (for learning rate scheduling)
   batch_size: 128
   max_epoches: 100
   patience: 5           # used in ReduceLROnPlateau scheduler
   factor: 0.2           # used in ReduceLROnPlateau scheduler
   weight_decay: 1e-4    # Weight decay used in the Adam optimizer
}

```
### Adding custom datasets
To add a custom dataset, you need to create a new dataset class in ```datasets/_dataset.py``` that follows the same structure as the Euroc class (```datasets/EuRoCdataset.py```).
> [!NOTE]
> The class name must match the name field in the ```data_list``` section of your dataset configuration file. 

**1. Loading and Interpolating Data**
The dataset should provide the following features in self.data
```python
self.data = {
    "dt": time intervals,
    "time": timestamps,
    "acc": accelerometer_data, # body-frame tensor
    "gyro": gyroscope_data, # body-frame tensor
    "gt_orientation" : ground-truth orientation, # pypose.lietensor.lietensor.SO3Type object
    "gt_translation" : ground-truth position, # tensor
    "velocity" : ground-truth velocity, # tensor
}
```
**2. Subsequent Steps** 

After loading and interpolating the data, using the following functions to prepare the dataset for training or evaluation:
```python
# Set orientation
self.set_orientation(rot_path, data_name, rot_type) # Load orientation data according to the dataset config file (e.g., from AIRIMU or preintegrated or ground truth)

# Transform Coordinates:
self.update_coordinate(coordinate, mode) # Perform body-frame or global-frame representation according to the dataset config file

# Remove Gravity term
self.remove_gravity(remove_g) # Remove the gravity component from accelerometer data if required in the dataset config file.
```

### Modifying new network architectures

Open ```model/code.py```
- You can modify the network class (e.g., CodeNetMotion or CodeNetMotionwithRot). 
- Define your custom network class and register the new network in the net_dict dictionary in the ```model/__init__.py```:
```python
net_dict = {
    'codenetmotion': CodeNetMotion,  # Existing network
    'codewithrot': CodeNetMotionwithRot,  # Existing network
    'your_network_name': YourCustomNetworkClass,  # Add your custom network here
}
```
- Use your custom network by specifying the network name in the training config file:
```yaml
train:
{
    network: your_network_name 
}
```

### Run with your custom config:
```
# train
python train_motion.py --config configs/DATASET_NAME/DATASET_NAME_training_setting.conf

# inference
python inference_motion.py --config configs/DATASET_NAME/DATASET_NAME_training_setting.conf
```


Citation
--------
```
@misc{qiu2025airiolearninginertialodometry,
      title={AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability}, 
      author={Yuheng Qiu and Can Xu and Yutian Chen and Shibo Zhao and Junyi Geng and Sebastian Scherer},
      year={2025},
      eprint={2501.15659},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2501.15659}, 
}

```
