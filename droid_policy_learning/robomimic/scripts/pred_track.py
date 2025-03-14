import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.algo import RolloutPolicy
import h5py
import fpsample
import numpy as np
import cv2
import sys
from scipy.spatial.transform import Rotation as R
import argparse
import random
import json
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def ext2mat(ext):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.from_euler("xyz", ext[3:]).as_matrix()
    extrinsic[:3, 3] = ext[:3]
    return extrinsic

def transform_point_cloud(points_camera, transformation_matrix):
    xyz_camera = points_camera[:, :3]
    xyz_camera_homogeneous = np.hstack((xyz_camera, np.ones((xyz_camera.shape[0], 1))))
    transformation_matrix_camera_to_world = transformation_matrix
    xyz_world_homogeneous = np.matmul(transformation_matrix_camera_to_world, xyz_camera_homogeneous.T).T
    xyz_world = xyz_world_homogeneous[:, :3]
    points = np.hstack((xyz_world, points_camera[:, 3:]))
    return points

def points_from_camera_name(f, name, i):
    xyz = np.array(f['observation/camera/pointcloud/{}_image'.format(name)][i])
    rgb = np.array(f['observation/camera/image/{}_image'.format(name)][i])
    rgb = rgb / 255.0
    extrinsics = np.array(f['observation/camera/extrinsics/{}'.format(name)][i])
    points = np.concatenate([xyz, rgb.reshape(-1,3)], axis=1)
    points = points[~np.isnan(points).any(axis=1)]
    points = points[~np.isinf(points).any(axis=1)]
    extrinsics = ext2mat(extrinsics)
    points = transform_point_cloud(points, extrinsics)
    mask = (np.abs(points[:, 0]) < 1.0 ) & (np.abs(points[:, 1]) < 1.0) & (np.abs(points[:, 2]) < 1.0)
    points = points[mask]
    # index = fpsample.fps_sampling(points[:,:3], 2000)
    # points = points[index]
    return points

def pred_track(f_in, policy, f_out, s=40):
    assert isinstance(policy, RolloutPolicy)
    # open input file
    f = h5py.File(f_in, 'r')

    # open output file
    f_out = h5py.File(f_out, 'w')

    policy.start_episode()
    
    obs_dict = f['observation']
    
    H = obs_dict['robot_state/cartesian_position'].shape[0]
    
    obs_list = []
    
    for i in range(H):
        obs = {}
        for key in [
                    "robot_state/cartesian_position",
                    "robot_state/gripper_position",
                    "lang_fixed/language_distilbert",
                    "camera/pointcloud/hand_camera_left_pcd",
                    "camera/pointcloud/varied_camera_1_left_pcd",
                    "camera/pointcloud/varied_camera_2_left_pcd"
                    ]:
            obs[key] = np.array(obs_dict[key][i])
            # obs[key] = np.expand_dims(obs[key], axis=0)
        obs_list.append(obs)
    
    for key in obs_list[0]:
        print(key, obs_list[0][key].shape)
    
    # stacked_obs_list = []
    
    
    
    # for i in range(H):
    #     stacked_obs = {}
    #     for key in obs_list[i]:
    #         if i == 0:
    #             stacked_obs[key] = np.concatenate([obs_list[i][key], obs_list[i][key]], axis=0)
    #         else:
    #             stacked_obs[key] = np.concatenate([obs_list[i-1][key], obs_list[i][key]], axis=0)
    #     stacked_obs_list.append(stacked_obs)
        
    # for key in stacked_obs_list[0]:
    #     print(key, stacked_obs_list[0][key].shape)
    
    track = policy(obs_list[s]).reshape(10, 3)
    
    print('track', track.shape)
    
    pointcloud = np.concatenate([points_from_camera_name(f, 'hand_camera_left', s), points_from_camera_name(f, 'varied_camera_1_left', s), points_from_camera_name(f, 'varied_camera_2_left', s)], axis=0)
    
    gt_track = np.array(f['action/track'])[s].reshape(10, 3)
    
    print('gt_track', gt_track.shape)
    
    color = np.array([1.0, 0.0, 0.0])
    track = np.concatenate([track, np.tile(color, (10, 1))], axis=1)

    pointcloud = np.concatenate([pointcloud, track], axis=0)
    
    color = np.array([0.0, 1.0, 0.0])
    gt_track = np.concatenate([gt_track, np.tile(color, (10, 1))], axis=1)
    
    pointcloud = np.concatenate([pointcloud, gt_track], axis=0)
    
    gripper = np.array(f['observation/robot_state/cartesian_position'])[10][:3]
    
    color = np.array([0.0, 0.0, 1.0])
    
    ## 在pointcloud中加入50个点，每个点在girpper附近，颜色为蓝色
    for i in range(50):
        gripper_n = gripper + np.random.normal(0, 0.01, 3)
        gripper_n = np.concatenate([gripper_n, color], axis=0)  
        pointcloud = np.concatenate([pointcloud, gripper_n.reshape(1, 6)], axis=0)
    
    print('pointcloud', pointcloud.shape)
    
    np.save('points2.npy', pointcloud)
    
    # close files
    f.close()
    f_out.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_f", type=str, help="input file path")
    parser.add_argument("--out_f", type=str, help="output file path")
    parser.add_argument("--policy", type=str, help="path to policy checkpoint")
    args = parser.parse_args()
    
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=args.policy , device='cuda', verbose=True)
    pred_track(args.in_f, policy, args.out_f)