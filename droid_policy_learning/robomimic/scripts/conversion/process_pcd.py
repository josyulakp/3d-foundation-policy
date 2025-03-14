import h5py
import fpsample
import numpy as np
# import cv2
import sys
from scipy.spatial.transform import Rotation as R
import argparse
import random
import json
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
# from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cuda
from openpoints.models.layers import furthest_point_sample
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

# def farthest_point_sample(xyz, npoint, use_cuda=True):
#   """
#   Modified to support both numpy array and torch tensor.

#   Input:
#       xyz: pointcloud data, [B, N, 3], can be either numpy array or tensor
#       npoint: number of samples
#   Return:
#       centroids: sampled pointcloud index, [B, npoint]
#   """
#   # Convert numpy array to PyTorch tensor if necessary
#   if isinstance(xyz, np.ndarray):
#       xyz = torch.tensor(xyz, dtype=torch.float32)

#   # Ensure the tensor is on the correct device
#   device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
#   xyz = xyz.to(device)

#   B, N, C = xyz.shape

#   if use_cuda and torch.cuda.is_available():
#     #   print('Use pointnet2_cuda!')
#       sampled_points_ids = farthest_point_sample(xyz, npoint)
#   else:
#       centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#       distance = torch.ones(B, N).to(device) * 1e10
#       farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
#       batch_indices = torch.arange(B, dtype=torch.long).to(device)
#       for i in range(npoint):
#           centroids[:, i] = farthest
#           centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#           dist = torch.sum((xyz - centroid)**2, -1)
#           mask = dist < distance
#           distance[mask] = dist[mask]
#           farthest = torch.max(distance, -1)[1]
#       sampled_points_ids = centroids

#   return sampled_points_ids.long()

def points_from_camera_name(f, name, i):
    xyz = np.array(f['observation/camera/pointcloud/{}_image'.format(name)][i])
    rgb = np.array(f['observation/camera/image/{}_image'.format(name)][i])
    rgb = 2 * (rgb / 255.0) - 1
    extrinsics = np.array(f['observation/camera/extrinsics/{}'.format(name)][i])
    points = np.concatenate([xyz, rgb.reshape(-1,3)], axis=1)

    extrinsics = ext2mat(extrinsics)
    points = transform_point_cloud(points, extrinsics)

    return points


def convert_pcd(path):
    print(path)
    try:
        f = h5py.File(path, 'a')
        H = f['observation/robot_state/cartesian_position'].shape[0]
        print(H)
        if H == 0:
            return False
        for camera_name in ['hand_camera_left', 'varied_camera_2_left']:
            # if 'observation/camera/pointcloud/{}_pcd_4000'.format(camera_name) in f:
            #     if 'observation/camera/pointcloud/{}_pcd'.format(camera_name) in f:
            #         del f['observation/camera/pointcloud/{}_pcd'.format(camera_name)]
            #     continue
            # else:
            #     # return False
            #     del f['observation/camera/pointcloud/{}_pcd'.format(camera_name)]
            # else:
            #     return False
            pcd = np.zeros((H, 128 * 128, 6))
            for i in range(H):
                points = points_from_camera_name(f, camera_name, i)
                pcd[i] = points
                # print('ok')
            # pcd = np.array(f["observation/camera/pointcloud/{}_pcd".format(camera_name)])
            # f['observation/camera/pointcloud'].create_dataset(f'{camera_name}_pcd', data=pcd)
            pcd = torch.tensor(pcd, dtype=torch.float32).cuda()
            pcd = torch.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)
            xyz = pcd[..., :3].contiguous()
            mask = (xyz < -1.0) | (xyz > 1.0)
            pcd[mask.any(dim=-1)] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=pcd.device)
            index = furthest_point_sample(pcd[..., :3].contiguous(), 4000)
            pcd = torch.gather(pcd, 1, index.unsqueeze(-1).long().expand(-1, -1, pcd.shape[-1]))
            pcd = pcd.cpu().numpy()
            f['observation/camera/pointcloud'].create_dataset(f'{camera_name}_pcd_4000', data=pcd)
            torch.cuda.empty_cache()
            
        f.close()
        return True
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise e
        print(f"failed to convert {path}")
        print(e)
        return False

def process_dataset(item):
    d, l = item['path'], item['lang']
    d = os.path.expanduser(d)
    if convert_pcd(d):
        print(f"converted {d} done")
        return item
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_file", type=str, help="manifest file path")
    args = parser.parse_args()

    with open(args.manifest_file, 'r') as file:
        datasets = json.load(file)
    print("converting pointcloud...")
    random.shuffle(datasets)
    
    success = []

    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = [executor.submit(process_dataset, item) for item in datasets]
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         result = future.result()
    #         if result is not None:
    #             success.append(result)
    
    for item in tqdm(datasets):
        result = process_dataset(item)
        if result is not None:
            success.append(result)
   
    with open('success4.json', 'w') as file:
        json.dump(success, file, indent=4)
