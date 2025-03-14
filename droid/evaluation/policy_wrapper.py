import numpy as np
import torch
from collections import deque

from droid.data_processing.timestep_processing import TimestepProcesser
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

from scipy.spatial.transform import Rotation as R
import fpsample

import open3d as o3d
import time
import random

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
#       xyz = torch.from_numpy(xyz)

#   # Ensure the tensor is on the correct device
#   device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
#   xyz = xyz.to(device)

#   B, N, C = xyz.shape

#   if use_cuda and torch.cuda.is_available():
#       print('Use pointnet2_cuda!')
#       from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cuda
#       sampled_points_ids = fps_cuda(xyz, npoint)
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

#   return sampled_points_ids

def converter_helper(data, batchify=True):
    if torch.is_tensor(data):
        pass
    elif isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    else:
        raise ValueError

    if batchify:
        data = data.unsqueeze(0)
    return data


def np_dict_to_torch_dict(np_dict, batchify=True):
    torch_dict = {}

    for key in np_dict:
        curr_data = np_dict[key]
        if isinstance(curr_data, dict):
            torch_dict[key] = np_dict_to_torch_dict(curr_data)
        elif isinstance(curr_data, np.ndarray) or torch.is_tensor(curr_data):
            torch_dict[key] = converter_helper(curr_data, batchify=batchify)
        elif isinstance(curr_data, list):
            torch_dict[key] = [converter_helper(d, batchify=batchify) for d in curr_data]
        else:
            raise ValueError

    return torch_dict

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

def get_pointcloud(xyz, rgb, ext):
    xyz, rgb, ext = np.array(xyz), np.array(rgb), np.array(ext)
    # print(rgb)
    xyz = xyz[:,:,:3]
    rgb = np.transpose(rgb, (1,2,0))
    # print(xyz.shape, rgb.shape)
    pcd_camera = np.concatenate([xyz, rgb], axis=2).reshape(-1,6)
    pcd_world = transform_point_cloud(pcd_camera, ext2mat(ext))
    # pcd_world = pcd_world[~np.isnan(pcd_world).any(axis=1)]
    # pcd_world = pcd_world[~np.isinf(pcd_world).any(axis=1)]
    pcd_world = np.nan_to_num(pcd_world, nan=0.0, posinf=0.0, neginf=0.0)
    pcd = torch.tensor(pcd_world, dtype=torch.float32).cuda()
    # pcd = torch.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)
    xyz = pcd[..., :3].contiguous()
    mask = (xyz < -1.0) | (xyz > 1.0)
    pcd[mask.any(dim=-1)] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=pcd.device)
    pcd = pcd.cpu().numpy()
    pcd_world = pcd
    # mask = (np.abs(pcd_world[:, 0]) < 1.0 ) & (np.abs(pcd_world[:, 1]) < 1.0) & (np.abs(pcd_world[:, 2]) < 1.0)
    # pcd_world[mask.any(axis=-1)] = np.array([0,0,0,0,0,0])
    t0 = time.time()
    index = fpsample.bucket_fps_kdline_sampling(pcd_world[..., :3], 8000, 7)
    # index = np.random.choice(pcd_world.shape[0], 10000, replace=False)
    # index = farthest_point_sample(torch.tensor(pcd_world[..., :3][None].copy(), dtype=torch.float32), 4000).cpu().numpy()[0]
    
    t1 = time.time()
    print("sample time:", t1-t0)
    pcd_world = pcd_world[index]
    return pcd_world

class PolicyWrapper:
    def __init__(self, policy, timestep_filtering_kwargs, image_transform_kwargs, eval_mode=True):
        self.policy = policy

        if eval_mode:
            self.policy.eval()
        else:
            self.policy.train()

        self.timestep_processor = TimestepProcesser(
            ignore_action=True, **timestep_filtering_kwargs, image_transform_kwargs=image_transform_kwargs
        )

    def forward(self, observation):
        timestep = {"observation": observation}
        processed_timestep = self.timestep_processor.forward(timestep)
        torch_timestep = np_dict_to_torch_dict(processed_timestep)
        action = self.policy(torch_timestep)[0]
        np_action = action.detach().numpy()

        # a_star = np.cumsum(processed_timestep['observation']['state']) / 7
        # print('Policy Action: ', np_action)
        # print('Expert Action: ', a_star)
        # print('Error: ', np.abs(a_star - np_action).mean())

        # import pdb; pdb.set_trace()
        return np_action


class PolicyWrapperRobomimic:
    def __init__(self, policy, timestep_filtering_kwargs, image_transform_kwargs, frame_stack, eval_mode=True):
        self.policy = policy

        assert eval_mode is True

        self.fs_wrapper = FrameStackWrapper(num_frames=frame_stack)
        self.fs_wrapper.reset()
        self.policy.start_episode()

        self.timestep_processor = TimestepProcesser(
            ignore_action=True, **timestep_filtering_kwargs, image_transform_kwargs=image_transform_kwargs
        )

    def convert_raw_extrinsics_to_Twc(self, raw_data):
        """
        helper function that convert raw extrinsics (6d pose) to transformation matrix (Twc)
        """
        raw_data = torch.from_numpy(np.array(raw_data))
        pos = raw_data[0:3]
        rot_mat = TorchUtils.euler_angles_to_matrix(raw_data[3:6], convention="XYZ")
        extrinsics = np.zeros((4, 4))
        extrinsics[:3,:3] = TensorUtils.to_numpy(rot_mat)
        extrinsics[:3,3] = TensorUtils.to_numpy(pos)
        extrinsics[3,3] = 1.0
        # invert the matrix to represent standard definition of extrinsics: from world to cam
        extrinsics = np.linalg.inv(extrinsics)
        return extrinsics

    def forward(self, observation):
        t0 = time.time()
        timestep = {"observation": observation}
        processed_timestep = self.timestep_processor.forward(timestep)

        extrinsics_dict = processed_timestep["extrinsics_dict"]
        intrinsics_dict = processed_timestep["intrinsics_dict"]
        # import pdb; pdb.set_trace()

        obs = {
            "robot_state/cartesian_position": observation["robot_state"]["cartesian_position"],
            "robot_state/gripper_position": [observation["robot_state"]["gripper_position"]], # wrap as array, raw data is single float
            
            "camera/image/hand_camera_left_image": processed_timestep["observation"]["camera"]["image"]["hand_camera"][0],
            "camera/image/hand_camera_right_image": processed_timestep["observation"]["camera"]["image"]["hand_camera"][1],
            "camera/image/varied_camera_2_left_image": processed_timestep["observation"]["camera"]["image"]["varied_camera"][0],
            "camera/image/varied_camera_2_right_image": processed_timestep["observation"]["camera"]["image"]["varied_camera"][1],
            # "camera/image/varied_camera_2_left_image": processed_timestep["observation"]["camera"]["image"]["varied_camera"][2],
            # "camera/image/varied_camera_2_right_image": processed_timestep["observation"]["camera"]["image"]["varied_camera"][3],

            "camera/pointcloud/hand_camera_left_pcd_4000": get_pointcloud(processed_timestep["observation"]["camera"]["pointcloud"]["hand_camera"][0], processed_timestep["observation"]["camera"]["image"]["hand_camera"][0], extrinsics_dict["hand_camera"][0]),
            # "camera/pointcloud/hand_camera_right_pcd": get_pointcloud(processed_timestep["observation"]["camera"]["pointcloud"]["hand_camera"][1], processed_timestep["observation"]["camera"]["image"]["hand_camera"][0], extrinsics_dict["hand_camera"][0]),
            "camera/pointcloud/varied_camera_2_left_pcd_4000": get_pointcloud(processed_timestep["observation"]["camera"]["pointcloud"]["varied_camera"][0], processed_timestep["observation"]["camera"]["image"]["varied_camera"][0], extrinsics_dict["varied_camera"][0]),
            # "camera/pointcloud/varied_camera_1_right_pcd": get_pointcloud(processed_timestep["observation"]["camera"]["pointcloud"]["varied_camera"][1], processed_timestep["observation"]["camera"]["image"]["varied_camera"][1], extrinsics_dict["varied_camera"][1]),
            # "camera/pointcloud/varied_camera_2_left_pcd_4000": get_pointcloud(processed_timestep["observation"]["camera"]["pointcloud"]["varied_camera"][0], processed_timestep["observation"]["camera"]["image"]["varied_camera"][0], extrinsics_dict["varied_camera"][0]),
            # "camera/pointcloud/varied_camera_2_right_pcd": get_pointcloud(processed_timestep["observation"]["camera"]["pointcloud"]["varied_camera"][3], processed_timestep["observation"]["camera"]["image"]["varied_camera"][3], extrinsics_dict["varied_camera"][3]),

            "camera/extrinsics/hand_camera_left": self.convert_raw_extrinsics_to_Twc(extrinsics_dict["hand_camera"][0]),
            "camera/extrinsics/hand_camera_right": self.convert_raw_extrinsics_to_Twc(extrinsics_dict["hand_camera"][2]),
            "camera/extrinsics/varied_camera_2_left": self.convert_raw_extrinsics_to_Twc(extrinsics_dict["varied_camera"][0]),
            "camera/extrinsics/varied_camera_2_right": self.convert_raw_extrinsics_to_Twc(extrinsics_dict["varied_camera"][1]),
            # "camera/extrinsics/varied_camera_2_left": self.convert_raw_extrinsics_to_Twc(extrinsics_dict["varied_camera"][2]),
            # "camera/extrinsics/varied_camera_2_right": self.convert_raw_extrinsics_to_Twc(extrinsics_dict["varied_camera"][3]),

            "camera/intrinsics/hand_camera_left": intrinsics_dict["hand_camera"][0],
            "camera/intrinsics/hand_camera_right": intrinsics_dict["hand_camera"][1],
            "camera/intrinsics/varied_camera_2_left": intrinsics_dict["varied_camera"][0],
            "camera/intrinsics/varied_camera_2_right": intrinsics_dict["varied_camera"][1],
            # "camera/intrinsics/varied_camera_2_left": intrinsics_dict["varied_camera"][2],
            # "camera/intrinsics/varied_camera_2_right": intrinsics_dict["varied_camera"][3],
        }
        # pcd_world = obs["camera/pointcloud/varied_camera_2_left_pcd"]
        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(pcd_world[:,:3])
        # o3d_pcd.colors = o3d.utility.Vector3dVector(pcd_world[:,3:] * 0.5 + 0.5)
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([o3d_pcd, axis])
        # time.sleep(60)
        # print(obs["camera/image/varied_camera_2_left_image"])

        # set item of obs as np.array
        for k in obs:
            obs[k] = np.array(obs[k])
        # print(obs["robot_state/cartesian_position"].shape)
        self.fs_wrapper.add_obs(obs)
        obs_history = self.fs_wrapper.get_obs_history()
        t1 = time.time()
        print("obs time:", t1-t0)
        action = self.policy(obs_history)
        t2 = time.time()
        print("action time:", t2-t1)
        return action

    def reset(self):
        self.fs_wrapper.reset()
        self.policy.start_episode()
    

class FrameStackWrapper:
    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """
    def __init__(self, num_frames):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_frames (int): number of past observations (including current observation)
                to stack together. Must be greater than 1 (otherwise this wrapper would
                be a no-op).
        """
        self.num_frames = num_frames

        ### TODO: add action padding option + adding action to obs to include action history in obs ###

        # keep track of last @num_frames observations for each obs key
        self.obs_history = None

    def _set_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        self.obs_history = {}
        for k in init_obs:
            self.obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )

    def reset(self):
        self.obs_history = None

    def get_obs_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        if self.num_frames == 1:
            return { k : np.concatenate(self.obs_history[k], axis=0)[0] for k in self.obs_history }
        else:
            return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }

    def add_obs(self, obs):
        if self.obs_history is None:
            self._set_initial_obs_history(obs)

        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
