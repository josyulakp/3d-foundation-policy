"""
Add pointcloud information to existing droid hdf5 file
"""
import h5py
import os
import numpy as np
import glob
from tqdm import tqdm
import argparse
import shutil
import torch
import random
import traceback
import json
import cv2
import sys

"""
Set up ZED camera by following the instructions here:
https://www.stereolabs.com/docs/installation/linux/
"""
import pyzed.sl as sl

import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

from droid.camera_utils.wrappers.recorded_multi_camera_wrapper import RecordedMultiCameraWrapper
from droid.trajectory_utils.trajectory_reader import TrajectoryReader
from droid.camera_utils.info import camera_type_to_string_dict

from droid.camera_utils.camera_readers.zed_camera import ZedCamera, standard_params
from scipy.spatial.transform import Rotation as R
from openpoints.models.layers import furthest_point_sample

# Convert extrinsic parameters to a transformation matrix
def ext2mat(ext):
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R.from_euler("xyz", ext[3:]).as_matrix()
    extrinsic[:3, 3] = ext[:3]
    return extrinsic

# Transform point cloud from camera to world coordinates
def transform_point_cloud(points_camera, transformation_matrix):
    xyz_camera = points_camera[:, :3]
    xyz_camera_homogeneous = np.hstack((xyz_camera, np.ones((xyz_camera.shape[0], 1))))
    xyz_world_homogeneous = np.matmul(transformation_matrix, xyz_camera_homogeneous.T).T
    xyz_world = xyz_world_homogeneous[:, :3]
    return np.hstack((xyz_world, points_camera[:, 3:]))

# Extract point cloud from camera data using extrinsic transformation
def points_from_camera_name(xyz, rgb, extrinsics, name, i):
    xyzi = xyz[i]
    rgbi = rgb[i] / 255.0  # Normalize RGB values
    extrinsicsi = extrinsics[i]
    points = np.concatenate([xyzi, rgbi.reshape(-1, 3)], axis=1)
    extrinsicsi = ext2mat(extrinsicsi)
    return transform_point_cloud(points, extrinsicsi)

# Get camera intrinsic parameters (currently empty)
def get_cam_instrinsics(svo_path):
    return {}

# Main dataset conversion function
def convert_dataset(path, args):
    output_path = os.path.join(os.path.dirname(path), "trajectory_pcd.h5")
    recording_folderpath = os.path.join(os.path.dirname(path), "recordings", "SVO")

    # Skip conversion if dataset already exists
    if os.path.exists(output_path):
        print("Skipping finished")
        return

    num_svo_files = len([f for f in os.listdir(recording_folderpath) if os.path.isfile(os.path.join(recording_folderpath, f))])

    # Define camera settings
    camera_kwargs = {
        "hand_camera": {"image": True, "concatenate_images": False, "resolution": (args.w, args.h), "resize_func": "cv2"},
        "varied_camera": {"image": True, "concatenate_images": False, "resolution": (args.w, args.h), "resize_func": "cv2"},
    }

    shutil.copyfile(path, output_path)
    f = h5py.File(output_path, "a")
    demo_len = f["action"]["cartesian_position"].shape[0]

    # Initialize camera groups if not already present
    if "camera" not in f["observation"]:
        f["observation"].create_group("camera").create_group("image")
        f["observation/camera"].create_group("pointcloud")
    if "camera" in f["observation"] and "image" not in f["observation/camera"]:
        f["observation/camera"].create_group("image")
    image_grp = f["observation/camera/image"]
    pcd_grp = f["observation/camera/pointcloud"]

    # Set up camera types and IDs
    CAM_ID_TO_TYPE = {}
    hand_cam_ids = []
    varied_cam_ids = []
    for k in f["observation"]["camera_type"]:
        cam_type = camera_type_to_string_dict[f["observation"]["camera_type"][k][0]]
        CAM_ID_TO_TYPE[k] = cam_type
        if cam_type == "hand_camera":
            hand_cam_ids.append(k)
        elif cam_type == "varied_camera":
            varied_cam_ids.append(k)
        else:
            raise ValueError

    hand_cam_ids = sorted(hand_cam_ids)
    varied_cam_ids = sorted(varied_cam_ids)

    IMAGE_NAME_TO_CAM_KEY_MAPPING = {
        "hand_camera_left_image": "{}_left".format(hand_cam_ids[0]),
        "varied_camera_2_left_image": "{}_left".format(varied_cam_ids[random.randint(0, len(varied_cam_ids)-1)]),
    }
    CAM_ID_TO_TYPE[hand_cam_ids[0]] = "hand_camera"
    CAM_ID_TO_TYPE[varied_cam_ids[0]] = "varied_camera"

    # Initialize containers for camera data
    cam_data = {cam_name: [] for cam_name in IMAGE_NAME_TO_CAM_KEY_MAPPING.keys()}
    cam_data_pcd = {cam_name: [] for cam_name in IMAGE_NAME_TO_CAM_KEY_MAPPING.keys()}

    traj_reader = TrajectoryReader(path, read_images=False)
    camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs, disable_camera_ids=[v_id for v_id in varied_cam_ids if v_id != varied_cam_ids[0]])

    # Loop through trajectory data
    for index in range(demo_len):
        timestep = traj_reader.read_timestep(index=index)
        camera_obs = camera_reader.read_cameras(index=index, camera_type_dict=CAM_ID_TO_TYPE)

        # Process camera data
        for cam_name in IMAGE_NAME_TO_CAM_KEY_MAPPING.keys():
            if camera_obs is None:
                im = np.zeros((args.h, args.w, 3))
                xyz = np.zeros((args.h, args.w, 4))
            else:
                im_key = IMAGE_NAME_TO_CAM_KEY_MAPPING[cam_name]
                im = camera_obs["image"][im_key]
                im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
                xyz = camera_obs["pointcloud"][im_key]

            im = im[:, :, ::-1]  # Convert from BGR to RGB
            xyz = xyz.reshape(-1, 4)[:, :3]  # Extract XYZ from point cloud

            cam_data[cam_name].append(im)
            cam_data_pcd[cam_name].append(xyz)

    # Convert to numpy arrays and store in HDF5
    for cam_name in cam_data.keys():
        cam_data[cam_name] = np.array(cam_data[cam_name]).astype(np.uint8)
        cam_data_pcd[cam_name] = np.array(cam_data_pcd[cam_name]).astype(np.float32)
        if cam_name not in image_grp:
            image_grp.create_dataset(cam_name, data=cam_data[cam_name], compression="gzip")

    # Process camera extrinsics data
    if "extrinsics" not in f["observation/camera"]:
        f["observation/camera"].create_group("extrinsics")
    extrinsics_grp = f["observation/camera/extrinsics"]
    for (im_name, cam_key) in IMAGE_NAME_TO_CAM_KEY_MAPPING.items():
        raw_data = f["observation/camera_extrinsics"][cam_key][:]
        extrinsics = raw_data
        extr_name = "_".join(im_name.split("_")[:-1])
        if extr_name in extrinsics_grp:
            del extrinsics_grp[extr_name]
        else:
            extrinsics_grp.create_dataset(extr_name, data=extrinsics)

    # Process point cloud data and store it
    for cam_name in cam_data_pcd.keys():
        name = cam_name.replace('image', 'pcd_4000')
        if name in pcd_grp:
            del pcd_grp[cam_name]

        xyz = cam_data_pcd[cam_name]
        rgb = cam_data[cam_name]
        cam_key = IMAGE_NAME_TO_CAM_KEY_MAPPING[cam_name]
        extrinsics = f["observation/camera_extrinsics"][cam_key][:]
        data = np.zeros((xyz.shape[0], xyz.shape[1], 6))

        for i in range(xyz.shape[0]):
            points = points_from_camera_name(xyz, rgb, extrinsics, cam_name, i)
            data[i] = points
        pcd = torch.tensor(data, dtype=torch.float32).cuda()
        pcd = torch.nan_to_num(pcd, nan=0.0, posinf=0.0, neginf=0.0)
        xyz = pcd[..., :3].contiguous()
        mask = (xyz < -1.0) | (xyz > 1.0)
        pcd[mask.any(dim=-1)] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=pcd.device)
        index = furthest_point_sample(pcd[..., :3].contiguous(), 8000)
        pcd = torch.gather(pcd, 1, index.unsqueeze(-1).long().expand(-1, -1, pcd.shape[-1]))
        pcd = pcd.cpu().numpy()
        pcd_grp.create_dataset(name, data=pcd, compression="gzip", dtype=np.float16)
    
    svo_path = os.path.join(os.path.dirname(path), "recordings", "SVO")
    cam_reader_svo = camera_reader  # Using a camera reader wrapper
    
    # Check if "intrinsics" group exists, otherwise create it
    if "intrinsics" not in f["observation/camera"]:
        f["observation/camera"].create_group("intrinsics")
    
    intrinsics_grp = f["observation/camera/intrinsics"]    
    
    # Process camera intrinsics for each camera in the dataset
    for cam_id, svo_reader in cam_reader_svo.camera_dict.items():
        cam = svo_reader._cam
        calib_params = cam.get_camera_information().camera_configuration.calibration_parameters
        
        # For each camera (left and right), store the intrinsic parameters
        for (suffix, params) in zip(["_left", "_right"], [calib_params.left_cam, calib_params.right_cam]):
            cam_key = cam_id + suffix  # Create camera key
            
            # Reverse search for corresponding image name
            im_name = None
            for k, v in IMAGE_NAME_TO_CAM_KEY_MAPPING.items():
                if v == cam_key:
                    im_name = k
                    break
            if im_name is None:  # Skip if no corresponding image found
                continue
            intr_name = "_".join(im_name.split("_")[:-1])  # Extract intrinsic name
            
            # Store the intrinsic matrix for each camera
            cam_intrinsics = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])
            data = np.repeat(cam_intrinsics[None], demo_len, axis=0)
            intrinsics_grp.create_dataset(intr_name, data=data)
    
    # Process actions (cartesian_position and cartesian_velocity)
    action_dict_group = f["action"]
    for in_ac_key in ["cartesian_position", "cartesian_velocity"]:
        in_action = action_dict_group[in_ac_key][:]
        in_pos = in_action[:, :3].astype(np.float64)
        in_rot = in_action[:, 3:6].astype(np.float64)  # Rotation in Euler format
        
        # Convert Euler angles to 6D rotation format
        rot_ = torch.from_numpy(in_rot)
        rot_6d = TorchUtils.euler_angles_to_rot_6d(rot_, convention="XYZ")
        rot_6d = rot_6d.numpy().astype(np.float64)
        
        # Prefix for the dataset names
        prefix = "abs_" if in_ac_key == "cartesian_position" else "rel_"
        
        # Store position and rotation data
        this_action_dict = {
            prefix + 'pos': in_pos,
            prefix + 'rot_euler': in_rot,
            prefix + 'rot_6d': rot_6d,
        }
        for key, data in this_action_dict.items():
            if key in action_dict_group:
                del action_dict_group[key]
            action_dict_group.create_dataset(key, data=data)
    
    # Ensure all action keys are batched (i.e., are not 0-dimensional)
    for k in action_dict_group:
        if isinstance(action_dict_group[k], h5py.Dataset) and len(action_dict_group[k].shape) == 1:
            reshaped_values = np.reshape(action_dict_group[k][:], (-1, 1))
            del action_dict_group[k]
            action_dict_group.create_dataset(k, data=reshaped_values)
    
    # Post-processing: Remove timesteps where robot movement is disabled
    movement_enabled = f["observation/controller_info/movement_enabled"][:]
    timesteps_to_remove = np.where(movement_enabled == False)[0]

    if not args.keep_idle_timesteps:
        remove_timesteps(f, timesteps_to_remove)

    f.close()  # Close the HDF5 file
    camera_reader.disable_cameras()  # Disable cameras after processing
    del camera_reader  # Free up memory

def remove_timesteps(f, timesteps_to_remove):
    """
    Remove the specified timesteps from the dataset.

    Args:
        f (h5py.File): HDF5 file to modify.
        timesteps_to_remove (array): Indices of timesteps to remove.
    """
    total_timesteps = f["action/cartesian_position"].shape[0]
    
    # Recursive function to remove timesteps from all groups
    def remove_timesteps_for_group(g):
        for k in g:
            if isinstance(g[k], h5py._hl.dataset.Dataset):
                if g[k].shape[0] != total_timesteps:
                    print(f"Skipping {k}")
                    continue
                new_dataset = np.delete(g[k], timesteps_to_remove, axis=0)
                del g[k]
                g.create_dataset(k, data=new_dataset)
            elif isinstance(g[k], h5py._hl.group.Group):
                remove_timesteps_for_group(g[k])
            else:
                raise NotImplementedError

    for k in f:
        remove_timesteps_for_group(f[k])

def process_dataset(d, args):
    """
    Process the dataset and call the conversion function.

    Args:
        d (str): Path to the dataset.
        args (argparse.Namespace): Arguments containing configuration parameters.
    """
    d = os.path.expanduser(d)  # Expand user directory
    
    # Convert the dataset
    convert_dataset(d, args)
    torch.cuda.empty_cache()  # Clear CUDA memory cache
    print(f"{d} Done")
    sys.stdout.flush()  # Ensure output is flushed
    return True


def worker(id, task_q, result_q, args):
    """
    Worker function to process tasks from the queue.

    Args:
        id (int): Worker ID.
        task_q (queue.Queue): Queue containing tasks to process.
        result_q (queue.Queue): Queue to store the results.
        args (argparse.Namespace): Arguments containing configuration parameters.
    """
    while True:
        try:
            item = task_q.get(timeout=5)  # Get a task from the queue
            if item is None:
                break  # Stop if None is encountered
            try:
                process_dataset(item, args)  # Process the dataset
                result_q.put((item, True, None))  # Put success result in the queue
            except Exception as e:
                result_q.put((item, False, e))  # Put failure result in the queue
        except queue.Empty:
            break


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder",
        type=str,
        help="Folder containing hdf5's to add camera images to",
        default="~/datasets/droid/success"
    )

    parser.add_argument(
        "--w",
        type=int,
        default=320,
        help="Image width",
    )
    
    parser.add_argument(
        "--h",
        type=int,
        default=180,
        help="Image height",
    )

    parser.add_argument(
        "--keep_idle_timesteps",
        action="store_true",
        help="Override the default behavior of truncating idle timesteps",
    )
    
    args = parser.parse_args()

    # Set multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)
    
    datasets = []
    # Walk through directories to find trajectory files
    j = os.walk(os.path.expanduser(args.folder))
    for root, dirs, files in j:
        for f in files:
            if f == "trajectory.h5":
                if "success" in root:
                    datasets.append(os.path.join(root, f))
                    print(len(datasets))

    print("Converting datasets...")
    random.shuffle(datasets)  # Shuffle the datasets for processing
    failed = 0
    num_process = 16
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Add tasks to the queue
    for item in datasets:
        task_queue.put(item)
        
    # Add None to the queue to stop workers
    for _ in range(num_process):
        task_queue.put(None)
        
    processes = []
    # Create worker processes
    for id in range(num_process):
        print('GPU ID:', id)
        p = mp.Process(target=worker, args=(id, task_queue, result_queue, args))
        p.start()
        processes.append(p)
    
    total_tasks = len(datasets)
    # Track progress with tqdm
    with tqdm(total=total_tasks, desc="Processing DROID data") as pbar:
        while total_tasks > 0:
            path, success, error = result_queue.get()
            if success:
                pass
            else:
                print(f"Error processing {path}: {error}")
            pbar.update(1)
            total_tasks -= 1
            
    # Join processes to ensure all workers finish
    for p in processes:
        p.join()