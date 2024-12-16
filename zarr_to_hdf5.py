# 实现 class ZARRVLADataset 所需要的内容
# observations : joint_position of the robot_arm (TRAJ_LEN, 6) | image : cam_high (TRAJ_LEN, 480, 640, 3)、cam_right_wrist (TRAJ_LEN, 480, 640, 3)
# action : desired joint positions of the two robot arms at the next time step (TRAJ_LEN, 14) (not the same as the actual joint positions at the next time step)
# Note: The number in episode_<NUMBER>.hdf5 is not necessarily consecutive. TRAJ_LEN may vary from episode to episode.
import os
import fnmatch
import json
import click
import h5py
import zarr
import yaml
import cv2
import numpy as np

from Diffusion_Policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from Diffusion_Policy.diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)
register_codecs()

def load_replay_buffer(zarr_path):
    # 加载 ReplayBuffer
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
    return replay_buffer

def save_episode_data(episode_data, save_dir, episode_idx):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存每个数据项到单独的文件
    for key, value in episode_data.items():
        file_path = os.path.join(save_dir, f'episode_{episode_idx}_{key}.npy')
        np.save(file_path, value)
        print(f"Saved {key} for episode {episode_idx} to {file_path}")

def print_episode_data(file_path):
    with h5py.File(file_path, 'r') as h5file:
        for key in h5file.keys():
            print(f"{key}:")
            data = h5file[key]
            if isinstance(data, h5py.Dataset):
                print(f"  type={type(data)}, shape={data.shape}")
            elif isinstance(data, h5py.Group):
                print(f"  Group with keys: {list(data.keys())}")

def save_episode_data_hdf5(episode_data, save_dir, episode_idx):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存每个 episode 到单独的 HDF5 文件
    file_path = os.path.join(save_dir, f'episode_{episode_idx}.hdf5')
    with h5py.File(file_path, 'w') as h5file:
        for key, value in episode_data.items():
            h5file.create_dataset(key, data=value)
        print(f"Saved episode {episode_idx} to {file_path}")

def print_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as h5file:
        def print_attrs(name, obj):
            print(name)
        h5file.visititems(print_attrs)

@click.command()
@click.option('--zarr_path', '-i',  required=True)
@click.option('--save_dir', '-o', default=None)

def main(zarr_path, save_dir):
    """
    # 加载 ReplayBuffer
    replay_buffer = load_replay_buffer(zarr_path)

    # 遍历每个 episode
    for episode_idx in range(replay_buffer.n_episodes):
        # 提取 episode 数据
        episode_data = replay_buffer.get_episode(episode_idx, copy=True)

        # 保存 episode 数据为 HDF5 格式
        save_episode_data_hdf5(episode_data, save_dir, episode_idx)
    """
    hdf5_file_path = os.path.join(save_dir, 'episode_0.hdf5')
    print_episode_data(hdf5_file_path)

if __name__ == "__main__":
    main()
