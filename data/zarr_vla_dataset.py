# 实现 class ZARRVLADataset 所需要的内容
# observations : joint_position of the robot_arm (TRAJ_LEN, 6) | image : cam_high (TRAJ_LEN, 480, 640, 3)、cam_right_wrist (TRAJ_LEN, 480, 640, 3)
# action : desired joint positions of the two robot arms at the next time step (TRAJ_LEN, 14) (not the same as the actual joint positions at the next time step)
# Note: The number in episode_<NUMBER>.hdf5 is not necessarily consecutive. TRAJ_LEN may vary from episode to episode.
import os
import fnmatch
import json

import zarr
import yaml
import cv2
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer

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

def main():
    # 指定 Zarr 文件路径
    zarr_path = 'path/to/your/replay_buffer.zarr'
    # 指定保存目录
    save_dir = 'path/to/save/episodes'

    # 加载 ReplayBuffer
    replay_buffer = load_replay_buffer(zarr_path)

    # 遍历每个 episode
    for episode_idx in range(replay_buffer.n_episodes):
        # 提取 episode 数据
        episode_data = replay_buffer.get_episode(episode_idx, copy=True)

        # 保存 episode 数据
        save_episode_data(episode_data, save_dir, episode_idx)

if __name__ == "__main__":
    main()
