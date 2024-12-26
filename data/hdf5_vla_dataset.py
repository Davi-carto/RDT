import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING

class HDF5VLADataset:
    """
    This class is used to sample episodes from the embodiment dataset
    stored in HDF5.
    """
    def __init__(self) -> None:
        # [Modify] The path to the HDF5 dataset directory
        # Each HDF5 file contains one episode
        HDF5_DIR = "data/output/pusht_episodes/" # TODO：修改路径
        self.DATASET_NAME = "pusht" # TODO：修改路径
        
        # 收集所有 .hdf5 文件路径,每个文件代表一个episode
        # os.walk()遍历 HDF5_DIR ，过滤出所有 .hdf5 文件路径，拼接出完整路径，存储到列表
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                self.file_paths.append(file_path)
                
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size'] # 预测多少步 action
        self.IMG_HISORY_SIZE = config['common']['img_history_size'] # 图像历史大小
        self.STATE_DIM = config['common']['state_dim'] # 统一的动作空间的维度大小（128）
    
        # Get each episode's len
        # valid 反应episode是否有效，res 是解析后的数据
        # episode_sample_weights 是每个episode长度占所有episode长度的比例，即权重
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    # len() 函数为Python内置函数，返回对象中元素数量
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    # 抽样一个episode样本用于训练
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.
        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        # 根据每个episode的权重，随机选择一个episode；
        # 如果index被提供，则直接使用index索引的episode；
        # 再检验是否有效
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            num_steps = f['robot_joint'].shape[0]
            if num_steps < 128:
                return False, None

            qpos = f['robot_joint'][:]
            eef_pose = f['robot_eef_pose'][:]
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # We randomly sample a timestep
            # 随机采样一个 timestep 从机械臂开始有效运动的到 episode 结束
            step_id = np.random.randint(first_idx-1, num_steps)
            

            """
            # Load the instruction
            dir_path = os.path.dirname(file_path)
            with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
                instruction_dict = json.load(f_instr)
            # We have 1/3 prob to use original instruction,
            # 1/3 to use simplified instruction,
            # and 1/3 to use expanded instruction.
            instruction_type = np.random.choice([
                'instruction', 'simplified_instruction', 'expanded_instruction'])
            instruction = instruction_dict[instruction_type]
            # 如果 instruction 是列表，随机选择一个 instruction ，如果 instruction 有很多种相似但不同的描述 （isinstance()检查一个对象是否是指定类或指定类元组）
            if isinstance(instruction, list):
                instruction = np.random.choice(instruction)
            # You can also use precomputed language embeddings (recommended)
            """
            instruction = "data/output/lang_embed_0.pt" # TODO: 修改路径
            
            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            target_qpos = f['action'][step_id:step_id+self.CHUNK_SIZE]
            
            # Parse the state and action
            state_qpos = qpos[step_id:step_id+1] 
            state_qpos_std = np.std(qpos, axis=0)
            state_qpos_mean = np.mean(qpos, axis=0)
            state_qpos_norm = np.sqrt(np.mean(qpos**2, axis=0))
            
            state_eef_pose = eef_pose[step_id:step_id+1]
            state_eef_pose_std = np.std(eef_pose, axis=0)
            state_eef_pose_mean = np.mean(eef_pose, axis=0)
            state_eef_pose_norm = np.sqrt(np.mean(eef_pose**2, axis=0))

            state_std = state_qpos_std + state_eef_pose_std # 为后面state_indicator的赋值提供数据

            actions = target_qpos

            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                # 如果 actions 总的 timestep 小于 CHUNK_SIZE，则用最后一个 action 填充
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            # Fill the state/action into the unified vector 
            # Please refer to configs/state_vec.py for an explanation of each element in the unified vector.
            def fill_in_state(values):
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            def fill_in_state_eef_pose(values):
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            state_qpos = fill_in_state(state_qpos)
            state_qpos_std = fill_in_state(state_qpos_std)
            state_qpos_mean = fill_in_state(state_qpos_mean)
            state_qpos_norm = fill_in_state(state_qpos_norm)

            state_eef_pose = fill_in_state_eef_pose(state_eef_pose)
            state_eef_pose_std = fill_in_state_eef_pose(state_eef_pose_std)
            state_eef_pose_mean = fill_in_state_eef_pose(state_eef_pose_mean)
            state_eef_pose_norm = fill_in_state_eef_pose(state_eef_pose_norm)
            
            state_indicator = fill_in_state(np.ones_like(state_std))
            # 每个 fill_in_state 生成不同的向量实例，
            # 将 state_qpos 和 state_eef_pose 拼接起来，生成 state 向量，在一个实例之中
            state = state_qpos + state_eef_pose
            state_std = state_qpos_std + state_eef_pose_std
            state_mean = state_qpos_mean + state_eef_pose_mean
            state_norm = state_qpos_norm + state_eef_pose_norm

            actions = fill_in_state(actions)

            # Parse the images
            # key 是图像类型，如 cam_high、cam_left_wrist、cam_right_wrist
            # TODO：更改图像部分代码
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    img = f[key][i]
                    # 解码图像数据为RGB格式

                    imgs.append(img)
                    
                    # imgs.append(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)) 这一句一直没加
                    # print(f"Image type: {type(imgs)}, shape: {imgs.shape}")  # 打印img的数据格式和维数        
                # 将图像列表转换为numpy数组
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs   
            # `cam_high` is the external camera image
            cam_high = parse_img('camera_0')
            # For step_id = first_idx - 1, the valid_len should be one
            # 通过掩码区分真实和填充的图像，计算损失和评估时时忽略填充帧
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_right_wrist = parse_img('camera_1')
            cam_right_wrist_mask = cam_high_mask.copy()

            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
            cam_left_wrist_mask = cam_high_mask.copy()
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist, # TODO:添加左手
                "cam_left_wrist_mask": cam_left_wrist_mask, # TODO:添加左手
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """
        [Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            num_steps = f['robot_joint'].shape[0]
            if num_steps < 128:
                return False, None

            qpos = f['robot_joint'][:]
            eef_pose = f['robot_eef_pose'][:]
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            target_qpos = f['action'][:]
            
            # Parse the state and action (state1: qpos, state2: eef_pose, action: target_qpos)
            state_qpos = qpos[first_idx-1:]
            state_eef_pose = eef_pose[first_idx-1:]
            action = target_qpos[first_idx-1:]
            
            # Fill the state/action into the unified vector
            def fill_in_state(values):
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            def fill_in_state_eef_pose(values):
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec
            
            state_qpos = fill_in_state(state_qpos)
            state_eef_pose = fill_in_state_eef_pose(state_eef_pose)
            state = state_qpos + state_eef_pose

            action = fill_in_state(action)
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)
