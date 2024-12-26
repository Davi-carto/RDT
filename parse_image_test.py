import h5py
import numpy as np
import matplotlib.pyplot as plt

def display_hdf5_image(file_path, key):
    # 打开 HDF5 文件
    with h5py.File(file_path, 'r') as f:
        # 读取图像数据
        image_data = f[key][:]
        
        # 检查图像数据的形状
        print(f"Image shape: {image_data.shape}")
        
        # 选择要显示的帧，例如第一帧
        frame_index = 0
        frame_data = image_data[frame_index]
        
        # 显示图像
        plt.imshow(frame_data)
        plt.axis('off')  # 关闭坐标轴
        plt.show()

if __name__ == "__main__":
    # 使用示例
    file_path = '/home/rzx/RDT/data/output/pusht_episodes/episode_0.hdf5'  # 替换为你的 HDF5 文件路径
    key = 'camera_0'  # 替换为你想要读取的 key
    display_hdf5_image(file_path, key)