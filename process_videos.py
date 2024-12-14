import av
import os
import numpy as np
from Diffusion_Policy.diffusion_policy.common.cv2_util import get_image_transform
from Diffusion_Policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from Diffusion_Policy.diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer

def load_replay_buffer(zarr_path):
    # 加载 ReplayBuffer
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
    return replay_buffer

def process_and_store_videos(dataset_path, out_store):
    lowdim_data_dir = os.path.join(dataset_path, 'replay_buffer.zarr')
    lowdim_data = load_replay_buffer(lowdim_data_dir)
    video_dir = os.path.join(dataset_path, 'videos')
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=out_store)

    for episode_idx in range(len(lowdim_data['episode_ends'])):
        episode_video_dir = video_dir.joinpath(str(episode_idx))
        episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))

        for video_path in episode_video_paths:
            camera_idx = int(video_path.stem)
            arr_name = f'camera_{camera_idx}'

            with av.open(str(video_path.absolute())) as container:
                video = container.streams.video[0]
                vcc = video.codec_context
                in_img_res = (vcc.width, vcc.height)

            # 假设out_img_res是目标分辨率
            out_img_res = in_img_res
            image_tf = get_image_transform(input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False)

            # 创建zarr数据集
            if arr_name not in replay_buffer.data:
                ow, oh = out_img_res
                replay_buffer.data.require_dataset(
                    name=arr_name,
                    shape=(len(lowdim_data['timestamps']), oh, ow, 3),
                    chunks=(1, oh, ow, 3),
                    dtype=np.uint8
                )

            arr = replay_buffer.data[arr_name]

            for frame_idx, frame in enumerate(container.decode(video=0)):
                img = image_tf(frame.to_ndarray(format='bgr24'))
                arr[frame_idx] = img

    # 将lowdim数据添加到replay_buffer中
    for key, value in lowdim_data.items():
        replay_buffer.data[key] = value

    return replay_buffer

def main():
    # 设置数据集路径和输出存储路径
    dataset_path = 'data/datasets/pusht_real'  # 替换为实际数据集路径
    out_store = 'data/datasets/new_pusht_real'  # 替换为实际输出存储路径

    # 处理视频并存储到Zarr文件中
    replay_buffer = real_data_to_replay_buffer(dataset_path=dataset_path, out_store=out_store)

    print("视频处理完成，数据已存储到:", out_store)

if __name__ == "__main__":
    main()