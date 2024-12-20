import av
import os
import numpy as np
import pathlib
import multiprocessing
from Diffusion_Policy.diffusion_policy.common.cv2_util import get_image_transform
from Diffusion_Policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from Diffusion_Policy.diffusion_policy.real_world.video_recorder import read_video

import concurrent.futures
from tqdm import tqdm
import zarr

def process_and_store_videos(dataset_path, out_store_path):
    input = pathlib.Path(os.path.expanduser(dataset_path))
    in_replay_buffer_dir = input.joinpath('replay_buffer.zarr')
    in_video_dir = input.joinpath('videos')

    assert in_replay_buffer_dir.is_dir()
    assert in_video_dir.is_dir()

    in_replay_buffer = ReplayBuffer.create_from_path(in_replay_buffer_dir, mode='r')
    
    # save lowdim data to single chunk
    chunks_map = dict()
    compressor_map = dict()
    for key, value in in_replay_buffer.data.items():
        chunks_map[key] = value.shape
        compressor_map[key] = None

    lowdim_keys = list(in_replay_buffer.data.keys())
    image_keys = ['camera_0', 'camera_1', 'camera_2', 'camera_3', 'camera_4']
    out_resolutions = (1280, 720)
    n_decoding_threads = multiprocessing.cpu_count()
    n_encoding_threads = multiprocessing.cpu_count()
    max_inflight_tasks = multiprocessing.cpu_count()*5

    print('Loading lowdim data')

    # 创建一个 Zarr DirectoryStore
    out_store = zarr.DirectoryStore(out_store_path) 
    out_replay_buffer = ReplayBuffer.copy_from_store(
        src_store=in_replay_buffer.root.store,
        store=out_store,
        keys=lowdim_keys,
        chunks=chunks_map,
        compressors=compressor_map
        )

    # worker function
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False


    # 复制低维数据
    n_cameras = 0
    camera_idxs = set() 

    # estimate number of cameras
    # 从视频文件中提取相机索引，0、1、2、3 ***
    if image_keys is not None:
        n_cameras = len(image_keys)
        # 获取 image_keys 中的相机索引，0、1、2、3
        camera_idxs = set(int(x.split('_')[-1]) for x in image_keys)
    else:
        # estimate number of cameras
        # 从视频文件中提取相机索引，0、1、2、3 ***
        episode_video_dir = in_video_dir.joinpath(str(0))
        episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
        camera_idxs = set(int(x.stem) for x in episode_video_paths)
        n_cameras = len(episode_video_paths)
    
    # 从 replay_buffer 中获取 n_steps、episode_starts、episode_lengths、timestamps ，用于后续处理视频
    n_steps = in_replay_buffer.n_steps
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    episode_lengths = in_replay_buffer.episode_lengths
    timestamps = in_replay_buffer['timestamp'][:]
    dt = timestamps[1] - timestamps[0]



    # 使用 tqdm 创建进度条，显示图像数据加载进度
    with tqdm(total=n_steps*n_cameras, desc="Loading image data", mininterval=1.0) as pbar:
        # one chunk per thread, therefore no synchronization needed
        # 创建线程池，用于并行处理视频，提高处理速度
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = set()
            for episode_idx, episode_length in enumerate(episode_lengths):
                episode_video_dir = in_video_dir.joinpath(str(episode_idx))
                episode_start = episode_starts[episode_idx]

                episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
                #文件名是整数，所以按整数排序
                this_camera_idxs = set(int(x.stem) for x in episode_video_paths)
                if image_keys is None:
                    for i in this_camera_idxs - camera_idxs:
                        print(f"Unexpected camera {i} at episode {episode_idx}")
                for i in camera_idxs - this_camera_idxs:
                    print(f"Missing camera {i} at episode {episode_idx}")
                    if image_keys is not None:
                        raise RuntimeError(f"Missing camera {i} at episode {episode_idx}")

                for video_path in episode_video_paths:
                    camera_idx = int(video_path.stem)
                    if image_keys is not None:
                        # if image_keys provided, skip not used cameras
                        if camera_idx not in camera_idxs:
                            continue




                    # read resolution 读取视频的分辨率
                    with av.open(str(video_path.absolute())) as container:
                        video = container.streams.video[0]
                        vcc = video.codec_context
                        this_res = (vcc.width, vcc.height)
                    in_img_res = this_res

                    arr_name = f'camera_{camera_idx}'

                    # figure out save resolution
                    out_img_res = in_img_res
                    if isinstance(out_resolutions, dict):
                        if arr_name in out_resolutions:
                            out_img_res = tuple(out_resolutions[arr_name])
                    elif out_resolutions is not None:
                        out_img_res = tuple(out_resolutions)

                    # allocate array
                    if arr_name not in out_replay_buffer:
                        ow, oh = out_img_res
                        _ = out_replay_buffer.data.require_dataset(
                            name=arr_name,
                            shape=(n_steps,oh,ow,3),
                            chunks=(1,oh,ow,3),
                            compressor=None,
                            dtype=np.uint8
                        )
                    arr = out_replay_buffer[arr_name]

                    image_tf = get_image_transform(
                        input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False)
                    for step_idx, frame in enumerate(read_video(
                            video_path=str(video_path),
                            dt=dt,
                            img_transform=image_tf,
                            thread_type='FRAME',
                            thread_count=n_decoding_threads
                        )):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))
                        
                        global_idx = episode_start + step_idx
                        futures.add(executor.submit(put_img, arr, global_idx, frame))


                        if step_idx == (episode_length - 1):
                            break

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))

    out_replay_buffer.save_to_path(out_store_path)

def main():
    dataset_path = 'data/datasets/pusht_real'  # 替换为实际数据集路径
    out_store_path = 'data/datasets/pusht_real_processed'  # 替换为实际输出存储路径

    process_and_store_videos(dataset_path, out_store_path)
    print("视频处理完成，数据已存储到:", out_store_path)

if __name__ == "__main__":
    main()