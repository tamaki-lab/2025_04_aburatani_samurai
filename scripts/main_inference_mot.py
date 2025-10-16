import cv2
import gc
import numpy as np
import os
import os.path as osp
# import pdb
from comet_ml import Experiment
import torch
from sam2.build_sam import build_sam2_video_predictor  # pylint: disable=import-error, no-name-in-module
# from tqdm import tqdm
from typing import Dict, List, Tuple
import csv

import time
import psutil
try:
    import pynvml  # GPUメモリ取得に使う
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False


def get_gpu_mem_mb(device_index: int = 0):
    # NVMLが使えなければtorchの値でフォールバック
    if NVML_OK:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / (1024**2)
        total = info.total / (1024**2)
        return used, total
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{device_index}")
        used = torch.cuda.memory_allocated(dev) / (1024**2)
        reserved = torch.cuda.memory_reserved(dev) / (1024**2)
        total = torch.cuda.get_device_properties(dev).total_memory / (1024**2)
        # reservedも見たいなら別キーで出す
        return used, total
    return 0.0, 0.0


def get_cpu_mem_mb():
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / (1024**2), vm.total / (1024**2)


def id2color(tid: int):
    # 決定的, 目で区別しやすい軽い擬似ハッシュ
    r = (37 * (tid + 1)) % 255
    g = (17 * (tid + 2)) % 255
    b = (97 * (tid + 3)) % 255
    return (b, g, r)  # OpenCVはBGR


def load_mot_gt(gt_path: str) -> Tuple[
    Dict[int, List[Tuple[Tuple[int, int, int, int], int]]],
    List[Tuple[int, int, Tuple[int, int, int, int]]]
]:
    """
    gt.txt: frame,id,x,y,w,h,conf,cls,vis を
    prompts[frame] = [ ((x1,y1,x2,y2), id), ... ] にまとめる.
    あわせて各idの初登場(最小frame)でのbboxを init_all に返す.
    zero_index=True のとき, frameを0始まりへシフト.
    """
    prompts: Dict[int, List[Tuple[Tuple[int, int, int, int], int]]] = {}         # frame(int) -> List[((x1,y1,x2,y2), id)]
    first_appear = {}    # id -> (frame, (x1,y1,x2,y2))

    if not osp.exists(gt_path):
        return {}, []  # GTなしの場合のフォールバック

    with open(gt_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # 空行やコメント行スキップ
            if not row or (row[0].strip().startswith('#')):
                continue
            # 列数は最低6列あればOK(以降は無視)
            fr = int(float(row[0]))
            tid = int(float(row[1]))
            x = int(float(row[2]))
            y = int(float(row[3]))
            w = int(float(row[4]))
            h = int(float(row[5]))

            x1, y1, x2, y2 = x, y, x + w, y + h
            fr0 = fr - 1
            prompts.setdefault(fr0, []).append(((x1, y1, x2, y2), tid))

            if tid not in first_appear:
                first_appear[tid] = (fr, (x1, y1, x2, y2))

    # id順に初期化リストを作る
    init_all = []
    for tid in sorted(first_appear.keys()):
        fr, bbox = first_appear[tid]
        init_all.append((tid, fr, bbox))  # frは1始まりで保持しておく

    return prompts, init_all


color = [
    (255, 0, 0),
]

testing_set = "data/MOT17/testing_set.txt"
with open(testing_set, 'r') as f:
    test_videos = f.readlines()

exp_name = "samurai"
model_name = "tiny"

checkpoint = f"sam2/checkpoints/sam2.1_hiera_{model_name}.pt"
if model_name == "base_plus":
    model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
else:
    model_cfg = f"configs/samurai/sam2.1_hiera_{model_name[0]}.yaml"

video_folder = "data/MOT17/train"
pred_folder = f"results/{exp_name}/{exp_name}_{model_name}"

save_to_video = True
if save_to_video:
    vis_folder = f"visualization/{exp_name}/{model_name}"
    os.makedirs(vis_folder, exist_ok=True)
    vis_mask = {}
    vis_bbox = {}

test_videos = sorted(test_videos)
for vid, video in enumerate(test_videos):

    video_basename = video.strip()
    frame_folder = osp.join(video_folder, video.strip(), "img1")

    num_frames = len(os.listdir(osp.join(video_folder, video.strip(), "img1")))

    print(f"\033[91mRunning video [{vid + 1}/{len(test_videos)}]: {video} with {num_frames} frames\033[0m")

    height, width = cv2.imread(osp.join(frame_folder, "000001.jpg")).shape[:2]

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

    predictions = []

    if save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(osp.join(vis_folder, f'{video_basename}.mp4'), fourcc, 30, (width, height))

    # comet設定
    experiment = Experiment()
    experiment.set_name(f"{exp_name}-{model_name}-{video_basename}")
    experiment.log_parameters({
        "model_name": model_name,
        "sequence": video_basename,
        "fps_target": 30,
    })

    # Start processing frames
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frame_folder, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)

        # ground-truthの読み込みをpromptsに統一
        mot_gt = osp.join(video_folder, video.strip(), "gt", "gt.txt")
        prompts, init_all = load_mot_gt(mot_gt)  # ← frameは0始まり

        for tid, fr, bbox_xyxy in init_all:
            predictor.add_new_points_or_box(state,
                                            box=bbox_xyxy,
                                            frame_idx=fr - 1,
                                            obj_id=tid)

        mot_rows = []  # (frame, id, x, y, w, h, score, -1, -1, -1)

        try:
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                # --- ここから計測開始 ---
                t_step_start = time.perf_counter()

                # 画像のロード時間を計測
                t0 = time.perf_counter()
                img = cv2.imread(f'{frame_folder}/{frame_idx + 1:06d}.jpg')
                load_time = time.perf_counter() - t0
                if img is None:
                    # 読めなかったらログだけ入れてスキップ
                    gpu_used_mb, gpu_total_mb = get_gpu_mem_mb(0)
                    cpu_used_mb, cpu_total_mb = get_cpu_mem_mb()
                    experiment.log_metric("gpu_mem_used_mb", gpu_used_mb, step=frame_idx)
                    experiment.log_metric("cpu_mem_used_mb", cpu_used_mb, step=frame_idx)
                    experiment.log_metric("load_time_s", load_time, step=frame_idx)
                    experiment.log_metric("step_time_s", 0.0, step=frame_idx)
                    experiment.log_metric("num_ids", 0, step=frame_idx)
                    continue
                # --- ここまでロード計測 ---

                mask_to_vis = {}
                bbox_to_vis = {}

                # マルチID対応
                for obj_id, mask in zip(object_ids, masks):
                    m = mask[0].detach().cpu().numpy() > 0.0
                    nz = np.argwhere(m)
                    if nz.size == 0:
                        x, y, w, h = 0, 0, 0, 0
                    else:
                        y_min, x_min = nz.min(axis=0).tolist()
                        y_max, x_max = nz.max(axis=0).tolist()
                        x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

                    bbox_to_vis[obj_id] = [x, y, w, h]
                    mask_to_vis[obj_id] = m
                    mot_rows.append((frame_idx + 1, obj_id, x, y, w, h, 1, -1, -1, -1))

                if save_to_video:
                    # 桁数6でゼロパディングされた画像ファイル名を想定
                    img = cv2.imread(f'{frame_folder}/{frame_idx + 1:06d}.jpg')
                    if img is None:
                        break

                    # 半透明マスク
                    if len(mask_to_vis) > 0:
                        overlay = np.zeros_like(img)
                        for obj_id, m in mask_to_vis.items():
                            overlay[m] = id2color(obj_id)
                        img = cv2.addWeighted(img, 1.0, overlay, 0.65, 0)

                    # 予測bboxとIDラベル
                    for obj_id, (x, y, w, h) in bbox_to_vis.items():
                        if w > 0 and h > 0:
                            c = id2color(obj_id)
                            p1 = (int(x), int(y))
                            p2 = (int(x + w), int(y + h))
                            cv2.rectangle(img, p1, p2, c, 2, lineType=cv2.LINE_AA)
                            label = f'ID {obj_id}'
                            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            tx, ty = p1[0], max(0, p1[1] - th - 4)
                            cv2.rectangle(img, (tx, ty), (tx + tw + 6, ty + th + 6), c, -1)
                            cv2.putText(img, label, (tx + 3, ty + th + 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                    # GT bbox (緑)
                    if frame_idx in prompts:
                        for (gx1, gy1, gx2, gy2), _tid in prompts[frame_idx]:
                            cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2, cv2.LINE_AA)

                    out.write(img)

                predictions.append(bbox_to_vis)

                # --- ここからCometへログ送信 ---
                infer_time = time.perf_counter() - t_step_start
                gpu_used_mb, gpu_total_mb = get_gpu_mem_mb(0)
                cpu_used_mb, cpu_total_mb = get_cpu_mem_mb()

                experiment.log_metric("gpu_mem_used_mb", gpu_used_mb, step=frame_idx)
                experiment.log_metric("gpu_mem_total_mb", gpu_total_mb, step=frame_idx)
                experiment.log_metric("cpu_mem_used_mb", cpu_used_mb, step=frame_idx)
                experiment.log_metric("cpu_mem_total_mb", cpu_total_mb, step=frame_idx)
                experiment.log_metric("load_time_s", load_time, step=frame_idx)
                experiment.log_metric("step_time_s", infer_time, step=frame_idx)
                experiment.log_metric("num_ids", len(object_ids), step=frame_idx)

                # IDの中身はノイジーになりやすいので間引いてテキストで
                if frame_idx % 10 == 0:
                    experiment.log_text(f"frame={frame_idx}, ids={list(map(int, object_ids))}", step=frame_idx)

                # 任意: 可視化フレームをたまに投げる(頻度に注意)
                # if save_to_video and frame_idx % 100 == 0:
                #     experiment.log_image(img[:, :, ::-1], name=f"frame_{frame_idx:06d}.png", step=frame_idx)
                # --- ログここまで ---
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                # どのフレームで落ちたかを記録
                experiment.log_metric("oom_at_frame", frame_idx)
                experiment.log_other("exception", "CUDA OOM")
                experiment.log_other("exception_msg", str(e)[:500])
                raise
        finally:
            experiment.end()

    # 予測保存
    mot_rows.sort(key=lambda r: (r[0], r[1]))
    os.makedirs(pred_folder, exist_ok=True)
    with open(osp.join(pred_folder, f'{video_basename}.txt'), 'w') as f:
        for r in mot_rows:
            f.write(','.join(map(str, r)) + '\n')

    if save_to_video:
        out.release()

    del predictor
    del state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()
