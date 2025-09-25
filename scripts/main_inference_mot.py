import cv2
import gc
import numpy as np
import os
import os.path as osp
# import pdb
import torch
from sam2.build_sam import build_sam2_video_predictor  # pylint: disable=import-error, no-name-in-module
# from tqdm import tqdm


def id2color(tid: int):
    # 決定的, 目で区別しやすい軽い擬似ハッシュ
    r = (37 * (tid + 1)) % 255
    g = (17 * (tid + 2)) % 255
    b = (97 * (tid + 3)) % 255
    return (b, g, r)  # OpenCVはBGR


def load_mot_gt(gt_path):
    """
    gt.txt: frame,id,x,y,w,h,conf,cls,vis を
    prompts[frame] = [ ((x1,y1,x2,y2), id), ... ] にまとめる.
    あわせて各idの初登場(最小frame)でのbboxを init_all に返す.
    zero_index=True のとき, frameを0始まりへシフト.
    """
    prompts = {}         # frame(int) -> List[((x1,y1,x2,y2), id)]
    first_appear = {}    # id -> (frame, (x1,y1,x2,y2))

    if not osp.exists(gt_path):
        return {}, []  # GTなしの場合のフォールバック

    with open(gt_path, 'r') as f:
        for line in f:
            fr, tid, x, y, w, h, *_ = line.strip().split(',')
            fr = int(fr); tid = int(tid)
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
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
model_name = "base_plus"

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

    # Start processing frames
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frame_folder, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)

        # ground-truthの読み込みをpromptsに統一
        mot_gt = osp.join(video_folder, video.strip(), "gt", "gt.txt")
        prompts, init_all = load_mot_gt(mot_gt)  # ← frameは0始まり

        for tid, fr, bbox_xyxy in init_all:
            predictor.add_new_points_or_box(state,
                                            box=bbox_xyxy,
                                            frame_idx=fr,
                                            obj_id=tid)

        mot_rows = []  # (frame, id, x, y, w, h, score, -1, -1, -1)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
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
