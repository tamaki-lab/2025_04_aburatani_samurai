import cv2
import gc
import numpy as np
import os
import os.path as osp
import torch
from sam2.build_sam import build_sam2_video_predictor  # pylint: disable=import-error, no-name-in-module

def visualize_and_save(video_folder, prompts, model_cfg, checkpoint, vis_folder, save_to_video=True):
    # 動画のフレームフォルダを取得
    frame_files = sorted(os.listdir(video_folder))

    # 最初の画像の高さと幅を取得
    img = cv2.imread(osp.join(video_folder, frame_files[0]))
    height, width = img.shape[:2]

    # Predictorの初期化
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

    if save_to_video:
        # 可視化のためのフォルダ作成
        os.makedirs(vis_folder, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(osp.join(vis_folder, 'bedroom.mp4'), fourcc, 30, (width, height))

    predictions = []

    # 推論を行う
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(video_folder, offload_video_to_cpu=True, offload_state_to_cpu=True, async_loading_frames=True)

        # 最初のプロンプト（1番目の物体）で追跡開始
        for fid, (bbox, track_label) in enumerate(prompts[:1]):  # 1番目の物体だけ最初に追加
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=fid, obj_id=fid)

        # 50フレーム目まで追跡した後、2番目のオブジェクトを追加する処理
        for frame_idx in range(50, len(frame_files)):
            object_ids, masks = [], []
            if frame_idx == 50:  # 50フレーム目で2番目の物体を指定して追跡開始
                bbox, track_label = prompts[1]  # 2番目の物体のバウンディングボックス
                frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=frame_idx, obj_id=1)

            # 追跡を続ける
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                mask_to_vis = {}
                bbox_to_vis = {}

                # 各オブジェクトのマスクとバウンディングボックスを可視化
                for obj_id, mask in zip(object_ids, masks):
                    mask = mask[0].cpu().numpy()
                    mask = mask > 0.0
                    non_zero_indices = np.argwhere(mask)
                    if len(non_zero_indices) == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = non_zero_indices.min(axis=0).tolist()
                        y_max, x_max = non_zero_indices.max(axis=0).tolist()
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    bbox_to_vis[obj_id] = bbox
                    mask_to_vis[obj_id] = mask

                if save_to_video:
                    # フレーム画像の読み込み
                    img = cv2.imread(osp.join(video_folder, frame_files[frame_idx]))
                    if img is None:
                        break

                    # マスクを画像に重ねる
                    for obj_id in mask_to_vis.keys():
                        mask_img = np.zeros((height, width, 3), np.uint8)
                        mask_img[mask_to_vis[obj_id]] = (255, 0, 0)  # 赤色で表示
                        img = cv2.addWeighted(img, 1, mask_img, 0.75, 0)

                    # バウンディングボックスを画像に描画
                    for obj_id in bbox_to_vis.keys():
                        cv2.rectangle(img, (bbox_to_vis[obj_id][0], bbox_to_vis[obj_id][1]),
                                      (bbox_to_vis[obj_id][0] + bbox_to_vis[obj_id][2], bbox_to_vis[obj_id][1] + bbox_to_vis[obj_id][3]),
                                      (0, 255, 0), 2)  # 緑色で表示

                    out.write(img)

                predictions.append(bbox_to_vis)

    # 結果の保存（テキストファイル）
    pred_folder = "results/bedroom"
    os.makedirs(pred_folder, exist_ok=True)
    with open(osp.join(pred_folder, 'bedroom.txt'), 'w') as f:
        for pred in predictions:
            for obj_id, bbox in pred.items():
                x, y, w, h = bbox
                f.write(f"{x},{y},{w},{h}\n")

    if save_to_video:
        out.release()

    del predictor
    del state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


# --- Main Code ---
# 使用するモデル設定とチェックポイントのパス
model_cfg = "configs/samurai/sam2.1_hiera_b+.yaml"
# model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
checkpoint = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"

# プロンプト（バウンディングボックス）を定義
prompts = [
    ((300, 0, 500, 400), 0),  # 1番目のオブジェクト
    ((100, 100, 300, 450), 1),  # 2番目のオブジェクト
]

# ベッドルームの画像フォルダと保存先
video_folder = "data/bedroom"
vis_folder = "visualization/bedroom"

# 可視化と結果保存の実行
visualize_and_save(video_folder, prompts, model_cfg, checkpoint, vis_folder)
