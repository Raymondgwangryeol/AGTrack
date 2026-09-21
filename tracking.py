import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess
from ByteTrack.yolox.utils.visualize import plot_tracking
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer
from AGPredictor import VideoRecognizer

from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict
from datetime import timedelta

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

people = []
age_gender_cache = {}


class ObjectTracker:
    def __init__(
        self,
        exp_file,
        ckpt=None,
        device="gpu",
        fp16=False,
        fuse=False,
        trt=False,
        conf=None,
        nms=None,
        tsize=None,
        track_thresh=0.58198,
        track_buffer=60,
        match_thresh=0.76091,
        aspect_ratio_thresh=1.33903,
        min_box_area=5,
        mot20=True,
        fps=30,
    ):
        self.device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
        self.fp16 = fp16
        self.exp = exp_file
        if conf:
            self.exp.test_conf = conf
        if nms:
            self.exp.nmsthre = nms
        if tsize:
            self.exp.test_size = (tsize, tsize)

        self.model = self.exp.get_model().to(self.device)
        self.model.eval()

        if fuse:
            self.model = fuse_model(self.model)

        logger.info("Loading checkpoint")
        ckpt_file = torch.load(ckpt, map_location=self.device)
        model_state = ckpt_file["model"]
        filtered_ckpt = {k: v for k, v in model_state.items() if "cls_preds" not in k}
        self.model.load_state_dict(filtered_ckpt, strict=False)
        logger.info("Checkpoint loaded.")

        self.predictor = self._build_predictor(trt)
        self.tracker = BYTETracker(
            type("Args", (), {
                "track_thresh": track_thresh,
                "track_buffer": track_buffer,
                "match_thresh": match_thresh,
                "aspect_ratio_thresh": aspect_ratio_thresh,
                "min_box_area": min_box_area,
                "mot20": mot20,
            }),
            frame_rate=fps
        )
        self.timer = Timer()

    def _build_predictor(self, trt):
        class Predictor:
            def __init__(self, model, exp, device, fp16):
                self.model = model
                self.num_classes = exp.num_classes
                self.confthre = exp.test_conf
                self.nmsthre = exp.nmsthre
                self.test_size = exp.test_size
                self.device = device
                self.fp16 = fp16
                self.rgb_means = (0.485, 0.456, 0.406)
                self.std = (0.229, 0.224, 0.225)
                self.age_gender_recognizer = VideoRecognizer()

            def inference(self, img, timer):
                img_info = {"id": 0}
                if isinstance(img, str):
                    img_info["file_name"] = osp.basename(img)
                    img = cv2.imread(img)
                else:
                    img_info["file_name"] = None

                height, width = img.shape[:2]
                img_info["height"] = height
                img_info["width"] = width
                img_info["raw_img"] = img

                img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
                img_info["ratio"] = ratio
                img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
                if self.fp16:
                    img = img.half()

                with torch.no_grad():
                    timer.tic()
                    outputs = self.model(img)
                    outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                    timer.toc()
                return outputs, img_info

        return Predictor(self.model, self.exp, self.device, self.fp16)

    def tlwh_to_xyxy(self, tlwh):
        x, y, w, h = tlwh
        return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]

    def update_cache_batch(self, tids, ages, genders, embed_vecs, frame_id):
        """
        현재 프레임 N명의 결과를 Similarity Matrix로 한 번에 처리.

        1. 기존 tid → [K, K] Similarity Matrix 대각선으로 self-similarity 검증 후 캐시 업데이트
        2. 신규 tid → 사라진 트랙과 [U, L] Re-ID Similarity Matrix 매칭 시도
        """
        global age_gender_cache

        MAX_HISTORY     = 30       # 이력 더 많이 유지
        SIM_THRESHOLD   = 0.82    
        REID_THRESHOLD  = 0.80    # Re-ID 기준도 살짝 낮춤
        MAX_LOST_FRAMES = 170    

        if not tids:
            return

        new_embeds = np.stack(embed_vecs)  # [N, D]

        # 현재 프레임에 등장한 tid를 active로 표시
        active_tids = set(tids)
        for tid in age_gender_cache:
            age_gender_cache[tid]["active"] = tid in active_tids

        # 기존 캐시에 있는 tid와 없는 tid 분리
        known_mask   = [i for i, t in enumerate(tids) if t in age_gender_cache]
        unknown_mask = [i for i, t in enumerate(tids) if t not in age_gender_cache]

        # ① 기존 tid: 유사도 검증 후 업데이트
        if known_mask:
            known_tids   = [tids[i] for i in known_mask]
            known_embeds = np.stack([age_gender_cache[t]["embedding"] for t in known_tids])
            query_embeds = new_embeds[known_mask]

            sim_matrix = query_embeds @ known_embeds.T
            self_sim   = np.diag(sim_matrix)

            for idx, (tid, sim) in enumerate(zip(known_tids, self_sim)):
                i    = known_mask[idx]
                prev = age_gender_cache[tid]

                if sim >= SIM_THRESHOLD:
                    # ✅ 같은 사람: 이력 누적 + 임베딩 EMA 업데이트 (갑작스러운 변화 완화)
                    alpha = 0.7  # 새 임베딩 반영 비율
                    blended = alpha * embed_vecs[i] + (1 - alpha) * prev["embedding"]
                    blended = blended / (np.linalg.norm(blended) + 1e-8)

                    prev["ages"].append(ages[i])
                    prev["genders"].append(genders[i])
                    prev["embedding"]  = blended       # 급격한 변화 방지
                    prev["last_frame"] = frame_id
                    prev["ages"]    = prev["ages"][-MAX_HISTORY:]
                    prev["genders"] = prev["genders"][-MAX_HISTORY:]
                else:
                    # ⚠️ 유사도 낮음 → 초기화 대신 이력은 유지하고 임베딩만 조심스럽게 업데이트
                    # (조명·자세 변화일 수 있으므로 바로 버리지 않음)
                    prev["ages"].append(ages[i])
                    prev["genders"].append(genders[i])
                    prev["last_frame"] = frame_id
                    prev["ages"]    = prev["ages"][-MAX_HISTORY:]
                    prev["genders"] = prev["genders"][-MAX_HISTORY:]
                    # 임베딩은 보수적으로만 업데이트
                    alpha = 0.3
                    blended = alpha * embed_vecs[i] + (1 - alpha) * prev["embedding"]
                    prev["embedding"] = blended / (np.linalg.norm(blended) + 1e-8)

        # ② 신규 tid: 사라진 트랙과 Re-ID 매칭
        if unknown_mask:
            lost_tids = [
                t for t, info in age_gender_cache.items()
                if not info["active"]
                and (frame_id - info["last_frame"]) <= MAX_LOST_FRAMES
            ]

            query_embeds = new_embeds[unknown_mask]

            if lost_tids:
                lost_embeds = np.stack([age_gender_cache[t]["embedding"] for t in lost_tids])
                reid_sim_matrix = query_embeds @ lost_embeds.T

                best_match_inds = np.argmax(reid_sim_matrix, axis=1)
                best_match_sims = reid_sim_matrix[
                    np.arange(len(unknown_mask)), best_match_inds
                ]

                matched_lost_tids = set()

                for idx, i in enumerate(unknown_mask):
                    tid       = tids[i]
                    best_sim  = best_match_sims[idx]
                    best_lost = lost_tids[best_match_inds[idx]]

                    if best_sim >= REID_THRESHOLD and best_lost not in matched_lost_tids:
                        # Re-ID 성공
                        recovered = age_gender_cache[best_lost]
                        recovered["ages"].append(ages[i])
                        recovered["genders"].append(genders[i])
                        recovered["embedding"]  = embed_vecs[i]
                        recovered["last_frame"] = frame_id
                        recovered["active"]     = True
                        age_gender_cache[tid]   = recovered
                        del age_gender_cache[best_lost]
                        matched_lost_tids.add(best_lost)
                    else:
                        # 신규 등록
                        age_gender_cache[tid] = {
                            "embedding":  embed_vecs[i],
                            "ages":       [ages[i]],
                            "genders":    [genders[i]],
                            "last_frame": frame_id,
                            "active":     True,
                        }
            else:
                for i in unknown_mask:
                    tid = tids[i]
                    age_gender_cache[tid] = {
                        "embedding":  embed_vecs[i],
                        "ages":       [ages[i]],
                        "genders":    [genders[i]],
                        "last_frame": frame_id,
                        "active":     True,
                    }

    def get_stable_age_gender(self, tid):
        global age_gender_cache
        if tid not in age_gender_cache:
            return None, None

        ages    = age_gender_cache[tid]["ages"]
        genders = age_gender_cache[tid]["genders"]

        valid_ages    = [a for a in ages    if a is not None and a > 0]
        valid_genders = [g for g in genders if g is not None and g in ["male", "female"]]

        avg_age     = int(sum(valid_ages) / len(valid_ages)) if valid_ages else None
        mode_gender = Counter(valid_genders).most_common(1)[0][0] if valid_genders else None

        return avg_age, mode_gender

    def process_image(self, img, frame_id=0):
        global age_gender_cache

        outputs, img_info = self.predictor.inference(img, self.timer)
        online_img = img_info["raw_img"].copy()
        current_frame_results      = []
        current_frame_tracked_pids = set()

        if outputs[0] is None:
            return online_img, current_frame_results, current_frame_tracked_pids

        online_targets = self.tracker.update(
            outputs[0], [img_info['height'], img_info['width']], self.exp.test_size
        )

        # ── 1단계: 필터링 + 크롭 수집 ────────────────────────────────
        valid_targets = []  # (t, tid, bbox_tlwh)
        crops = []
        online_tlwhs, online_ids, online_scores = [], [], []

        for t in online_targets:
            tlwh = t.tlwh
            tid  = t.track_id
            if tlwh[2] <= 0 or tlwh[3] <= 0:
                continue
            vertical = tlwh[2] / tlwh[3] > self.tracker.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] <= self.tracker.min_box_area or vertical:
                continue

            bbox_tlwh = [round(x, 2) for x in tlwh]
            xyxy      = self.tlwh_to_xyxy(bbox_tlwh)
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_info["width"], x2); y2 = min(img_info["height"], y2)
            crop = img_info["raw_img"][y1:y2, x1:x2]

            if crop.shape[0] <= 0 or crop.shape[1] <= 0:
                continue

            valid_targets.append((t, tid, bbox_tlwh))
            crops.append(crop)  # resize는 prepare_classification_images 내부에서 처리
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)

        # ── 2단계: 배치 추론 (GPU 왕복 1번) ──────────────────────────
        batch_results = self.predictor.age_gender_recognizer.run_batch(crops)

        # ── 3단계: Similarity Matrix 기반 캐시 일괄 업데이트 ─────────
        tids_batch    = [tid for _, tid, _ in valid_targets]
        ages_batch    = [r[0] for r in batch_results]
        genders_batch = [r[1] for r in batch_results]
        embeds_batch  = [r[2] for r in batch_results]

        self.update_cache_batch(tids_batch, ages_batch, genders_batch, embeds_batch, frame_id)

        # ── 4단계: 결과 취합 ──────────────────────────────────────────
        for t, tid, bbox_tlwh in valid_targets:
            stable_age, stable_gender = self.get_stable_age_gender(tid)
            current_frame_results.append({
                "id":     tid,
                "bbox":   bbox_tlwh,
                "score":  round(t.score, 2),
                "age":    stable_age,
                "gender": stable_gender,
            })
            current_frame_tracked_pids.add(tid)

        # ── 5단계: 시각화 ─────────────────────────────────────────────
        fps_text = 1. / self.timer.average_time if self.timer.average_time > 0 else 0
        online_img = plot_tracking(
            img_info["raw_img"].copy(), online_tlwhs, online_ids,
            frame_id=frame_id, fps=fps_text, age_gender_info=age_gender_cache
        )

        return online_img, current_frame_results, current_frame_tracked_pids

    def process_video(self, video_path, save_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video {video_path}")

        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.tracker.frame_rate = fps

        vid_writer = None
        if save_path:
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        overall_people_info                  = defaultdict(dict)
        hourly_age_distribution_per_interval = []
        hourly_inflow_outflow_per_interval   = []

        last_interval_time_sec = 0.0
        interval_length_sec    = 10
        currently_tracked_people_status = {}

        for frame_id in tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                break

            current_msec     = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_time_sec = current_msec / 1000.0

            online_img, current_frame_tracked_data, pids_in_current_frame = self.process_image(frame, frame_id=frame_id + 1)

            prev_active_pids = set(currently_tracked_people_status.keys())

            for person_data in current_frame_tracked_data:
                tid           = person_data["id"]
                stable_age    = person_data["age"]
                stable_gender = person_data["gender"]

                overall_people_info[tid]["age"]    = stable_age
                overall_people_info[tid]["gender"] = stable_gender

                if tid not in currently_tracked_people_status:
                    currently_tracked_people_status[tid] = {
                        "entry_time_sec":              current_time_sec,
                        "last_seen_time_sec":          current_time_sec,
                        "age":                         stable_age,
                        "gender":                      stable_gender,
                        "counted_inflow_in_interval":  False,
                    }
                else:
                    currently_tracked_people_status[tid]["last_seen_time_sec"] = current_time_sec
                    currently_tracked_people_status[tid]["age"]                = stable_age
                    currently_tracked_people_status[tid]["gender"]             = stable_gender

            for pid in list(currently_tracked_people_status.keys()):
                if pid not in pids_in_current_frame:
                    overall_people_info[pid]["exit_time_sec"] = current_time_sec
                    del currently_tracked_people_status[pid]

            if current_time_sec - last_interval_time_sec >= interval_length_sec or frame_id == total_frames - 1:
                interval_start_sec = last_interval_time_sec
                interval_end_sec   = current_time_sec

                interval_age_counts = defaultdict(int)
                for pid, info in overall_people_info.items():
                    entry_time = info.get("entry_time_sec", -1)
                    exit_time  = info.get("exit_time_sec", float('inf'))
                    if not (exit_time <= interval_start_sec or entry_time >= interval_end_sec):
                        age = info.get("age", -1)
                        if age is not None and age > 0:
                            age_bin = f"{(age // 10) * 10}s"
                            interval_age_counts[age_bin] += 1

                hourly_age_distribution_per_interval.append({
                    "timestamp_sec": interval_end_sec,
                    "age_counts":    dict(interval_age_counts),
                })

                interval_inflows  = []
                interval_outflows = []

                for pid, info in currently_tracked_people_status.items():
                    if (info["entry_time_sec"] >= interval_start_sec and
                            info["entry_time_sec"] <= interval_end_sec and
                            not info["counted_inflow_in_interval"]):
                        interval_inflows.append({
                            "pid":            pid,
                            "age":            info.get("age"),
                            "gender":         info.get("gender"),
                            "entry_time_sec": info["entry_time_sec"],
                        })
                        currently_tracked_people_status[pid]["counted_inflow_in_interval"] = True

                for pid, info in overall_people_info.items():
                    if ("exit_time_sec" in info and
                            info["exit_time_sec"] >= interval_start_sec and
                            info["exit_time_sec"] <= interval_end_sec and
                            pid not in currently_tracked_people_status):
                        interval_outflows.append({
                            "pid":           pid,
                            "age":           info.get("age"),
                            "gender":        info.get("gender"),
                            "exit_time_sec": info["exit_time_sec"],
                        })

                hourly_inflow_outflow_per_interval.append({
                    "timestamp_sec": interval_end_sec,
                    "inflows":       interval_inflows,
                    "outflows":      interval_outflows,
                })

                last_interval_time_sec = current_time_sec

            if vid_writer:
                vid_writer.write(online_img)

        cap.release()
        if vid_writer:
            vid_writer.release()

        for pid, info in currently_tracked_people_status.items():
            if "exit_time_sec" not in overall_people_info[pid]:
                overall_people_info[pid]["exit_time_sec"] = current_time_sec

        return overall_people_info, hourly_age_distribution_per_interval, hourly_inflow_outflow_per_interval
