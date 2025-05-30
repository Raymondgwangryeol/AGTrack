import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from ByteTrack.yolox.data.data_augment import preproc
from ByteTrack.yolox.exp import get_exp # MyExp는 여기서 가져오는 exp 클래스를 상속받음
from ByteTrack.yolox.utils import fuse_model, get_model_info, postprocess
from ByteTrack.yolox.utils.visualize import plot_tracking
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer
from AGPredictor import VideoRecognizer # MiVOLO 대신 사용하시는 AGPredictor

from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict # defaultdict 추가
from datetime import timedelta # 시간 계산을 위해 추가

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global variables for caching age/gender inference (consider passing as instance attributes)
# For better encapsulation, it's generally better to make these instance attributes if possible.
# For now, we'll keep them as global as per your original code structure.
people = [] # This seems to be a list of all detected instances, consider if you want a final aggregated list.
age_gender_cache = {} # Used for stabilizing age/gender for each track ID


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
        track_thresh=0.6564,
        track_buffer=60,
        match_thresh=0.9466,
        aspect_ratio_thresh=1.362,
        min_box_area=20,
        mot20=True,
        fps=30, # Initial FPS value, will be updated from video
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
        # ⚠️ class predictor는 버린다
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
            frame_rate=fps # This will be set more accurately in process_video
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
                # Your AGPredictor is correctly initialized here
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
    

    # 새로운 추론 결과 업데이트 함수
    def update_age_gender_cache(self, tid, age, gender, embed_vec, frame_id):
        global age_gender_cache # Access global cache
        MAX_HISTORY = 10  # 최근 10개 유지

        # 기존 로직 유지: 5프레임마다 또는 새로운 ID일 경우 캐시 갱신/초기화
        if frame_id % 5 == 0 or tid not in age_gender_cache:
            age_gender_cache[tid] = {
                "embedding": embed_vec,
                "ages": [age],
                "genders": [gender]
            }
        else:
            prev = age_gender_cache[tid]
            similarity = np.dot(prev["embedding"], embed_vec)

            if similarity >= 0.9: # 유사도가 높으면 기존 이력에 추가
                prev["ages"].append(age)
                prev["genders"].append(gender)
                prev["embedding"] = embed_vec # 최신 임베딩으로 업데이트
            else: # 유사하지 않으면 새로 덮어쓰기 (새로운 얼굴로 간주)
                age_gender_cache[tid] = {
                    "embedding": embed_vec,
                    "ages": [age],
                    "genders": [gender]
                }

            # 최근 MAX_HISTORY개로 제한
            prev["ages"] = prev["ages"][-MAX_HISTORY:]
            prev["genders"] = prev["genders"][-MAX_HISTORY:]

    # 안정화된 age, gender 반환
    def get_stable_age_gender(self, tid):
        global age_gender_cache # Access global cache
        if tid not in age_gender_cache:
            return None, None

        ages = age_gender_cache[tid]["ages"]
        genders = age_gender_cache[tid]["genders"]

        valid_ages = [a for a in ages if a is not None and a > 0] # Filter out None or invalid ages
        valid_genders = [g for g in genders if g is not None and g in ["male", "female"]]

        avg_age = int(sum(valid_ages) / len(valid_ages)) if valid_ages else None
        mode_gender = Counter(valid_genders).most_common(1)[0][0] if valid_genders else None

        return avg_age, mode_gender
    
    # process_image 메서드는 기존 로직을 유지하고, age/gender 추론 결과를 반환하도록 함
    def process_image(self, img, frame_id=0):
        # global people # This global list should ideally be handled within process_video for aggregation
        global age_gender_cache
        
        outputs, img_info = self.predictor.inference(img, self.timer)
        
        # Initialize online_img with raw_img to ensure it's always set
        online_img = img_info["raw_img"].copy() 
        current_frame_tracked_pids = set() # Track PIDs seen in current frame
        
        # This list will hold results specific to the current frame for `results` return
        current_frame_results = [] 

        if outputs[0] is not None:
            online_targets = self.tracker.update(
                outputs[0], [img_info['height'], img_info['width']], self.exp.test_size
            )
            online_tlwhs, online_ids, online_scores = [], [], []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                
                # Check for valid bounding box dimensions (width and height must be positive)
                if tlwh[2] <= 0 or tlwh[3] <= 0:
                    continue

                vertical = tlwh[2] / tlwh[3] > self.tracker.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.tracker.min_box_area and not vertical: # Apply area and aspect ratio filters
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    
                    bbox_tlwh = [round(x, 2) for x in tlwh]
                    xyxy = self.tlwh_to_xyxy(bbox_tlwh)
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Ensure crop coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_info["width"], x2)
                    y2 = min(img_info["height"], y2)

                    image_cropped = img_info["raw_img"][y1:y2, x1:x2] # Use raw_img for cropping
                    
                    age, gender, embed_vec = None, None, None # Initialize
                    if image_cropped.shape[0] > 0 and image_cropped.shape[1] > 0: # Ensure valid crop
                        image_resized = cv2.resize(image_cropped, (224, 224))

                        # 나이/성별 추론
                        age_gender_embed = self.predictor.age_gender_recognizer.run(image_resized)
                        if age_gender_embed:
                            age, gender, embed_vec = age_gender_embed[0]
                            if embed_vec is not None: # Ensure embedding is valid
                                embed_vec = embed_vec / np.linalg.norm(embed_vec) # Normalize embedding

                                self.update_age_gender_cache(tid, age, gender, embed_vec, frame_id)

                    stable_age, stable_gender = self.get_stable_age_gender(tid)

                    # Add to current frame results
                    current_frame_results.append({
                        "id": tid,
                        "bbox": bbox_tlwh, # Use TLWH for consistency with ByteTrack outputs
                        "score": round(t.score, 2),
                        "age": stable_age,
                        "gender": stable_gender,
                    })
                    current_frame_tracked_pids.add(tid) # Mark as seen in this frame

            # Plot tracking results on the image
            fps_text = 1. / self.timer.average_time if self.timer.average_time > 0 else 0
            online_img = plot_tracking(
                img_info["raw_img"].copy(), online_tlwhs, online_ids,
                frame_id=frame_id,
                fps=fps_text,
                age_gender_info=age_gender_cache # Pass the cache for drawing labels
            )
        
        # Return processed image and current frame's tracking results
        return online_img, current_frame_results, current_frame_tracked_pids

    def process_video(self, video_path, save_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # IMPORTANT: Update the tracker's frame rate based on the video's actual FPS
        self.tracker.frame_rate = fps 

        vid_writer = None
        if save_path:
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        # Initialize data structures for overall and time-based statistics
        overall_people_info = defaultdict(dict) # Stores final aggregated info for each unique person ID
        
        hourly_age_distribution_per_interval = [] # List of dicts: {"timestamp_sec": float, "age_counts": dict}
        hourly_inflow_outflow_per_interval = [] # List of dicts: {"timestamp_sec": float, "inflows": list, "outflows": list}

        last_interval_time_sec = 0.0 # Time when the last interval ended (seconds)
        interval_length_sec = 10 # Interval duration for statistics (10 seconds)
        
        # Store currently tracked people's status for inflow/outflow logic
        # {pid: {"entry_time_sec": float, "last_seen_time_sec": float, "age": int, "gender": str}}
        currently_tracked_people_status = {} 


        for frame_id in tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                break

            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_time_sec = current_msec / 1000.0

            # Process the current frame
            # online_img: frame with plotted detections
            # current_frame_tracked_data: list of dicts for people in this frame (id, bbox, score, age, gender)
            # pids_in_current_frame: set of track IDs present in this frame
            online_img, current_frame_tracked_data, pids_in_current_frame = self.process_image(frame, frame_id=frame_id + 1)
            
            # --- Update overall_people_info and handle inflow/outflow ---
            
            # Identify people who just appeared (inflow) or disappeared (outflow)
            # PIDs that were tracked in the previous frame but not in the current frame
            # are considered potential outflows.
            
            # Store PIDs that were active in the previous frame (before current frame's processing)
            prev_active_pids = set(currently_tracked_people_status.keys())

            # Update currently_tracked_people_status with current frame's data
            for person_data in current_frame_tracked_data:
                tid = person_data["id"]
                stable_age = person_data["age"]
                stable_gender = person_data["gender"]
                
                # Update overall_people_info with latest stable age/gender
                overall_people_info[tid]["age"] = stable_age
                overall_people_info[tid]["gender"] = stable_gender
                
                if tid not in currently_tracked_people_status:
                    # New person (inflow)
                    currently_tracked_people_status[tid] = {
                        "entry_time_sec": current_time_sec,
                        "last_seen_time_sec": current_time_sec,
                        "age": stable_age,
                        "gender": stable_gender,
                        "counted_inflow_in_interval": False # Flag to avoid double counting inflow in the same interval
                    }
                else:
                    # Existing person, just update last seen time and demographics
                    currently_tracked_people_status[tid]["last_seen_time_sec"] = current_time_sec
                    currently_tracked_people_status[tid]["age"] = stable_age # Update with latest stable age
                    currently_tracked_people_status[tid]["gender"] = stable_gender # Update with latest stable gender

            # Mark people who are no longer being tracked (outflow)
            for pid in list(currently_tracked_people_status.keys()): # Iterate over a copy of keys
                if pid not in pids_in_current_frame: # If person disappeared from current frame
                    # Record exit time in overall_people_info
                    overall_people_info[pid]["exit_time_sec"] = current_time_sec
                    # Remove from currently_tracked_people_status
                    del currently_tracked_people_status[pid]

            # --- Interval-based data aggregation ---
            if current_time_sec - last_interval_time_sec >= interval_length_sec or frame_id == total_frames - 1:
                interval_start_sec = last_interval_time_sec
                interval_end_sec = current_time_sec

                # 1. Hourly Customer Age Distribution (10초마다)
                interval_age_counts = defaultdict(int)
                for pid, info in overall_people_info.items(): # Consider all known people
                    # Only count if the person was active within this interval
                    entry_time = info.get("entry_time_sec", -1)
                    exit_time = info.get("exit_time_sec", float('inf')) # If not exited yet, assume still present
                    
                    # Check for overlap with current interval
                    if not (exit_time <= interval_start_sec or entry_time >= interval_end_sec):
                        age = info.get("age", -1)
                        if age is not None and age > 0:
                            age_bin = f"{(age // 10) * 10}s"
                            interval_age_counts[age_bin] += 1
                
                hourly_age_distribution_per_interval.append({
                    "timestamp_sec": interval_end_sec,
                    "age_counts": dict(interval_age_counts)
                })

                # 2. Hourly Customer Inflow Information (10초마다)
                interval_inflows = []
                interval_outflows = []

                # Inflow: People whose entry_time_sec falls within this interval AND haven't been counted yet
                for pid, info in currently_tracked_people_status.items():
                    if info["entry_time_sec"] >= interval_start_sec and \
                       info["entry_time_sec"] <= interval_end_sec and \
                       not info["counted_inflow_in_interval"]:
                        interval_inflows.append({
                            "pid": pid,
                            "age": info.get("age"),
                            "gender": info.get("gender"),
                            "entry_time_sec": info["entry_time_sec"]
                        })
                        currently_tracked_people_status[pid]["counted_inflow_in_interval"] = True # Mark as counted
                
                # Outflow: People who registered an exit_time_sec within this interval
                for pid, info in overall_people_info.items():
                    if "exit_time_sec" in info and \
                       info["exit_time_sec"] >= interval_start_sec and \
                       info["exit_time_sec"] <= interval_end_sec and \
                       pid not in currently_tracked_people_status: # Ensure they are truly gone
                        interval_outflows.append({
                            "pid": pid,
                            "age": info.get("age"),
                            "gender": info.get("gender"),
                            "exit_time_sec": info["exit_time_sec"]
                        })

                hourly_inflow_outflow_per_interval.append({
                    "timestamp_sec": interval_end_sec,
                    "inflows": interval_inflows,
                    "outflows": interval_outflows
                })

                last_interval_time_sec = current_time_sec # Update for the next interval

            # Write the processed frame to video
            if vid_writer:
                vid_writer.write(online_img)

        cap.release()
        if vid_writer:
            vid_writer.release()
            
        # At the very end, handle any remaining people in currently_tracked_people_status 
        # (they haven't exited yet when video ends)
        for pid, info in currently_tracked_people_status.items():
            if "exit_time_sec" not in overall_people_info[pid]:
                overall_people_info[pid]["exit_time_sec"] = current_time_sec # Assume they exit at video end
                
        # The `people` global list needs to be handled carefully.
        # If it's meant to be the same as overall_people_info values, copy them.
        # Otherwise, clarify its purpose. For now, it will be the values of overall_people_info.
        # global people # This is already declared globally, so we'll re-populate it.
        # people.clear() # Clear previous runs data if `people` is used for final output for the run
        # for pid, info in overall_people_info.items():
        #     people.append({"id": pid, **info}) # Add track ID to the info dict

        # Return results: overall_people_info is what you'll use for overall plots
        # hourly_age_distribution_per_interval and hourly_inflow_outflow_per_interval
        # are for time-based plots.
        return overall_people_info, hourly_age_distribution_per_interval, hourly_inflow_outflow_per_interval