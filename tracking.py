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
from collections import Counter

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
        track_thresh=0.5,
        track_buffer=60,
        match_thresh=0.8,
        aspect_ratio_thresh=1.6,
        min_box_area=3,
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
  

    # 새로운 추론 결과 업데이트 함수
    def update_age_gender_cache(self, tid, age, gender, embed_vec, frame_id):
        global age_gender_cache
        MAX_HISTORY = 10  # 최근 10개 유지
        if frame_id % 5==0 or tid not in age_gender_cache:
            age_gender_cache[tid] = {
                "embedding": embed_vec,
                "ages": [age],
                "genders": [gender]
            }
        else:
            prev = age_gender_cache[tid]
            similarity = np.dot(prev["embedding"], embed_vec)

            if similarity >= 0.9:
                prev["ages"].append(age)
                prev["genders"].append(gender)
                prev["embedding"] = embed_vec
            else:
                # 유사하지 않으면 새로 덮어쓰기
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
        if tid not in age_gender_cache:
            return None, None

        ages = age_gender_cache[tid]["ages"]
        genders = age_gender_cache[tid]["genders"]

        valid_ages = [a for a in ages if a is not None]
        valid_genders = [g for g in genders if g is not None]

        avg_age = int(sum(valid_ages) / len(valid_ages)) if valid_ages else None
        mode_gender = Counter(valid_genders).most_common(1)[0][0] if valid_genders else None

        return avg_age, mode_gender
    
    def process_image(self, img, frame_id=0):
        global people
        global age_gender_cache
        outputs, img_info = self.predictor.inference(img, self.timer)
        results = []
        online_img = img_info["raw_img"]  # 기본값 설정
        if outputs[0] is not None:
            online_targets = self.tracker.update(
                outputs[0], [img_info['height'], img_info['width']], self.exp.test_size
            )
            online_tlwhs, online_ids, online_scores = [], [], []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.tracker.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.tracker.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    bbox = [round(x, 2) for x in tlwh]
                    xyxy = self.tlwh_to_xyxy(bbox)
                    x1, y1, x2, y2 = map(int, xyxy)
                    image_cropped = img[y1:y2, x1:x2]
                    image_resized = cv2.resize(image_cropped, (224, 224))

                    # 얼굴 이미지 저장 (선택)
                    # cropped_path = os.path.join("outputs", "cropped", f"{tid}_crop.jpg")
                    # os.makedirs(os.path.dirname(cropped_path), exist_ok=True)
                    # cv2.imwrite(cropped_path, image_resized)

                    # 나이/성별 추론
                    age_gender_embed = self.predictor.age_gender_recognizer.run(image_resized)
                    if age_gender_embed:
                        age, gender, embed_vec = age_gender_embed[0]
                        embed_vec = embed_vec / np.linalg.norm(embed_vec)

                        self.update_age_gender_cache(tid, age, gender, embed_vec, frame_id)

                        stable_age, stable_gender = self.get_stable_age_gender(tid)

                        people.append({
                            "id": tid,
                            "bbox": bbox,
                            "score": round(t.score, 2),
                            "age": stable_age,
                            "gender": stable_gender
                        })

                        results.append({
                            "id": tid,
                            "bbox": bbox,
                            "score": round(t.score, 2)
                        })

            # 이 부분은 outputs[0]이 None이 아닐 때만 실행됨
            fps_text = 1. / self.timer.average_time if self.timer.average_time > 0 else 0
            online_img = plot_tracking(
                img_info["raw_img"].copy(), online_tlwhs, online_ids,
                frame_id=frame_id,
                fps=fps_text,
                age_gender_info=age_gender_cache
            )

        # 항상 반환하게 수정
        return online_img, results

    def process_video(self, video_path, save_path=None):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []

        if save_path:
            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        for frame_id in tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret:
                break

            online_img, result = self.process_image(frame, frame_id=frame_id + 1)
            results.extend([
                dict(frame=frame_id + 1, **r) for r in result
            ])
            if save_path:
                vid_writer.write(online_img)

        cap.release()
        if save_path:
            vid_writer.release()
        return results, people