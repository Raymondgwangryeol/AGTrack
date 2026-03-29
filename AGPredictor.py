
import logging
import os

import cv2
import torch
import yt_dlp
from MiVOLO.mivolo.data.data_reader import InputType, get_all_files, get_input_type
from MiVOLO.mivolo.predictor import Predictor
from MiVOLO.mivolo.data.misc import prepare_classification_images
from timm.utils import setup_default_logging
import argparse
import numpy as np


_logger = logging.getLogger("inference")


class VideoRecognizer:
    def __init__(
        self,
        detector_weights: str = "C:\\Users\\hcc98\\nayoung\\AGTrack\\AGTrack\\MiVOLO\\mivolo\\model\\yolov8x_person_face.pt",
        checkpoint_path: str = "C:\\Users\\hcc98\\nayoung\\AGTrack\\AGTrack\\MiVOLO\\mivolo\\model\\model_imdb_cross_person_4.22_99.46.pth",
        device: str = "cuda",
        with_persons: bool = True,
        disable_faces: bool = True,
        draw: bool = False,
    ):
        self.detector_weights = detector_weights
        self.checkpoint = checkpoint_path
        self.device = device
        self.with_persons = with_persons
        self.disable_faces = disable_faces
        self.draw = draw

        setup_default_logging()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True


        # Predictor에 맞는 args dict 만들어 전달
        class Args:
            pass
        args = Args()
        args.input = None
        args.output = None
        args.detector_weights = self.detector_weights
        args.checkpoint = self.checkpoint
        args.device = self.device
        args.with_persons = self.with_persons
        args.disable_faces = self.disable_faces
        args.draw = self.draw

        self.args = args
        self.predictor = Predictor(self.args, verbose=True)

    def run_batch(self, person_crops: list) -> list:
        if not person_crops:
            return []
    
        mivolo = self.predictor.age_gender_model
    
        # None 리스트를 face 자리에 넘기면 내부에서 zeros 처리
        none_list = [None] * len(person_crops)
    
        person_input = prepare_classification_images(
            person_crops,          # BGR np.ndarray 그대로 OK
            mivolo.input_size,
            mivolo.data_config["mean"],
            mivolo.data_config["std"],
            device=mivolo.device,
        )  # [N, 3, H, W]
    
        face_input = prepare_classification_images(
            none_list,             # 전부 zeros로 채워짐
            mivolo.input_size,
            mivolo.data_config["mean"],
            mivolo.data_config["std"],
            device=mivolo.device,
        )  # [N, 3, H, W] zeros
    
        model_input = torch.cat((face_input, person_input), dim=1)  # [N, 6, H, W]
    
        output = mivolo.inference(model_input)  # [N, 3]
    
        # 후처리
        age_output = output[:, 2]
        gender_output = output[:, :2].softmax(-1)
        gender_probs, gender_indx = gender_output.topk(1)
    
        results = []
        for i in range(len(person_crops)):
            age = age_output[i].item()
            age = age * (mivolo.meta.max_age - mivolo.meta.min_age) + mivolo.meta.avg_age
            age = round(age, 2)
            gender = "male" if gender_indx[i].item() == 0 else "female"
    
            embed_vec = person_input[i].cpu().float().numpy().flatten()
            embed_vec = embed_vec / (np.linalg.norm(embed_vec) + 1e-8)
    
            results.append((age, gender, embed_vec))
    
        return results
