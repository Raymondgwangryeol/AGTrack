
import logging
import os

import cv2
import torch
import yt_dlp
from MiVOLO.mivolo.data.data_reader import InputType, get_all_files, get_input_type
from MiVOLO.mivolo.predictor import Predictor
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

    def run(self, image: np.ndarray):
        detected_objects, _ = self.predictor.recognize(image)
        results = []
        for age, gender in zip(detected_objects.ages, detected_objects.genders):
            embedding = image.flatten()
            results.append((age, gender, embedding))

        return results
