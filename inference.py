from tracking import ObjectTracker
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.tracking_utils.timer import Timer
import json
import os
from ByteTrack.exps.example.mot.yolox_x_ablation import Exp as MyExp

if __name__ == "__main__":
    exp = MyExp()
    video_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\data\\VS1\\Scenario19\\test.mp4"
    timer = Timer()

    checkpoint_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\AGTrack\\ByteTrack\\pretrained\\yolox_x.pth"

    tracker = ObjectTracker(exp, checkpoint_path)

    out_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\outputs\\output.mp4"
    results, people = tracker.process_video(
        video_path=video_path,
        save_path=out_path,
    )
    json_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\outputs\\json\\output.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(people, f, indent=4, default=str)