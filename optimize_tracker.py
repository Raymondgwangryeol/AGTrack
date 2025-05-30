import wandb
import sys
import time
from tracking import ObjectTracker
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.tracking_utils.timer import Timer
import json
import os
from ByteTrack.exps.example.mot.yolox_x_ablation import Exp as MyExp
from collections import Counter
import yaml

def compute_custom_metric(people):
    track_durations = Counter([p["id"] for p in people])
    long_tracks = [tid for tid, count in track_durations.items() if count >= 15]
    return len(long_tracks) / len(track_durations) if track_durations else 0

def sweep_iteration():
    total_start = time.time()
    wandb.init()
    args = wandb.config

    exp = MyExp()
    video_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\data\\VS1\\Scenario19\\test.mp4"
    checkpoint_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\AGTrack\\ByteTrack\\pretrained\\yolox_x.pth"
    out_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\outputs\\output.mp4"

    print("🚀 Tracking 시작")
    start = time.time()
    tracker = ObjectTracker(
        exp_file=exp,
        ckpt=checkpoint_path,
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        aspect_ratio_thresh=args.aspect_ratio_thresh,
        min_box_area=args.min_box_area,
    )
    results, people = tracker.process_video(
        video_path=video_path,
        save_path=out_path,
    )
    track_time = time.time() - start
    print(f"✅ Tracking 완료 (소요 시간: {track_time:.2f}초)")

    print("🧮 Metric 계산 중...")
    start = time.time()
    score = compute_custom_metric(people)
    metric_time = time.time() - start
    print(f"✅ Metric 계산 완료 (소요 시간: {metric_time:.2f}초)")

    print("💾 JSON 저장 중...")
    start = time.time()
    json_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\outputs\\json\\output.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(people, f, indent=4, default=str)
    json_time = time.time() - start
    print(f"✅ JSON 저장 완료 (소요 시간: {json_time:.2f}초)")

    total_time = time.time() - total_start
    print(f"🕒 전체 실행 시간: {total_time:.2f}초")

    # 🪵 W&B에 모든 시간 기록
    wandb.log({
        "score": score,
        "time/tracking": track_time,
        "time/metric": metric_time,
        "time/json_save": json_time,
        "time/total": total_time,
    })

    if score > 0.90:
        print("🎯 성능 기준 달성, 자동 종료.")
        wandb.finish(exit_code=0)
        sys.exit(0)

if __name__ == "__main__":
    with open("C:/Users/hcc98/nayoung/AGTrack/AGTrack/sweep.yaml", encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="optimize_tracker")
    wandb.agent(sweep_id, function=sweep_iteration)
