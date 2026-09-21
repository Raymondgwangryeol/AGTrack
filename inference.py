from tracking import ObjectTracker
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.tracking_utils.timer import Timer
import json
import os
from ByteTrack.exps.example.mot.yolox_x_ablation import Exp as MyExp
import json
import os
import time

from tracking import ObjectTracker
from ByteTrack.yolox.tracking_utils.timer import Timer
from ByteTrack.exps.example.mot.yolox_x_ablation import Exp as MyExp

if __name__ == "__main__":
    exp = MyExp()
    video_path = "C:..\\test.mp4"
    timer = Timer()

    checkpoint_path = "C:..\\ByteTrack\\pretrained\\yolox_x.pth"

    tracker = ObjectTracker(exp, checkpoint_path)

    out_path = "..\\outputs\\output.mp4"

    start_time = time.time()  # 처리 시간 직접 측정 (포트폴리오 수치 근거용)

    overall_people_info, hourly_age_distribution, hourly_inflow_outflow = tracker.process_video(
        video_path=video_path,
        save_path=out_path,
    )

    elapsed_sec = time.time() - start_time
    print(f"처리 시간: {elapsed_sec:.1f}초 ({elapsed_sec/60:.2f}분)")

    json_path = "..\\outputs\\json\\output.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(overall_people_info, f, indent=4, default=str)

    # 참고: hourly_age_distribution, hourly_inflow_outflow도 필요하면 별도 저장
    # age_json_path = "...\\outputs\\json\\age_distribution.json"
    # with open(age_json_path, "w") as f:
    #     json.dump(hourly_age_distribution, f, indent=4, default=str)
if __name__ == "__main__":
    exp = MyExp()
    video_path = "..\\test.mp4"
    timer = Timer()

    checkpoint_path = "..\\yolox_x.pth"

    tracker = ObjectTracker(exp, checkpoint_path)

    out_path = "..\\output.mp4"
    import time

    start_time = time.time()

    overall_people_info, hourly_age_distribution, hourly_inflow_outflow = tracker.process_video(
        video_path=video_path,
        save_path=out_path,
    )

    elapsed_sec = time.time() - start_time
    print(f"처리 시간: {elapsed_sec:.1f}초 ({elapsed_sec/60:.2f}분)")

    json_path = "..\\json\\output.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(overall_people_info, f, indent=4, default=str)
