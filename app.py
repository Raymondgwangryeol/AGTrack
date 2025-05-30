import gradio as gr
import json
import tempfile
import os
import matplotlib.pyplot as plt

from tracking import ObjectTracker
from ByteTrack.exps.example.mot.yolox_x_ablation import Exp as MyExp


def run_tracking(video_path):
    exp = MyExp()
    checkpoint_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\AGTrack\\ByteTrack\\pretrained\\yolox_x.pth"
    tracker = ObjectTracker(exp, checkpoint_path)

    # 출력 영상 저장 경로
    save_dir = "C:\\Users\\hcc98\\nayoung\\AGTrack\\outputs"
    os.makedirs(save_dir, exist_ok=True)
    out_video_path = os.path.join(save_dir, "gradio_output.mp4")

    results, people = tracker.process_video(
        video_path=video_path,
        save_path=out_video_path,
    )

    return people, out_video_path


def visualize_people(people_dict):
    age_counts = {}
    gender_counts = {"male": 0, "female": 0}

    for pid, info in people_dict.items():
        age = int(info.get("age", -1))
        gender = info.get("gender", "").lower()

        # 나이대 분류
        if age > 0:
            age_bin = f"{(age // 10) * 10}s"
            age_counts[age_bin] = age_counts.get(age_bin, 0) + 1

        if gender in gender_counts:
            gender_counts[gender] += 1

    # 연령 분포 시각화
    fig1, ax1 = plt.subplots()
    ax1.bar(age_counts.keys(), age_counts.values(), color='skyblue')
    ax1.set_title("Age Distribution")
    ax1.set_xlabel("Age Group")
    ax1.set_ylabel("Count")

    # 성별 비율 시각화
    fig2, ax2 = plt.subplots()
    ax2.pie(
        gender_counts.values(),
        labels=gender_counts.keys(),
        autopct='%1.1f%%',
        colors=['lightcoral', 'lightblue']
    )
    ax2.set_title("Gender Distribution")

    return fig1, fig2


def analyze_video(video_path):
    with open(video_path, "rb") as f:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(f.read())  # 올바르게 bytes를 씀
            tmp_path = tmp.name

    # 추적 + 분석
    people, output_video_path = run_tracking(tmp_path)

    # 시각화
    fig_age, fig_gender = visualize_people(people)

    return output_video_path, fig_age, fig_gender


# Gradio 인터페이스 정의
demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload a video"),
    outputs=[
        gr.Video(label="Tracked Video (with age/gender)"),
        gr.Plot(label="Age Distribution"),
        gr.Plot(label="Gender Distribution"),
    ],
    title="Age/Gender Tracking and Analysis",
    description="Upload a video to detect and analyze people based on age and gender.",
)

if __name__ == "__main__":
    demo.launch()
