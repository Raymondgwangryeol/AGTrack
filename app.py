import gradio as gr
import json
import tempfile
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # For better time axis formatting

# Make sure 'tracking' module (ObjectTracker) is accessible
from tracking import ObjectTracker 
# Make sure your YOLOX Exp is correctly imported
from ByteTrack.exps.example.mot.yolox_x_ablation import Exp as MyExp

from collections import defaultdict
import numpy as np # For potential array operations in plotting


def run_tracking_and_analysis(video_path):
    """
    Executes the object tracking and demographic analysis on a given video.
    Returns:
        tuple: (final_people_info, hourly_age_dist, hourly_inflow_outflow_data, output_video_path)
    """
    exp = MyExp()
    # Ensure this checkpoint path is correct for your system
    checkpoint_path = "C:\\Users\\hcc98\\nayoung\\AGTrack\\AGTrack\\ByteTrack\\pretrained\\yolox_x.pth"
    tracker = ObjectTracker(exp, checkpoint_path)

    # Output video save directory
    save_dir = "C:\\Users\\hcc98\\nayoung\\AGTrack\\outputs"
    os.makedirs(save_dir, exist_ok=True)
    out_video_path = os.path.join(save_dir, "gradio_output.mp4")

    # Call the modified process_video which now returns more data
    overall_people_info, hourly_age_dist, hourly_inflow_outflow_data = tracker.process_video(
        video_path=video_path,
        save_path=out_video_path,
    )
    
    # Note: 'results' from ObjectTracker is no longer directly returned here as it's processed internally
    return overall_people_info, hourly_age_dist, hourly_inflow_outflow_data, out_video_path


def visualize_overall_demographics(people_dict):
    """
    Visualizes the overall age and gender distribution from all tracked people.
    """
    age_counts = {}
    gender_counts = {"male": 0, "female": 0}

    for pid, info in people_dict.items():
        age = info.get("age") # age is now directly int from ObjectTracker
        
        gender_val = info.get("gender")
        if gender_val is not None: # 값이 None이 아닌 경우에만 lower()를 호출합니다.
            gender = str(gender_val).lower() # 혹시 모를 다른 타입도 문자열로 변환 후 lower()
        else:
            gender = "" # None인 경우 빈 문자열로 처리하거나, "unknown" 등으로 처리할 수 있습니다.

        # Age group categorization
        if age is not None and age > 0:
            age_bin = f"{(age // 10) * 10}s"
            age_counts[age_bin] = age_counts.get(age_bin, 0) + 1

        if gender in gender_counts:
            gender_counts[gender] += 1
        # 추가: gender가 male/female이 아닌 다른 값(예: "unknown")일 경우
        # else:
        #     gender_counts["unknown"] = gender_counts.get("unknown", 0) + 1


    # Overall Age Distribution plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    if age_counts:
        # Sort age bins for better visualization (e.g., '10s', '20s' instead of '20s', '10s')
        sorted_age_bins = sorted(age_counts.keys(), key=lambda x: int(x[:-1]))
        ax1.bar([str(b) for b in sorted_age_bins], [age_counts[b] for b in sorted_age_bins], color='skyblue')
    ax1.set_title("Overall Age Distribution")
    ax1.set_xlabel("Age Group")
    ax1.set_ylabel("Count")
    plt.close(fig1) # Close the figure to prevent it from showing automatically

    # Overall Gender Distribution plot
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    if sum(gender_counts.values()) > 0: # Avoid division by zero if no people detected
        ax2.pie(
            gender_counts.values(),
            labels=gender_counts.keys(),
            autopct='%1.1f%%',
            colors=['lightcoral', 'lightblue'],
            startangle=90
        )
        ax2.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title("Overall Gender Distribution")
    plt.close(fig2)

    return fig1, fig2

def visualize_hourly_age_distribution(hourly_age_dist_data):
    """
    Visualizes the age breakdown of customers at different times of the day (every 10 seconds).
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if not hourly_age_dist_data:
        ax.set_title("No Data for Hourly Age Distribution")
        ax.text(0.5, 0.5, "No data available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.close(fig)
        return fig

    time_points = [entry["timestamp_sec"] for entry in hourly_age_dist_data]
    
    # Collect all unique age bins across all intervals
    all_age_bins = set()
    for entry in hourly_age_dist_data:
        all_age_bins.update(entry["age_counts"].keys())
    
    # Sort age bins numerically (e.g., '10s', '20s')
    sorted_age_bins = sorted(list(all_age_bins), key=lambda x: int(x[:-1]))
    
    # Prepare data for stacked bar chart
    age_data_by_bin = defaultdict(list)
    for entry in hourly_age_dist_data:
        for age_bin in sorted_age_bins:
            # Get count for this age_bin in the current interval, default to 0 if not present
            age_data_by_bin[age_bin].append(entry["age_counts"].get(age_bin, 0))
            
    # Create a stacked bar chart
    bottom = np.zeros(len(time_points))
    for age_bin in sorted_age_bins:
        ax.bar(time_points, age_data_by_bin[age_bin], bottom=bottom, label=age_bin)
        bottom += np.array(age_data_by_bin[age_bin]) # Update bottom for stacking
        
    ax.set_title("Hourly Customer Age Distribution (Every 10 Seconds)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Number of Customers")
    ax.legend(title="Age Group", loc='upper left', bbox_to_anchor=(1,1))
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout to prevent labels from overlapping with legend
    fig.tight_layout()
    plt.close(fig)
    return fig

def visualize_inflow_outflow(hourly_inflow_outflow_data):
    """
    Visualizes customer inflow and outflow over time (every 10 seconds).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if not hourly_inflow_outflow_data:
        ax.set_title("No Data for Customer Inflow/Outflow")
        ax.text(0.5, 0.5, "No data available", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.close(fig)
        return fig
    
    time_points = [entry["timestamp_sec"] for entry in hourly_inflow_outflow_data]
    inflow_counts = [len(entry["inflows"]) for entry in hourly_inflow_outflow_data]
    outflow_counts = [len(entry["outflows"]) for entry in hourly_inflow_outflow_data]

    ax.plot(time_points, inflow_counts, label="Inflow (New Customers)", marker='o', linestyle='-', color='green')
    ax.plot(time_points, outflow_counts, label="Outflow (Exited Customers)", marker='x', linestyle='--', color='red')

    ax.set_title("Hourly Customer Inflow/Outflow (Every 10 Seconds)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Number of Customers")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.tight_layout()
    plt.close(fig)
    return fig


def analyze_video(video_path):
    """
    Main function to analyze the uploaded video and return all results for Gradio.
    """
    # Use tempfile to handle video uploads from Gradio securely
    with open(video_path, "rb") as f:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

    # Run tracking and analysis, receiving all new data
    overall_people_info, hourly_age_dist, hourly_inflow_outflow_data, output_video_path = \
        run_tracking_and_analysis(tmp_path)

    # Visualize overall demographics
    fig_overall_age, fig_overall_gender = visualize_overall_demographics(overall_people_info)
    
    # Visualize hourly age distribution
    fig_hourly_age = visualize_hourly_age_distribution(hourly_age_dist)
    
    # Visualize customer inflow/outflow
    fig_inflow_outflow = visualize_inflow_outflow(hourly_inflow_outflow_data)

    # Clean up the temporary file
    os.unlink(tmp_path) 

    return output_video_path, fig_overall_age, fig_overall_gender, fig_hourly_age, fig_inflow_outflow


# Gradio Interface Definition
demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload a video"),
    outputs=[
        gr.Video(label="Tracked Video (with age/gender)"),
        gr.Plot(label="Overall Age Distribution"),
        gr.Plot(label="Overall Gender Distribution"),
        gr.Plot(label="Hourly Age Distribution (Every 10s)"),
        gr.Plot(label="Customer Inflow/Outflow (Every 10s)"),
    ],
    title="AGTrack: Advanced Customer Traffic Analysis",
    description=(
        "Upload a video to get real-time age and gender tracking. "
        "The system provides an annotated video, overall demographic breakdowns, "
        "and detailed time-based insights into age distribution and customer traffic flow."
    ),
    allow_flagging="auto", # Allows users to flag examples for review
)

if __name__ == "__main__":
    output_dir = "C:\\Users\\hcc98\\nayoung\\AGTrack\\outputs"
    # output_dir = os.path.join(os.getcwd(), "outputs") # 또는 이렇게 현재 작업 디렉토리 기준 상대 경로로 설정하는 것이 더 좋습니다.
    
    # 윈도우 경로를 리스트로 넣어줍니다.
    demo.launch(allowed_paths=[output_dir])