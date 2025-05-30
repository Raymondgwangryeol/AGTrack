# 🔎AGTrack
Personal Project | Duration: 2025.05.13 ~ 2025.05.30

## Introduction
This project, AGTrack, focuses on real-time object tracking and demographic inference. It leverages ByteTrack for robust multi-object tracking and MiVOLO for accurate age and gender estimation, specifically applied to tracking individuals.

The core objective of AGTrack is to provide a service that allows users to manage and analyze customer information, such as hourly customer age distribution and inflow patterns.

## Dataset
For this project, we utilized the Multi-sensor Movement Tracking Data (멀티 센서 동선 추적 데이터) available from AI Hub: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&amp;topMenu=100&amp;dataSetSn=620.

To facilitate rapid experimentation and development, only the test data portion of this dataset was used.

## Features
AGTrack offers several key functionalities:

- Person Tracking: Accurately tracks multiple individuals simultaneously using ByteTrack.
- Age and Gender Inference: Estimates the age and gender of tracked individuals in real time with MiVOLO.
- Customer Traffic Analysis: Provides insights into customer movement paths and patterns.
- Demographic Insights: Enables analysis of customer demographics, such as:
  - Hourly Customer Age Distribution: Understand the age breakdown of customers at different times of the day.
  - Hourly Customer Inflow Information: Track when specific demographics of customers enter and exit a monitored area.
- Potential for Business Intelligence: Offers a foundation for businesses to better understand their customer base and optimize operations.

## Demo Video

## Project Structure
```
AGTrack/
├── MiVOLO/                # Pre-trained MiVOLO model
├── ByteTrack/             # Pre-trained ByteTrack model
├── data/                  # Multi-sensor Movement Tracking Data
├── AGPredictor.py         # Predict object's age and gender
├── app.py                 # Demo Web
├── tracking.py            # Tracking objects
├── inference.py           # For inference
├── README.md              # Project README
└── (other configuration files, e.g., sweep.yaml)
```
