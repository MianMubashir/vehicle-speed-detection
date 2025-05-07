# Vehicle Detection and Speed Monitoring System

A computer vision system that detects vehicles and monitors their speed using YOLOv8.

## Features
- Real-time vehicle detection (cars and buses)
- Speed calculation and monitoring
- Speed violation detection (threshold: 80 km/h)
- Visual indicators for speed violations
- Data logging and analytics
- CSV reports generation

## Requirements
- Python 3.x
- OpenCV
- YOLOv8
- EasyOCR
- Pandas
- NumPy

## Installation
1. Clone the repository
2. Install dependencies:
```pip3 install ultralytics opencv-python easyocr pandas numpy```

## Usage
Run the main script:
```python3 car_detection.py```

## Output
- Real-time video feed with vehicle detection
- Speed monitoring with visual indicators
- Individual vehicle data in vehicles_data.csv
- Summary statistics in analysis_report.csv
