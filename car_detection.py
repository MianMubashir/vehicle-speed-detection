import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import easyocr
import datetime
import threading
import time
import os
from queue import Queue

def initialize_models():
    # Load YOLOv8 models with specific weights
    vehicle_model = YOLO('yolov8n.pt')
    plate_model = YOLO('yolov8n-seg.pt')  # Using segmentation model for better plate detection
    reader = easyocr.Reader(['en'])
    return vehicle_model, plate_model, reader

def calculate_speed(prev_box, current_box, fps, pixels_per_meter=10):
    # Calculate center points
    prev_center = ((prev_box[0] + prev_box[2])/2, (prev_box[1] + prev_box[3])/2)
    curr_center = ((current_box[0] + current_box[2])/2, (current_box[1] + current_box[3])/2)
    
    # Calculate distance in pixels
    distance_pixels = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                            (curr_center[1] - prev_center[1])**2)
    
    # Convert to meters
    distance_meters = distance_pixels / pixels_per_meter
    
    # Calculate speed (meters per second)
    speed = distance_meters * fps
    
    # Convert to km/h
    speed_kmh = speed * 3.6
    
    return speed_kmh

def process_frame(frame, vehicle_model, plate_model, reader, prev_detections, fps):
    # Resize with maintaining aspect ratio
    height, width = frame.shape[:2]
    scale = 640 / max(height, width)
    new_height, new_width = int(height * scale), int(width * scale)
    frame = cv2.resize(frame, (new_width, new_height))
    
    # Run detection with lower confidence threshold
    results = vehicle_model(frame, conf=0.25)  # Lowered from 0.3
    
    # Initialize vehicle_data list
    vehicle_data_list = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf)
            class_name = result.names[int(box.cls)]
            
            if class_name in ['car', 'bus'] and confidence > 0.25:
                # Calculate speed
                speed = 0
                current_box = [x1, y1, x2, y2]
                
                if prev_detections and len(prev_detections) > 0:
                    try:
                        speed = calculate_speed(prev_detections[0], current_box, fps)
                        speed = min(speed, 130)
                    except Exception as e:
                        print(f"Speed calculation error: {e}")
                
                # Draw smaller detection box
                box_color = (0, 255, 0) if speed < 80 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                
                # Create more compact annotation
                speed_text = f'{speed:.0f}km/h'
                status_text = 'VIO' if speed > 80 else 'OK'
                
                # Calculate text sizes for better positioning
                font_scale = 0.5
                font = cv2.FONT_HERSHEY_SIMPLEX
                speed_size = cv2.getTextSize(speed_text, font, font_scale, 1)[0]
                status_size = cv2.getTextSize(status_text, font, font_scale, 1)[0]
                
                # Add smaller background for text
                padding = 2
                bg_height = speed_size[1] * 2 + padding * 3
                bg_width = max(speed_size[0], status_size[0]) + padding * 2
                
                # Draw compact background and text
                cv2.rectangle(frame, 
                            (x1, y1-bg_height), 
                            (x1+bg_width, y1), 
                            (0, 0, 0), 
                            -1)
                
                text_color = (0, 255, 0) if speed < 80 else (0, 0, 255)
                cv2.putText(frame, 
                          speed_text, 
                          (x1+padding, y1-bg_height+speed_size[1]+padding),
                          font, 
                          font_scale, 
                          text_color, 
                          1)
                cv2.putText(frame, 
                          status_text, 
                          (x1+padding, y1-padding),
                          font, 
                          font_scale, 
                          text_color, 
                          1)
                
                # Store detection data
                current_time = datetime.datetime.now()
                detection_data = {
                    'detection_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'vehicle_type': class_name,
                    'speed': round(speed, 2),
                    'speed_violation': speed > 80,
                    'confidence': round(confidence, 3),
                    'box': current_box,
                    'vehicle_id': f'Vehicle_{len(vehicle_data_list)}'
                }
                
                # Save to vehicles CSV file
                df = pd.DataFrame([{
                    'Time': detection_data['detection_time'],
                    'Vehicle_Type': class_name,
                    'Vehicle_ID': detection_data['vehicle_id'],
                    'Speed_kmh': round(speed, 1),
                    'Status': 'VIOLATION' if speed > 80 else 'OK',
                    'Confidence': round(confidence, 3)
                }])
                
                vehicles_csv = 'vehicles_data.csv'
                if not os.path.exists(vehicles_csv):
                    df.to_csv(vehicles_csv, index=False)
                else:
                    df.to_csv(vehicles_csv, mode='a', header=False, index=False)
                
                # Add to vehicle data list for analytics
                vehicle_data_list.append(detection_data)
    
    return frame, vehicle_data_list

class CarAnalyzer:
    def __init__(self):
        self.data_queue = Queue()
        self.analysis_results = {
            'total_cars': 0,
            'cars_count': 0,
            'buses_count': 0,
            'average_speed': 0,
            'max_speed': 0,
            'violations_count': 0,
            'compliant_count': 0
        }
    
    def update_analytics(self, vehicle_data):
        if vehicle_data:
            for data in vehicle_data:
                # Update vehicle counts
                self.analysis_results['total_cars'] += 1
                if data['vehicle_type'] == 'car':
                    self.analysis_results['cars_count'] += 1
                else:
                    self.analysis_results['buses_count'] += 1
                
                # Update speed statistics
                speed = data['speed']
                self.analysis_results['max_speed'] = max(self.analysis_results['max_speed'], speed)
                self.analysis_results['average_speed'] = (
                    (self.analysis_results['average_speed'] * (self.analysis_results['total_cars'] - 1) +
                     speed) / self.analysis_results['total_cars']
                )
                
                # Update violation counts
                if speed > 80:
                    self.analysis_results['violations_count'] += 1
                else:
                    self.analysis_results['compliant_count'] += 1
    
    def generate_report(self):
        report_data = {
            'Metric': [
                'Total Vehicles',
                'Cars Count',
                'Buses Count',
                'Average Speed (km/h)',
                'Maximum Speed (km/h)',
                'Speed Violations (>80 km/h)',
                'Compliant Vehicles'
            ],
            'Value': [
                self.analysis_results['total_cars'],
                self.analysis_results['cars_count'],
                self.analysis_results['buses_count'],
                round(self.analysis_results['average_speed'], 2),
                round(self.analysis_results['max_speed'], 2),
                self.analysis_results['violations_count'],
                self.analysis_results['compliant_count']
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv('analysis_report.csv', index=False)
        return report_df

def main():
    # Initialize models and analyzer
    vehicle_model, plate_model, reader = initialize_models()
    analyzer = CarAnalyzer()
    
    # Video capture
    video_path = '/Users/ali/Documents/Practics/Car_Model/Test Video.mp4'
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_detections = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame and get vehicle data
            processed_frame, vehicle_data = process_frame(frame, vehicle_model, plate_model, 
                                                        reader, prev_detections, fps)
            
            # Update previous detections for next frame
            if vehicle_data:
                prev_detections = [data['box'] for data in vehicle_data]
            else:
                prev_detections = []  # Reset if no detections
            
            # Display frame (remove duplicate display)
            cv2.imshow('Vehicle Detection', processed_frame)
            
            # Update analytics
            analyzer.update_analytics(vehicle_data)
            
            # Generate and save reports periodically
            if time.time() % 5 < 0.1:  # Every 5 seconds approximately
                report = analyzer.generate_report()
                print("\nCurrent Analysis Report:")
                print(report)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Error in main loop: 'box'")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


def update_summary_statistics():
    if os.path.exists('vehicles_data.csv'):
        df = pd.read_csv('vehicles_data.csv')
        
        summary_data = {
            'Metric': [
                'Total Vehicles',
                'Cars Count',
                'Buses Count',
                'Average Speed (km/h)',
                'Maximum Speed (km/h)',
                'Speed Violations (>80 km/h)',
                'Compliant Vehicles'
            ],
            'Value': [
                len(df),
                len(df[df['Vehicle_Type'] == 'car']),
                len(df[df['Vehicle_Type'] == 'bus']),
                round(df['Speed_kmh'].mean(), 1),
                round(df['Speed_kmh'].max(), 1),
                len(df[df['Status'] == 'VIOLATION']),
                len(df[df['Status'] == 'OK'])
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('analysis_report.csv', index=False)