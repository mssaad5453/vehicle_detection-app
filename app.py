from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}  # Class IDs for vehicles

def detect_vehicles(image_path):
    image = cv2.imread(image_path)
    results = model(image)
    vehicle_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = int(box.cls[0])
            
            if label in vehicle_classes:
                vehicle_type = vehicle_classes[label]
                vehicle_counts[vehicle_type] += 1
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{vehicle_type} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    output_path = "static/output.jpg"
    cv2.imwrite(output_path, image)
    return output_path, vehicle_counts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    image_path = os.path.join("static", file.filename)
    file.save(image_path)
    
    output_image, vehicle_counts = detect_vehicles(image_path)
    return jsonify({'output_image': output_image, 'vehicle_counts': vehicle_counts})

if __name__ == '__main__':
    app.run(debug=True)
