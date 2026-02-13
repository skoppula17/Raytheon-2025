import os
import time
import cv2
import base64
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from ultralytics import YOLO
from PIL import Image
import torch
import os
import sys
import pathlib
pathlib.WindowsPath = pathlib.PosixPath  # ✅ Forces compatibility on macOS/Linux

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Define the correct YOLOv5 path
YOLOV5_PATH = os.path.join(os.path.dirname(__file__), "yolov5")

# Ensure YOLOv5 is added to the system path
sys.path.append(YOLOV5_PATH)

# Correct model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model/best.pt")

# Load YOLOv5 model correctly
yolo_model = torch.hub.load(YOLOV5_PATH, 'custom', path=MODEL_PATH, source='local', force_reload=True)

print("✅ YOLOv5 Model Loaded Successfully!")


# Folder containing spectrogram images
IMAGES_FOLDER = "images"

# Global variables for streaming and graphing
STREAM_RUNNING = False
global_history = []  # List of dicts: {"time": frame_number, "5G": count, "LTE": count, "LSS": count, "All": count}
global_counts = {"5G": 0, "LTE": 0, "LSS": 0, "All": 0}
frame_count = 0
processed_files = set()

def encode_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")

def process_images():
    """Background task that monitors the images folder, runs detection, updates graph data, and emits events."""
    global frame_count, global_counts, global_history, processed_files
    while True:
        if STREAM_RUNNING:
            try:
                image_files = sorted(os.listdir(IMAGES_FOLDER))
            except Exception as e:
                print("Error reading images folder:", e)
                time.sleep(2)
                continue
            
            for filename in image_files:
                if filename in processed_files:
                    continue
                filepath = os.path.join(IMAGES_FOLDER, filename)
                if not os.path.isfile(filepath):
                    continue

                print(f"Processing {filename}...")
                frame_count += 1
                try:
                    img = Image.open(filepath)
                except Exception as e:
                    print(f"Error opening image {filename}: {e}")
                    processed_files.add(filename)
                    continue
                
                # Run YOLO detection
                try:
                    results = yolo_model(img)
                    detections = results[0].pandas().xyxy[0].to_dict(orient="records")
                except Exception as e:
                    print(f"Error running YOLO on image {filename}: {e}")
                    detections = []
                
                # Update counts based on detections
                for det in detections:
                    name = det.get("name", "").lower()
                    if name == "5g":
                        global_counts["5G"] += 1
                    elif name == "lte":
                        global_counts["LTE"] += 1
                    else:
                        global_counts["LSS"] += 1
                global_counts["All"] = global_counts["5G"] + global_counts["LTE"] + global_counts["LSS"]
                
                # Append a new point to the history (using frame_count as a time index)
                history_point = {
                    "time": frame_count,
                    "5G": global_counts["5G"],
                    "LTE": global_counts["LTE"],
                    "LSS": global_counts["LSS"],
                    "All": global_counts["All"]
                }
                global_history.append(history_point)
                
                # Set warning flag if more than 5 detections appear in this image
                warning_flag = len(detections) > 5
                
                # Encode the image to Base64 for sending to the client
                encoded_img = encode_image(filepath)
                
                # Emit the new detection event with image, detections, warning flag, and graph data
                socketio.emit("new_detection", {
                    "image": encoded_img,
                    "detections": detections,
                    "warning": warning_flag,
                    "graphData": global_history
                })
                
                processed_files.add(filename)
                time.sleep(1)  # Small delay between images
        else:
            time.sleep(1)
        time.sleep(1)

# --- Control Endpoints ---

@app.route("/start", methods=["POST"])
def start_stream():
    global STREAM_RUNNING
    STREAM_RUNNING = True
    return jsonify({"message": "Started processing"}), 200

@app.route("/stop", methods=["POST"])
def stop_stream():
    global STREAM_RUNNING
    STREAM_RUNNING = False
    return jsonify({"message": "Stopped processing"}), 200

@app.route("/reset", methods=["POST"])
def reset_stream():
    global STREAM_RUNNING, global_history, global_counts, frame_count, processed_files
    STREAM_RUNNING = False
    global_history = []
    global_counts = {"5G": 0, "LTE": 0, "LSS": 0, "All": 0}
    frame_count = 0
    processed_files = set()
    STREAM_RUNNING = True
    return jsonify({"message": "Reset successful"}), 200

@socketio.on("connect")
def handle_connect():
    print("Client connected – starting image processing background task.")
    socketio.start_background_task(target=process_images)

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)

