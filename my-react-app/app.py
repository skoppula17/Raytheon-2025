import os
import time
import cv2
import base64
import warnings
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
from ultralytics import YOLO
from PIL import Image
import torch
import sys
import pathlib
import numpy as np
import xml.etree.ElementTree as ET

# Suppress FutureWarnings from torch
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure compatibility on macOS/Linux
pathlib.WindowsPath = pathlib.PosixPath  

# Determine the base directory (where app.py resides)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set paths
YOLOV5_PATH = os.path.join(BASE_DIR, "yolov5")
MODEL_PATH = os.path.join(BASE_DIR, "backend", "model", "best.pt")
IMAGES_FOLDER = os.path.join(BASE_DIR, "images")
ANNOTATIONS_FOLDER = "/Users/spoorthikoppula/Desktop/Raytheon/1300 spectrograms"

# Folder for high interference spectrograms
HIGH_INTERFERENCE_FOLDER = os.path.join(BASE_DIR, "high_interference")
os.makedirs(HIGH_INTERFERENCE_FOLDER, exist_ok=True)

sys.path.append(YOLOV5_PATH)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:3000"])

# Load YOLO model
yolo_model = torch.hub.load(YOLOV5_PATH, 'custom', path=MODEL_PATH, source='local', force_reload=True)
print("✅ YOLOv5 Model Loaded Successfully!")

# Global variables for streaming and graphing
STREAM_RUNNING = False
frame_count = 0
global_history = []  # We'll keep only the last 10 history points
bg_thread = None

def parse_annotation(xml_path):
    """Parse XML annotation (Pascal VOC format) and return a list of detections."""
    detections = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text if obj.find("name") is not None else "Unknown"
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                detections.append({
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "name": name,
                    "confidence": 1.0
                })
        print(f"[DEBUG] Parsed {len(detections)} detections from {xml_path}")
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
    return detections

def annotate_image(image, detections):
    """Draw bounding boxes and labels on the image."""
    h, w = image.shape[:2]
    def get_color(label):
        colors = {
            "5g": (0, 0, 255),      # Neon Red
            "lte": (255, 0, 255),    # Neon Magenta
            "radar": (0, 255, 0),    # Neon Green
            "jsss": (255, 165, 0)    # Neon Orange
        }
        return colors.get(label.lower(), (255, 255, 255))
    for det in detections:
        try:
            x1, y1, x2, y2 = map(int, [det["xmin"], det["ymin"], det["xmax"], det["ymax"]])
            label = det["name"]
            confidence = float(det["confidence"])
            box_color = get_color(label)
            text = f"{label} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)
            font_scale = 0.8
            text_thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5),
                          (text_x + text_size[0], text_y), (0, 0, 0), -1)
            cv2.putText(image, text, (text_x, text_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness)
        except Exception as e:
            print(f"Error annotating detection {det}: {e}")
    return image

def compute_graph_data(detections, img_width, img_height):
    """Compute the ratio of each bounding box's area to the image area for each signal type."""
    ratios = {"5G": 0, "LTE": 0, "Radar": 0, "JSSS": 0, "All": 0}
    img_area = img_width * img_height
    for det in detections:
        try:
            x1, y1, x2, y2 = map(float, [det["xmin"], det["ymin"], det["xmax"], det["ymax"]])
            bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
            ratio = bbox_area / img_area if img_area > 0 else 0
            name = det["name"].lower()
            if name == "5g":
                ratios["5G"] += ratio
            elif name == "lte":
                ratios["LTE"] += ratio
            elif name == "radar":
                ratios["Radar"] += ratio
            elif name == "jsss":
                ratios["JSSS"] += ratio
            ratios["All"] += ratio
        except Exception as e:
            print(f"Error computing ratio for detection {det}: {e}")
    return ratios

def process_images():
    """Cycle through images every 2 seconds, compute annotations, graph data, and emit updates."""
    global frame_count, global_history, STREAM_RUNNING
    images = sorted(os.listdir(IMAGES_FOLDER))
    if not images:
        print("[DEBUG] No images found in the images folder.")
        return
    idx = 0
    while STREAM_RUNNING:
        filename = images[idx % len(images)]
        filepath = os.path.join(IMAGES_FOLDER, filename)
        if not os.path.isfile(filepath):
            idx += 1
            continue
        print(f"[DEBUG] Processing image: {filename}")
        try:
            img = Image.open(filepath).convert("RGB")
        except Exception as e:
            print(f"Error opening image {filename}: {e}")
            idx += 1
            continue
        base_name, _ = os.path.splitext(filename)
        xml_path = os.path.join(ANNOTATIONS_FOLDER, base_name + ".xml")
        if os.path.exists(xml_path):
            print(f"[DEBUG] Found XML annotation for {filename}")
            detections = parse_annotation(xml_path)
        else:
            print(f"[DEBUG] No XML for {filename}; using YOLO detection.")
            results = yolo_model(img)
            try:
                detections = results.pandas().xyxy[0].to_dict(orient="records")
            except Exception as e:
                print(f"Error extracting detections from YOLO for {filename}: {e}")
                detections = []
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        annotated_img = annotate_image(img_cv, detections)
        h, w = annotated_img.shape[:2]
        ratios = compute_graph_data(detections, w, h)
        frame_count += 1
        history_point = {
            "time": frame_count,
            "5G": ratios["5G"],
            "LTE": ratios["LTE"],
            "Radar": ratios["Radar"],
            "JSSS": ratios["JSSS"],
            "All": ratios["All"]
        }
        global_history.append(history_point)
        # Keep only the last 10 history points
        if len(global_history) > 10:
            global_history = global_history[-10:]
            
        # Save spectrogram if interference (All ratio) is >= 50%
        noisePercent = ratios["All"] * 100
        if noisePercent >= 50:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            high_intf_filename = f"high_interference_{frame_count}_{timestamp}.jpg"
            high_intf_filepath = os.path.join(HIGH_INTERFERENCE_FOLDER, high_intf_filename)
            cv2.imwrite(high_intf_filepath, annotated_img)
            print(f"[DEBUG] Saved high interference image: {high_intf_filepath}")
            
        debug_path = os.path.join(BASE_DIR, "debug_annotated.jpg")
        cv2.imwrite(debug_path, annotated_img)
        print(f"[DEBUG] Saved debug image: {debug_path}")
        _, buffer = cv2.imencode(".jpg", annotated_img)
        encoded_img = base64.b64encode(buffer).decode("utf-8")
        socketio.emit("new_detection", {
            "image": encoded_img,
            "detections": detections,
            "graphData": global_history,
            "time": frame_count
        })
        idx += 1
        time.sleep(2)  # Update every 2 seconds

# --- Control Endpoints ---
@app.route("/start", methods=["POST"])
def start_stream():
    global STREAM_RUNNING, bg_thread
    if not STREAM_RUNNING:
        STREAM_RUNNING = True
        print("[DEBUG] Start command received")
        bg_thread = socketio.start_background_task(target=process_images)
    return jsonify({"message": "Started processing images"}), 200

@app.route("/stop", methods=["POST"])
def stop_stream():
    global STREAM_RUNNING
    STREAM_RUNNING = False
    print("[DEBUG] Stop command received")
    return jsonify({"message": "Stopped processing images"}), 200

@app.route("/reset", methods=["POST"])
def reset_stream():
    global STREAM_RUNNING, frame_count, global_history
    STREAM_RUNNING = False
    frame_count = 0
    global_history = []
    print("[DEBUG] Reset command received")
    STREAM_RUNNING = True
    socketio.start_background_task(target=process_images)
    return jsonify({"message": "Reset and restarted processing images"}), 200

@socketio.on("connect")
def handle_connect():
    print("[DEBUG] Client connected.")

if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
