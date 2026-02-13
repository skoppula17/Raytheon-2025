import pathlib
pathlib.WindowsPath = pathlib.PosixPath

import torch
import os
import io
import sys
from pathlib import Path
from PIL import Image
import CONSTANTS

# Add YOLOv5 directory to system path
YOLOV5_DIR = str(Path(__file__).resolve().parent / "yolov5")
sys.path.append(YOLOV5_DIR)  # ✅ Fixes "No module named 'yolov5'"

from yolov5.models.common import DetectMultiBackend

# ✅ Use the fixed model path
FIXED_MODEL_PATH = "/Users/spoorthikoppula/Desktop/Raytheon/Model/best_fixed.pt"

# Load YOLOv5 model correctly
class modelAPI:
    def __init__(self, modelPath=str(Path(__file__).resolve().parent / "Model" / "best.pt")):
        print(f"Loading YOLOv5 model from {modelPath}...")
        self.model = torch.hub.load(YOLOV5_DIR, 'custom', path=modelPath, source='local')
        self.model.eval()
        # Save the names mapping (if available)
        self.names = self.model.names if hasattr(self.model, 'names') else {}
        print("✅ YOLOv5 Model Loaded Successfully!")

    def classify(self, filePath=None):
        if not filePath:
            return (CONSTANTS.FAILURE, "No file path given")
        try:
            img = Image.open(filePath)
            results = self.model(img)
            # Get the detections as a list of dictionaries.
            detections = results.pandas().xyxy[0].to_dict(orient="records")
            # Update each detection with the correct name if missing or wrong.
            for det in detections:
                # If the 'name' key is missing or not set correctly, try to use the class index.
                if 'name' not in det or not det['name']:
                    # Use the model’s names mapping.
                    class_idx = int(det.get('class', -1))
                    det['name'] = self.names.get(class_idx, "Unknown")
            return detections
        except Exception as e:
            return (CONSTANTS.FAILURE, f"Error processing image: {e}")
