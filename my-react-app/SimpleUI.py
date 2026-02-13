# SimpleUI.py
import CONSTANTS
import sys
import cv2
import pyqtgraph as pg
import numpy as np
import os
import xml.etree.ElementTree as ET
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QSplitter
)

##############################################
# CustomButton Definition
##############################################
class CustomButton(QPushButton):
    def __init__(self, func, text):
        super().__init__()
        self.setStyleSheet(CONSTANTS.BUTTON_DEFAULT)
        self.setText(text)
        self.clicked.connect(func)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def setColors(self, backgroundColor, textColor):
        self.setStyleSheet(CONSTANTS.BUTTON_DEFAULT +
                           f"background-color: {backgroundColor}; color: {textColor};")

##############################################
# Utility Functions
##############################################
def parse_annotation(xml_path):
    """
    Parse an XML file (assumed to be in Pascal VOC format) and extract detection data.
    Returns a list of detection dictionaries.
    """
    detections = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text if obj.find('name') is not None else "Unknown"
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = bndbox.find('xmin').text if bndbox.find('xmin') is not None else "0"
                ymin = bndbox.find('ymin').text if bndbox.find('ymin') is not None else "0"
                xmax = bndbox.find('xmax').text if bndbox.find('xmax') is not None else "0"
                ymax = bndbox.find('ymax').text if bndbox.find('ymax') is not None else "0"
                detection = {
                    'name': name,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'confidence': 1.0  # Default confidence
                }
                detections.append(detection)
        print(f"[DEBUG] Parsed detections from {xml_path}: {detections}")
    except Exception as e:
        print(f"Error parsing annotation file {xml_path}: {e}")
    return detections

def annotate_image(image, detections):
    """
    Draw bounding boxes and labels on the image with thin lines and clearly visible labels.
    The labels are drawn on a solid black background to ensure they are readable.
    """
    
    
    h, w = image.shape[:2]

    def get_color(label):
        label_lower = label.lower()
        if label_lower == "5g":
            return (0, 0, 255)  # Red
        elif label_lower == "lte":
            return (0, 255, 0)  # Green
        elif label_lower == "radar":
            return (255, 0, 0)  # Blue
        else:
            return (0, 255, 255)  # Yellow for others

    for det in detections:
        try:
            # Get bounding box coordinates
            x1 = float(det.get("xmin", 0))
            y1 = float(det.get("ymin", 0))
            x2 = float(det.get("xmax", 0))
            y2 = float(det.get("ymax", 0))
            
            # Check if the coordinates are relative (i.e. <= 1) and scale them
            if x2 <= 1 and y2 <= 1:
                x1, y1 = int(x1 * w), int(y1 * h)
                x2, y2 = int(x2 * w), int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            label = det.get("name", "Unknown")
            confidence = float(det.get("confidence", 1.0))
            color = get_color(label)

            # Use a thicker line so boxes are more visible
            thickness = 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Prepare label text
            text = f"{label} {confidence:.2f}"
            font_scale = 0.5
            text_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

            # Draw a black rectangle behind the text for visibility
            cv2.rectangle(image,
                          (text_x, text_y - text_height - baseline),
                          (text_x + text_width, text_y),
                          (0, 0, 0), -1)
            cv2.putText(image, text,
                        (text_x, text_y - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255, 255, 255), text_thickness)
        except Exception as e:
            print("Error in annotation for detection:", det, e)
    return image
##############################################
# MainWindow Class Definition
##############################################
class MainWindow(QMainWindow):
    def __init__(self, restart, stop, pause, resume):
        super().__init__()
        self.restart = restart
        self.stop = stop
        self.pause = pause
        self.resume = resume
        self.paused = False

        self.setWindowTitle("RTX 5G Interference Detector")

        # Create central widget and main layout.
        mainWidget = QWidget()
        self.setCentralWidget(mainWidget)
        self.mainLayout = QVBoxLayout(mainWidget)

        # --- Button Layout ---
        self.buttonLayout = QHBoxLayout()
        self.restartButton = CustomButton(self.restart, "Restart")
        self.restartButton.setColors("#1492ff", "#ffffff")
        self.buttonLayout.addWidget(self.restartButton)

        self.stopButton = CustomButton(self.stop, "Stop")
        self.stopButton.setColors("#ff4336", "#ffffff")
        self.buttonLayout.addWidget(self.stopButton)

        self.pauseButton = CustomButton(self.togglePause, "Pause")
        self.pauseButton.setColors("#919180", "#ffffff")
        self.buttonLayout.addWidget(self.pauseButton)
        self.mainLayout.addLayout(self.buttonLayout)

        # --- Splitter Layout for Spectrogram and Graph ---
        self.dataLayout = QSplitter(Qt.Orientation.Horizontal)
        # Set equal stretch factors.
        self.dataLayout.setStretchFactor(0, 1)
        self.dataLayout.setStretchFactor(1, 1)
        self.dataLayout.setSizes([600, 600])

        # Left panel: Spectrogram image.
        self.imageLabel = QLabel()
        self.imageLabel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imageLabel.setMinimumSize(QSize(600, 600))
        self.dataLayout.addWidget(self.imageLabel)

        # Right panel: Graph.
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.setBackground('w')
        self.plotWidget.setMinimumSize(QSize(600, 600))
        self.dataLayout.addWidget(self.plotWidget)

        self.mainLayout.addWidget(self.dataLayout)

        # --- Status Label ---
        self.updatingLabel = QLabel()
        font = QFont()
        font.setPointSize(18)
        self.updatingLabel.setFont(font)
        self.mainLayout.addWidget(self.updatingLabel)

        # Initially, load a default annotation.
        # (Make sure "loading.xml" exists in your annotation folder or change the filename accordingly.)
        self.updateLabelAndImage("Loading...", "loading.xml", [])

    def togglePause(self):
        """
        Toggle the pause/resume state.
        """
        if self.paused:
            self.pauseButton.setText("Pause")
            self.pauseButton.setColors("#919180", "#ffffff")
            self.resume()  # Call the resume callback.
        else:
            self.pauseButton.setText("Resume")
            self.pauseButton.setColors("#ffffff", "#000000")
            self.pause()  # Call the pause callback.
        self.paused = not self.paused

    def updateLabelAndImage(self, newLabel, newAnnotationFile, detectionData):
  
        if isinstance(newAnnotationFile, str) and newAnnotationFile.lower().endswith('.xml'):
            annotated_folder = "/Users/spoorthikoppula/Desktop/Raytheon/1300 spectrograms"
            xml_path = os.path.join(annotated_folder, newAnnotationFile)
            print(f"🔍 Loading annotation from: {xml_path}")
            if not os.path.exists(xml_path):
                print(f"❌ Annotation file does not exist: {xml_path}")
                return
            detections = parse_annotation(xml_path)
            base_name = os.path.splitext(newAnnotationFile)[0]
            image_filename = base_name + ".jpg"
            original_folder = "/Users/spoorthikoppula/Desktop/Raytheon/images"
            image_path = os.path.join(original_folder, image_filename)
            print(f"🔍 Loading original image from: {image_path}")
            if not os.path.exists(image_path):
                print(f"❌ Original image file does not exist: {image_path}")
                return
            newImage = cv2.imread(image_path)
            if newImage is None:
                print(f"❌ Error: Failed to load image from {image_path}")
                return
            newImage = annotate_image(newImage, detections)
            graph_detections = detections  # Use parsed detections for the graph.
        else:
            image_folder = "/Users/spoorthikoppula/Desktop/Raytheon/images"
            image_path = os.path.join(image_folder, newAnnotationFile)
            print(f"🔍 Loading image from: {image_path}")
            if not os.path.exists(image_path):
                print(f"❌ File does not exist: {image_path}")
                return
            newImage = cv2.imread(image_path)
            if newImage is None:
                print(f"❌ Error: Failed to load image from {image_path}")
                return
        # *** FIX: Annotate the image using the detection data from the model ***
            newImage = annotate_image(newImage, detectionData)
            graph_detections = detectionData

        try:
            height, width, channel = newImage.shape
            bytesPerLine = 3 * width
            qtImage = QImage(newImage.data, width, height, bytesPerLine, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qtImage)
            self.imageLabel.setPixmap(pixmap.scaled(
                self.imageLabel.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            print(f"❌ Error updating spectrogram: {e}")

        self.updatingLabel.setText(newLabel)

        # --- Update the Graph ---
        signal_scores = {}
        for det in graph_detections:
            name = det.get("name", "Unknown")
            confidence = float(det.get("confidence", 0))
            signal_scores[name] = signal_scores.get(name, 0) + confidence

        self.plotWidget.clear()
        if signal_scores:
            names = list(signal_scores.keys())
            values = list(signal_scores.values())
            x = list(range(len(names)))
            self.plotWidget.plot(x, values, pen=pg.mkPen(color='b', width=2), symbol='o')
            ax = self.plotWidget.getAxis('bottom')
            ax.setTicks([list(zip(x, names))])
        else:
            self.plotWidget.plot()
