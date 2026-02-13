from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import os
import io
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model_path = os.path.join(os.path.dirname(__file__), '../models/best.pt')
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

@app.route("/")
def hello_world():
    return {
        "response": "ok"
    }

@app.route("/detect", methods=['POST'])
def segment():
    """
    This endpoint is called in the frontend.
    It runs the YOLO model on an image file and returns the bounding boxes.
    """
    file = None

    if 'file_path' in request.json:
        file_path = request.json['file_path']
        file = open(file_path, 'rb')

    if not file:
        return jsonify({"message": "No file part"}), 400

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # preprocess
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0)

    results = model(img)

    results_json = results.pandas().xyxy[0].to_json(orient="records")

    response = jsonify({"results": json.loads(results_json)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run()
    