import onnxruntime
import numpy as np
from PIL import Image

MODEL_PATH = "best_quantized.onnx"

# Load ONNX model
yolo_session = onnxruntime.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

def preprocess(img: Image.Image):
    img = img.resize((640, 640))
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return arr

def run_inference(img: Image.Image):
    inp = preprocess(img)
    ort_inputs = {yolo_session.get_inputs()[0].name: inp}

    outputs = yolo_session.run(None, ort_inputs)[0]

    # YOLOv8 ONNX output format: [N, 84]
    # [x1, y1, x2, y2, conf, class_prob...]

    boxes = outputs[:, :4]
    scores = outputs[:, 4]
    classes = np.argmax(outputs[:, 5:], axis=1)

    # filter low confidence
    mask = scores > 0.3
    return boxes[mask], scores[mask], classes[mask]
