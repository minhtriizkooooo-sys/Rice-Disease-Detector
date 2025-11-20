from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import cv2
import numpy as np
import onnxruntime as ort
import base64
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
import torch

# DÒNG DUY NHẤT ĐƯỢC THÊM – TỰ ĐỘNG TẢI MODEL TỪ GOOGLE DRIVE
from download_model import *   # ← Đây là dòng quan trọng nhất! Chạy ngay khi app khởi động

# ==================== CẤU HÌNH APP ====================
app = FastAPI()

# Static files (logo, css, js...)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Import tên bệnh + màu
from disease_names import DISEASE_NAMES, COLORS

# ==================== HÀM TIỀN XỬ LÝ ====================
def preprocess(img):
    img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==================== CÁC ROUTE ====================

# Trang đăng nhập (tùy chọn)
@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Trang chủ
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Nếu muốn bắt buộc login → bỏ comment dòng dưới
    # return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {"request": request})

# API dự đoán
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Không thể đọc ảnh. Vui lòng thử lại!"}

    # Inference
    input_data = preprocess(img)
    outputs = session.run(None, {input_name: input_data})

    # Post-process YOLOv8 ONNX
    pred = torch.from_numpy(outputs[0]).sigmoid()
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45, max_det=100)[0]

    # Scale boxes về ảnh gốc
    if len(pred) > 0:
        pred[:, :4] = scale_boxes((640, 640), pred[:, :4], img.shape[:2]).round()

    # Vẽ kết quả
    result_img = img.copy()
    results = []

    for det in pred:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)
        label = DISEASE_NAMES.get(cls, "Không xác định")
        color_hex = COLORS.get(cls, "#ffffff")
        color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # RGB → BGR

        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(result_img, f"{label} {conf:.2f}",
                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        results.append({"disease": label, "confidence": round(conf, 3)})

    # Encode ảnh
    _, encoded = cv2.imencode('.jpg', result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    image_base64 = base64.b64encode(encoded).decode()

    return {"image": image_base64, "diseases": results}

# ==================== CHẠY SERVER ====================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

