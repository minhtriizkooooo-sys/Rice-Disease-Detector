import os
import sys
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime
import numpy as np
from PIL import Image

print("--- [LOG] Bắt đầu khởi tạo Ứng dụng ---")
sys.stdout.flush()

app = FastAPI()

# Static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load ONNX model (quantized version)
MODEL_PATH = "best_quantized.onnx"

try:
    print(f"--- [LOG] Loading model: {MODEL_PATH} ---")
    sys.stdout.flush()
    session = onnxruntime.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("--- [LOG] ONNX Model Loaded Successfully ---")
    sys.stdout.flush()
except Exception as e:
    print(f"--- [FATAL ERROR] Model load failed: {e} ---")
    sys.stdout.flush()
    raise e


def predict_image(img: Image.Image):
    img = img.resize((640, 640))
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]

    ort_inputs = {session.get_inputs()[0].name: arr}
    outputs = session.run(None, ort_inputs)[0]

    return str(outputs[0][:5])


@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == "user_demo" and password == "Test@123456":
        return RedirectResponse("/predict", status_code=302)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Sai tài khoản hoặc mật khẩu!"
    })


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")
    result = predict_image(img)

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "result": result
    })


print("--- [LOG] Khởi động thành công, FastAPI sẵn sàng trên Render ---")
sys.stdout.flush()
