import os 
import sys 
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import onnxruntime
import numpy as np
from PIL import Image
from pathlib import Path 

# Thêm logging để theo dõi khởi động
print("--- [LOG] Starting Application Initialization ---")
sys.stdout.flush()

app = FastAPI()

# --- SỬ LÝ ĐƯỜNG DẪN MẠNH MẼ ---
BASE_DIR = Path(__file__).resolve().parent

# Setup Static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Setup Templates (Bắt lỗi khởi tạo Jinja2)
try:
    templates = Jinja2Templates(directory=BASE_DIR / "templates")
    print("--- [LOG] Jinja2Templates initialized successfully. ---")
    sys.stdout.flush()
except Exception as e:
    print(f"--- [FATAL ERROR] JINJA2 TEMPLATES FAILED TO INITIALIZE: {e} ---")
    sys.stdout.flush()
    raise e 

# Load model ONNX (Tối ưu hóa)
try:
    MODEL_PATH = "best_quantized.onnx"
    print(f"--- [LOG] Starting model loading for {MODEL_PATH} (Quantized) ---")
    sys.stdout.flush()
    
    if not os.path.exists(MODEL_PATH):
        print(f"--- [FATAL ERROR] FILE NOT FOUND: Model file '{MODEL_PATH}' not found. ---")
        sys.stdout.flush()
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

    # SỬA LỖI CUỐI CÙNG: Thêm SessionOptions để tối ưu hóa môi trường cloud
    sess_options = onnxruntime.SessionOptions()
    # Tắt các tối ưu hóa đa luồng không cần thiết (giúp tránh xung đột)
    sess_options.intra_op_num_threads = 1 
    sess_options.inter_op_num_threads = 1

    # Khởi tạo Inference Session
    session = onnxruntime.InferenceSession(
        MODEL_PATH, 
        sess_options=sess_options,  # Sử dụng tùy chọn tối ưu hóa
        providers=["CPUExecutionProvider"]
    )
    print("--- [LOG] ONNX model loaded successfully! ---")
    sys.stdout.flush()
except Exception as e:
    print(f"--- [FATAL ERROR] MODEL LOADING FAILED: {e} ---")
    sys.stdout.flush()
    raise e 

### --- Helper predict function ---
def predict_image(img: Image.Image):
    """
    Pre-processes the image and runs inference. 
    """
    img = img.resize((640, 640))
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    
    ort_inputs = {session.get_inputs()[0].name: arr}
    outputs = session.run(None, ort_inputs)[0]
    
    return str(outputs[0][:5])

### ---------- LOGIN PAGE ----------
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

### ---------- PREDICT PAGE ----------
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

print("--- [LOG] Initialization complete, ready to serve FastAPI ---")
sys.stdout.flush()
