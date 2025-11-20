# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
import gdown
import time

app = FastAPI(title="YOLOv8 ONNX - Render + Google Drive")

# CORS cho frontend gọi thoải mái
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Thư mục tạm
os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/results", exist_ok=True)

# Cấu hình model
MODEL_PATH = "/tmp/best.onnx"
DRIVE_ID = "11myMCifUc1iMzobHWvRO-S2B9H6cmL-JA"

# ──────────────────────────────
# HÀM TẢI MODEL SIÊU ỔN ĐỊNH (bất tử với Google Drive)
# ──────────────────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        print("best.onnx đã có sẵn → bỏ qua tải")
        return

    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    print("Đang tải best.onnx từ Google Drive... (chờ chút nha)")

    for attempt in range(5):  # thử tối đa 5 lần
        try:
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=False)
            print("TẢI MODEL THÀNH CÔNG!")
            return
        except Exception as e:
            print(f"Lần {attempt+1}/5 thất bại: {e}")
            time.sleep(8)  # chờ 8 giây rồi thử lại

    # Nếu 5 lần đều fail → báo lỗi rõ ràng để bạn biết
    raise Exception("""
    KHÔNG TẢI ĐƯỢC MODEL!
    Hãy làm đúng 2 bước sau:
    1. Vào https://drive.google.com/file/d/11myMCifUc1iMzobHWvRO-S2B9H6cmL-JA/view
    2. Nhấn Share → General access → Anyone with the link → Viewer
    Sau đó deploy lại là chạy ngay!
    """)

# Gọi hàm tải model (chỉ chạy 1 lần khi khởi động)
download_model()

# Load model ONNX
print("Đang load model ONNX vào RAM...")
model = YOLO(MODEL_PATH, task="detect")
print("MODEL SẴN SÀNG – BÂY GIỜ BẠN GỌI /predict THOẢI MÁI!")

# ──────────────────────────────
# API Routes
# ──────────────────────────────
@app.get("/")
async def root():
    return {"message": "YOLOv8 ONNX API đang chạy ngon lành!", "status": "ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lưu ảnh upload
    filename = file.filename
    input_path = f"/tmp/uploads/{filename}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Inference
    results = model(input_path, save=True, project="/tmp/results", name="predict", exist_ok=True)
    
    # Đường dẫn ảnh kết quả
    saved_file = results[0].save_dir + "/" + os.path.basename(input_path)
    result_filename = os.path.basename(saved_file)

    # Hostname chính xác của Render
    hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")

    return {
        "message": "Success",
        "original_image": filename,
        "result_image_url": f"https://{hostname}/img/{result_filename}",
        "detections": len(results[0].boxes) if results[0].boxes is not None else 0
    }

@app.get("/img/{filename}")
async def get_image(filename: str):
    file_path = f"/tmp/results/predict/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Ảnh kết quả không tồn tại!")
    return FileResponse(file_path, media_type="image/jpeg")
