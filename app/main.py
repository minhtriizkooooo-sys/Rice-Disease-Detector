# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import os
import sys
import requests # Thêm thư viện requests

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Cấu hình Đường dẫn và Biến Môi trường ---
os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/results", exist_ok=True)

MODEL_PATH = "/tmp/best.onnx"
# DRIVE_ID = "11myMCifUc1iMzobHWvRO-S2B9H6cmL-JA" # Không cần dùng ID nữa
MODEL_URL = os.environ.get("MODEL_DOWNLOAD_URL") # Lấy URL từ Biến môi trường

# --- Hàm tải file sử dụng Requests ---
def download_model(url: str, dest: str):
    """Tải file mô hình từ URL sử dụng thư viện requests."""
    if not url:
        print("❌ Lỗi: Không tìm thấy Biến môi trường MODEL_DOWNLOAD_URL.")
        print("Vui lòng thiết lập giá trị cho biến này trên Render Dashboard.")
        sys.exit(1)

    print(f"Đang tải best.onnx từ URL...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Kiểm tra lỗi HTTP
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("TẢI XONG!")

    except requests.exceptions.RequestException as e:
        print(f"❌ Lỗi khi tải file: {e}")
        print("Vui lòng kiểm tra lại URL tải trực tiếp đã được thiết lập đúng chưa.")
        sys.exit(1)

# --- Logic khởi chạy và Tải Mô hình ---
if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH)

try:
    model = YOLO(MODEL_PATH, task="detect")
    print("MODEL ĐÃ SẴN SÀNG – BÂY GIỜ GỌI /predict THOẢI MÁI!")
except Exception as e:
    print(f"❌ Lỗi khi khởi tạo model YOLO: {e}")
    sys.exit(1)


@app.get("/")
async def root():
    return {"message": "API YOLOv8 ONNX chạy ngon rồi nè!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ... (các hàm predict và get_image giữ nguyên)
    input_path = f"/tmp/uploads/{file.filename}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Đảm bảo thư mục results tồn tại trước khi gọi model(save=True)
    os.makedirs("/tmp/results/predict", exist_ok=True)
    
    results = model(input_path, save=True, project="/tmp/results", name="predict", exist_ok=True)
    
    # Lấy tên file đã lưu trong thư mục predict
    saved_filename = os.path.basename(input_path)
    
    hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost")
    return {
        "result_url": f"https://{hostname}/img/{saved_filename}"
    }

@app.get("/img/{filename}")
async def get_image(filename: str):
    # Đường dẫn đúng đến file kết quả
    path = f"/tmp/results/predict/{filename}"
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path)
