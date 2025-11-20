# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import gdown

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Thư mục tạm Render cho sẵn
os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/results", exist_ok=True)

# Đường dẫn model (Render sẽ tự tải về đây)
MODEL_PATH = "/tmp/best.onnx"
DRIVE_ID = "11myMCifUc1iMzobHWvRO-S2B9H6cmL-JA"

# Tự động tải nếu chưa có (chỉ chạy khi khởi động)
if not os.path.exists(MODEL_PATH):
    print("Đang tải best.onnx từ Google Drive lần đầu...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_ID}", MODEL_PATH, quiet=False)
    print("Tải xong! Bắt đầu load model...")
else:
    print("best.onnx đã có sẵn trong /tmp")

# Load model ONNX
model = YOLO(MODEL_PATH, task="detect")
print("Model ONNX đã sẵn sàng!")

@app.get("/")
async def root():
    return {"message": "YOLOv8 ONNX từ Google Drive – chạy ngon!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    input_path = f"/tmp/uploads/{file.filename}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(input_path, save=True, project="/tmp/results", name="predict", exist_ok=True)
    saved_file = results[0].save_dir + "/" + os.path.basename(input_path)

    hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")
    return {
        "result_url": f"https://{hostname}/img/{os.path.basename(saved_file)}"
    }

@app.get("/img/{filename}")
async def get_image(filename: str):
    path = f"/tmp/results/predict/{filename}"
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    from fastapi.responses import FileResponse
    return FileResponse(path)
