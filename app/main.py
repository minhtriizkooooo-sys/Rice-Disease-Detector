# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import os
import gdown

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/results", exist_ok=True)

MODEL_PATH = "/tmp/best.onnx"
DRIVE_ID = "11myMCifUc1iMzobHWvRO-S2B9H6cmL-JA"   # ← ID của em

# Hàm tải chắc chắn thành công
if not os.path.exists(MODEL_PATH):
    print("Đang tải model từ Google Drive (chắc chắn thành công lần này)...")
    gdown.download(
        url=f"https://drive.google.com/uc?id={DRIVE_ID}",
        output=MODEL_PATH,
        quiet=False,
        fuzzy=False
    )
    print("TẢI XONG 100%!")

model = YOLO(MODEL_PATH, task="detect")
print("MODEL ĐÃ SẴN SÀNG – BÂY GIỜ GỌI /predict THOẢI MÁI!")

@app.get("/"); async def root(): return {"message": "API YOLOv8 ONNX chạy ngon rồi nè!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    input_path = f"/tmp/uploads/{file.filename}"
    with open(input_path, "wb") as f: shutil.copyfileobj(file.file, f)
    
    results = model(input_path, save=True, project="/tmp/results", name="predict", exist_ok=True)
    saved_file = results[0].save_dir + "/" + os.path.basename(input_path)
    
    hostname = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost")
    return {
        "result_url": f"https://{hostname}/img/{os.path.basename(saved_file)}"
    }

@app.get("/img/{filename}")
async def get_image(filename: str):
    path = f"/tmp/results/predict/{filename}"
    if not os.path.exists(path): raise HTTPException(404)
    return FileResponse(path)
