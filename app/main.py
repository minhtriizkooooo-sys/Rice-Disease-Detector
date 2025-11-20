# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from ultralytics import YOLO
from pydantic import BaseModel
import shutil
import os
import sys
# Không cần requests vì YOLO sẽ tự tải model từ ID/tên file

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Cấu hình Đường dẫn và Biến Môi trường ---
os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/results", exist_ok=True)

# Lấy ID mô hình từ Biến môi trường. 
# Ví dụ: 'ten-nguoi-dung/ten-repo-cua-ban.onnx' hoặc 'best.onnx' (nếu đã upload sẵn trong repo)
# Nếu không có, sử dụng mô hình mặc định 'yolov8n.pt' làm fallback.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "yolov8n.pt") 

# --- Hardcoded User (Chỉ dùng cho mục đích demo) ---
DEMO_USER = "user_demo"
DEMO_PASSWORD = "Test@123456"

# Class dùng cho Token response
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# --- Logic khởi chạy và Tải Mô hình ---
try:
    # YOLO sẽ tự động tải model từ Hugging Face Hub (nếu là ID repo)
    # hoặc tìm kiếm model trong thư mục làm việc (nếu là tên file như 'best.onnx')
    print(f"Đang tải và khởi tạo model: {HF_MODEL_ID}...")
    model = YOLO(HF_MODEL_ID, task="detect")
    print("MODEL ĐÃ SẴN SÀNG – BÂY GIỜ GỌI /predict THOẢI MÁI!")
except Exception as e:
    print(f"❌ Lỗi khi khởi tạo model YOLO bằng ID {HF_MODEL_ID}: {e}")
    print("Vui lòng kiểm tra lại HF_MODEL_ID, đảm bảo model đã được tải lên Hugging Face Hub hoặc file mô hình đã nằm trong thư mục gốc của dự án.")
    sys.exit(1)

# --- Endpoint Xác thực ---
@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Lưu ý: form_data.username và form_data.password
    if form_data.username == DEMO_USER and form_data.password == DEMO_PASSWORD:
        # Trong ứng dụng thực tế, bạn sẽ tạo JWT ở đây. 
        # Ở đây, chúng ta chỉ trả về username làm "token" đơn giản để client lưu trữ.
        return {"access_token": form_data.username, "token_type": "bearer"}
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

# --- Dependency để kiểm tra Token (Rất đơn giản) ---
def verify_token(token: str = Depends(OAuth2PasswordRequestForm)):
    if token != DEMO_USER:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# --- Endpoint Phục vụ Giao diện ---
@app.get("/", response_class=HTMLResponse)
async def root():
    # Phục vụ file HTML
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend HTML file not found!</h1>", status_code=500)

@app.post("/predict")
async def predict(file: UploadFile = File(...), user: str = Depends(verify_token)):
    # user: Biến này chứa 'user_demo' nếu xác thực thành công
    # Logic dự đoán giữ nguyên
    input_path = f"/tmp/uploads/{file.filename}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    os.makedirs("/tmp/results/predict", exist_ok=True)
    
    # Chạy mô hình
    results = model(input_path, save=True, project="/tmp/results", name="predict", exist_ok=True)
    
    # Lấy tên file đã lưu trong thư mục predict
    saved_filename = os.path.basename(input_path)
    
    # Lưu ý: Client sẽ tự ghép hostname.
    return {
        "result_filename": saved_filename
    }

@app.get("/img/{filename}")
async def get_image(filename: str, user: str = Depends(verify_token)):
    # user: Biến này chứa 'user_demo' nếu xác thực thành công
    path = f"/tmp/results/predict/{filename}"
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path)
