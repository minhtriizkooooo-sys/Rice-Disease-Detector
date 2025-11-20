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
import requests # Thêm thư viện requests

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Cấu hình Đường dẫn và Biến Môi trường ---
os.makedirs("/tmp/uploads", exist_ok=True)
os.makedirs("/tmp/results", exist_ok=True)

MODEL_PATH = "/tmp/best.onnx"
MODEL_URL = os.environ.get("MODEL_DOWNLOAD_URL") # Lấy URL từ Biến môi trường

# --- Hardcoded User (Chỉ dùng cho mục đích demo) ---
DEMO_USER = "user_demo"
DEMO_PASSWORD = "Test@123456"

# Class dùng cho Token response
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

# --- Hàm tải file sử dụng Requests ---
def download_model(url: str, dest: str):
    """Tải file mô hình từ URL sử dụng thư viện requests."""
    if not url:
        print("❌ Lỗi: Không tìm thấy Biến môi trường MODEL_DOWNLOAD_URL.")
        print("Vui lòng thiết lập giá trị cho biến này trên Render Dashboard.")
        sys.exit(1)

    print(f"Đang tải best.onnx từ URL...")
    
    try:
        response = requests.get(url, stream=True, timeout=300) # Thêm timeout 5 phút
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
    # Phục vụ file HTML (sẽ được tạo ra ở bước 2)
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
