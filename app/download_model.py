# app/download_model.py
import os
import gdown

# THAY ID CỦA BẠN VÀO DÒNG DƯỚI ĐÂY
DRIVE_ID = "11myMCifUc1iMzobHWvRO-S2B9H6cmL-JA"   # ←←← sửa thành ID của bạn

URL = f"https://drive.google.com/uc?id={DRIVE_ID}"
MODEL_PATH = "model/best.onnx"

if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    print("Đang tải best.onnx từ Google Drive... (lần đầu hơi lâu tí)")
    gdown.download(URL, MODEL_PATH, quiet=False)
    print("Tải model xong! Bây giờ có thể dùng bình thường")
else:
    print("Model đã có sẵn, bỏ qua tải")