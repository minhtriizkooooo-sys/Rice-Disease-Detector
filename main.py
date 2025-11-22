import sys
import numpy as np
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from .model import yolo_session, run_inference
from .utils import draw_boxes
from .disease_names import DISEASE_NAMES

print("\n--- [LOG] STARTING FASTAPI APP ---")
sys.stdout.flush()

app = FastAPI()

# static + template
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# -----------------------------
# LOGIN PAGE
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == "user_demo" and password == "Test@123456":
        return RedirectResponse("/predict", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Sai tài khoản hoặc mật khẩu!"})


# -----------------------------
# PREDICT PAGE
# -----------------------------
@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    img = Image.open(image.file).convert("RGB")
    results = run_inference(img)

    boxes, scores, classes = results

    # convert names
    diseases_info = []
    for b, s, c in zip(boxes, scores, classes):
        diseases_info.append({
            "disease": DISEASE_NAMES.get(int(c), "Unknown"),
            "confidence": float(s)
        })

    # draw bounding boxes
    img_out = draw_boxes(img, boxes, scores, classes)
    img_b64 = img_out

    return JSONResponse({
        "image": img_b64,
        "diseases": diseases_info
    })
