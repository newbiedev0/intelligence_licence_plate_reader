import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
import base64


MODEL_PATH ="model/best_yolo.pt"



LOCAL_BG_PATH = "background/custom_BG.jpg"


FALLBACK_BG_URL = "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?q=80&w=1920&auto=format&fit=crop"

def simple_normalize_plate(raw_text):
    if not raw_text:
        return ""
    cleaned = "".join(ch for ch in raw_text.upper() if ch.isalnum())

    replace_map = {
        "O": "0",
        "0": "O",
        "I": "1",
        "1": "I"
    }

    return "".join(replace_map.get(c, c) for c in cleaned)


def draw_boxes(img, detections):
    img2 = img.copy()
    for d in detections:
        x1,y1,x2,y2 = d["bbox"]
        conf = d.get("conf", 0)
        label = d.get("normalized","")

        cv2.rectangle(img2, (x1,y1), (x2,y2), (0,255,0), 2)

        text = f"{label} {conf:.2f}" if label else f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img2, (x1, max(0,y1-22)), (x1+tw+6, y1), (0,255,0), -1)
        cv2.putText(img2, text, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
    return img2


@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return YOLO(path)

@st.cache_resource
def load_easyocr_reader(gpu_flag=True):
  
    try:
        return easyocr.Reader(['en'], gpu=gpu_flag)
    except Exception:
       
        return easyocr.Reader(['en'], gpu=False)


def set_background_image(local_path=None, url=None, opacity=0.12):

    css = ""
    if local_path and os.path.exists(local_path):
        with open(local_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        mime = "image/jpeg"
        if local_path.lower().endswith(".png"):
            mime = "image/png"
        css = f"""
        <style>
        .stApp {{
          background-image: url("data:{mime};base64,{b64}");
          background-size: cover;
          background-position: center;
        }}
        /* overlay to dim the image so UI controls are readable */
        .stApp::before {{
          content: "";
          position: fixed;
          inset: 0;
          background: rgba(255,255,255,{opacity});
          pointer-events: none;
        }}
        </style>
        """
    elif url:
        css = f"""
        <style>
        .stApp {{
          background-image: url("{url}");
          background-size: cover;
          background-position: center;
        }}
        .stApp::before {{
          content: "";
          position: fixed;
          inset: 0;
          background: rgba(255,255,255,{opacity});
          pointer-events: none;
        }}
        </style>
        """
    if css:
        st.markdown(css, unsafe_allow_html=True)


st.set_page_config(page_title="ILP-Reader", layout="centered")
set_background_image(local_path=LOCAL_BG_PATH, url=FALLBACK_BG_URL, opacity=0.20)

st.markdown("""
<h1 style="
text-align:center;
font-family:Segoe UI;
background: linear-gradient(90deg, #00C9FF, #92FE9D);
-webkit-background-clip: text;
color: transparent;">
Intelligent License Plate Reader
</h1>
""", unsafe_allow_html=True)

use_gpu = st.checkbox("Use GPU for EasyOCR (if available)", value=True)
conf_thr = st.slider("Detection confidence threshold", 0.05, 0.9, 0.25, 0.01)

img_file = st.file_uploader("Upload an image to test", type=["jpg", "jpeg", "png"])
run_btn = st.button("Run")

status = st.empty()

if run_btn:

    
    model_path_to_use = MODEL_PATH

    
    if not os.path.exists(model_path_to_use):
        status.error(f"Model not found at '{model_path_to_use}'. Fix your MODEL_PATH at the top of the script.")
        st.stop()

    status.info("Loading model and OCR ...")

   
    try:
        model = load_yolo_model(model_path_to_use)
    except Exception as e:
        status.error(f"Failed to load model: {e}")
        st.stop()

   
    try:
        reader = load_easyocr_reader(use_gpu)
    except Exception as e:
        status.warning(f"EasyOCR failed with GPU={use_gpu}. Retrying CPU.")
        reader = load_easyocr_reader(False)

    
    if img_file is None:
        status.error("Please upload an image to test.")
        st.stop()


    image = Image.open(img_file).convert("RGB")
    img_np = np.array(image)[:, :, ::-1] 

    status.info("Running detection...")
    try:
        results = model.predict(source=img_np, imgsz=640, conf=conf_thr, verbose=False)
    except Exception as e:
        status.error(f"Detection failed: {e}")
        st.stop()

    r = results[0]
    detections = []
    try:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
    except Exception:
        boxes = []
        scores = []

    for (box, conf) in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        h, w = img_np.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if (x2-x1) < 15 or (y2-y1) < 8:
            continue

        try:
            ocr_out = reader.readtext(crop)
            ocr_texts = [t[1] for t in ocr_out]
            raw_joined = " ".join(ocr_texts)
        except Exception:
            raw_joined = ""
            ocr_texts = []

        normalized = simple_normalize_plate(raw_joined)
        detections.append({
            "bbox":[x1,y1,x2,y2],
            "conf": float(conf),
            "ocr_raw": raw_joined,
            "normalized": normalized
        })

    status.success(f"RESULT — {len(detections)} plate detected")

    img_with_boxes = draw_boxes(img_np.copy(), detections)
    img_display = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    st.image(img_display, caption="Detections", use_container_width=True)


    if detections:
        df = pd.DataFrame([{
            "bbox": d["bbox"],
            "conf": d["conf"],
            "ocr_raw": d["ocr_raw"],
            "normalized": d["normalized"]
        } for d in detections])
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results CSV", csv, "detections.csv", "text/csv")
    else:
        st.info("No detections were found at your confidence threshold.")





