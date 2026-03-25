import os
import cv2
import numpy as np
import threading
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from ultralytics import YOLO
from supabase import create_client, Client

app = Flask(__name__)
trap_status = {"is_closed": False, "last_conf": 0.0, "last_seen": "None"}
lock = threading.Lock()

# Supabase Configuration
SUPABASE_URL = "https://fiygbkzcuepatjykwqns.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZpeWdia3pjdWVwYXRqeWt3cW5zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0MzA2NDcsImV4cCI6MjA2NDAwNjY0N30.yir3D_TXcEYoOdtsDy2kNRcPaX2r7rtGMnaYA7vQ9dE"
SUPABASE_BUCKET = "detection-images"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- AI MODEL LOADING (MEMORY OPTIMIZED) ---
MODEL_PATH = "model/best.pt"
# We load the model outside the request to keep the worker responsive
try:
    # 'task=detect' and '.to(cpu)' help prevent unnecessary memory allocation
    model = YOLO(MODEL_PATH, task='detect')
    model.to('cpu')
    print("AI Model Loaded Successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    # This route helps Render's "Port Scan" succeed quickly
    return "RODWAY Cloud Control Center: ONLINE", 200

@app.route('/predict', methods=['POST'])
def predict():
    global trap_status
    
    if model is None:
        return "model_not_ready", 503

    # Get RAW bytes from ESP32-CAM
    img_bytes = request.data 
    if not img_bytes:
        return "no_data", 400

    # Convert bytes to OpenCV image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return "invalid_img", 400

    # Run AI Inference (imgsz=320 is vital for 512MB RAM)
    results = model.predict(img, imgsz=320, conf=0.5, verbose=False)
    
    found_rat = False
    max_conf = 0.0

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            max_conf = max(max_conf, conf)
            # Adjust the confidence threshold here if needed
            if label == "rat" and conf >= 0.60:
                found_rat = True

    if found_rat and not trap_status["is_closed"]:
        with lock:
            trap_status["is_closed"] = True
            trap_status["last_conf"] = round(max_conf, 2)
            trap_status["last_seen"] = f"Rat detected at {datetime.now().strftime('%H:%M:%S')}"

        # Trigger Supabase & Upload Image in background threads
        threading.Thread(target=cloud_trigger_signal).start()
        ts = datetime.now(timezone.utc)
        threading.Thread(target=background_upload, args=(img_bytes, "rat", ts, max_conf)).start()

        return "trigger" 

    return "monitoring"

def cloud_trigger_signal():
    try:
        supabase.table("trap_control").update({"is_triggered": True}).eq("id", 1).execute()
    except Exception as e:
        print(f"Bridge Error: {e}")

def background_upload(img_bytes, label, ts, confidence):
    try:
        file_name = f"{ts.strftime('%Y%m%d_%H%M%S')}_{label}.jpg"
        storage_path = f"trap_1/{file_name}"
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            path=storage_path, 
            file=img_bytes, 
            file_options={"content-type": "image/jpeg"}
        )
        
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
        data = {
            "trap_id": 1, 
            "image_url": public_url, 
            "is_rodent": True, 
            "captured_at": ts.isoformat(), 
            "confidence": confidence, 
            "status": "detected"
        }
        supabase.table("captures").insert(data).execute()
    except Exception as e:
        print(f"Upload error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)