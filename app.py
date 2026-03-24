import os
import cv2
import numpy as np
import threading
from datetime import datetime, timezone
from flask import Flask, render_template_string, request, jsonify
from ultralytics import YOLO
from supabase import create_client, Client

# =======================
# Configuration & State
# =======================
app = Flask(__name__)
trap_status = {"is_closed": False, "last_conf": 0.0, "last_seen": "None"}
lock = threading.Lock()

# Supabase (Credentials from your script)
SUPABASE_URL = "https://fiygbkzcuepatjykwqns.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZpeWdia3pjdWVwYXRqeWt3cW5zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0MzA2NDcsImV4cCI6MjA2NDAwNjY0N30.yir3D_TXcEYoOdtsDy2kNRcPaX2r7rtGMnaYA7vQ9dE"
SUPABASE_BUCKET = "detection-images"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load Model (Note: path updated for typical Render folder structure)
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH).to('cpu') # Force CPU for Render stability

# =======================
# Routes
# =======================
@app.route('/')
def index():
    return "<h1>RODWAY Cloud Control Center: ONLINE</h1>"

@app.route('/status')
def get_status(): return jsonify(trap_status)

@app.route('/predict', methods=['POST'])
def predict():
    global trap_status
    if 'imageFile' not in request.files:
        return jsonify({"status": "error", "message": "No image"}), 400

    # 1. Receive Image from ESP32-CAM
    file = request.files['imageFile']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run AI Inference (Optimized imgsz for 512MB RAM)
    results = model.predict(img, imgsz=320, conf=0.5, verbose=False)
    
    found_rat = False
    max_conf = 0.0

    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])].lower()
            conf = float(box.conf[0])
            max_conf = max(max_conf, conf)
            if label == "rat" and conf >= 0.60:
                found_rat = True

    # 3. Trigger Decision
    if found_rat and not trap_status["is_closed"]:
        with lock:
            trap_status["is_closed"] = True
            trap_status["last_conf"] = round(max_conf, 2)
            trap_status["last_seen"] = f"Rat detected at {datetime.now().strftime('%H:%M:%S')}"

        # Bridge: Tell Supabase to signal the NodeMCU
        threading.Thread(target=cloud_trigger_signal).start()
        
        # Cloud Log
        ts = datetime.now(timezone.utc)
        threading.Thread(target=background_upload, args=(img_bytes, "rat", ts, max_conf)).start()

        return jsonify({"status": "trigger", "message": "CLOSE_TRAP"})

    return jsonify({"status": "monitoring", "conf": round(max_conf, 2)})

# =======================
# Cloud Workers
# =======================
def cloud_trigger_signal():
    """Updates a flag in Supabase that the NodeMCU polls for"""
    try:
        supabase.table("trap_control").update({"is_triggered": True}).eq("id", 1).execute()
    except Exception as e:
        print(f"Bridge Error: {e}")

def background_upload(img_bytes, label, ts, confidence):
    try:
        file_name = f"{ts.strftime('%Y%m%d_%H%M%S')}_{label}.jpg"
        storage_path = f"trap_1/{file_name}"
        supabase.storage.from_(SUPABASE_BUCKET).upload(path=storage_path, file=img_bytes, file_options={"content-type": "image/jpeg"})
        
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
        data = {"trap_id": 1, "image_url": public_url, "is_rodent": True, "captured_at": ts.isoformat(), "confidence": confidence, "status": "detected"}
        supabase.table("captures").insert(data).execute()
    except Exception as e:
        print(f"Upload error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)