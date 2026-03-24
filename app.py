import requests
from datetime import datetime, timezone
from ultralytics import YOLO
import cv2
import os
import time
import threading
from flask import Flask, render_template_string, Response, jsonify
from supabase import create_client, Client

# =======================
# Configuration & State
# =======================
app = Flask(__name__)
trap_status = {
    "is_closed": False, 
    "last_conf": 0.0, 
    "last_seen": "None", 
    "hardware_msg": "System Online"
}
output_frame = None
lock = threading.Lock()

# ARDUINO SYNC SETTINGS
FALL_DELAY = 7  # Matches 'const int FALL_DELAY = 7000' in Arduino
NODEMCU_IP = "10.83.114.63" 
ESP32_IP = "10.83.114.125" 
ESP32_URL = f"http://{ESP32_IP}/cam.mjpeg"

# AI Sensitivity
REAL_RAT_CONF = 0.60      
FALSE_ALARM_CONF = 0.30   

# Supabase Setup
SUPABASE_URL = "https://fiygbkzcuepatjykwqns.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZpeWdia3pjdWVwYXRqeWt3cW5zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0MzA2NDcsImV4cCI6MjA2NDAwNjY0N30.yir3D_TXcEYoOdtsDy2kNRcPaX2r7rtGMnaYA7vQ9dE"
SUPABASE_BUCKET = "detection-images"
MODEL_PATH = "best.pt"
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =======================
# Wireless Trigger Logic
# =======================
def send_wireless_trigger():
    url = f"http://{NODEMCU_IP}/close"
    try:
        # Short timeout: NodeMCU handles the 7s delay; Python just needs the 'OK'
        response = requests.get(url, timeout=5) 
        if response.status_code == 200:
            print(f"[WIRELESS] Success: Triggered NodeMCU at {NODEMCU_IP}")
            return True
    except Exception as e:
        print(f"[WIRELESS] Trigger failed: {e}")
    return False

def sync_status_with_hardware():
    """Waits for hardware reset cycle before updating UI."""
    print(f"[SYNC] Hardware is busy for {FALL_DELAY}s...")
    time.sleep(FALL_DELAY + 0.5) 
    with lock:
        trap_status["is_closed"] = False
        print("[SYNC] Trap status reset to MONITORING")

# =======================
# Flask Web Interface
# =======================
@app.route('/')
def index():
    return render_template_string("""
    <body style="font-family:sans-serif; text-align:center; background:#121212; color:white; padding:20px;">
        <h1 style="color: #00ff00;">RODWAY Wireless Control Center</h1>
        <div style="margin:20px auto; width:640px; border:5px solid #333; background:#000; border-radius:10px; overflow:hidden;">
            <img src="/video_feed" style="width:100%;">
        </div>
        <div style="background:#222; padding:20px; display:inline-block; border-radius:10px; min-width:400px; border: 1px solid #444;">
            <h2>Status: <span id="st">...</span></h2>
            <p>AI Confidence: <b id="co" style="color:#007bff;">0.0</b> | Last Seen: <b id="ev">None</b></p>
            <hr style="border: 0.5px solid #333;">
            <button onclick="fetch('/trigger')" style="padding:15px; background:#d9534f; color:white; border:none; cursor:pointer; font-weight:bold; border-radius:5px; margin:5px; width:180px;">MANUAL TRIGGER</button>
            <button onclick="fetch('/reset')" style="padding:15px; background:#5cb85c; color:white; border:none; cursor:pointer; font-weight:bold; border-radius:5px; margin:5px; width:180px;">UI RESET</button>
        </div>
        <script>
            setInterval(async () => {
                try {
                    const r = await fetch('/status');
                    const d = await r.json();
                    const stEl = document.getElementById('st');
                    stEl.innerText = d.is_closed ? "TRAP CLOSED (Wait 7s)" : "MONITORING";
                    stEl.style.color = d.is_closed ? "#ff4444" : "#00ff00";
                    document.getElementById('co').innerText = d.last_conf;
                    document.getElementById('ev').innerText = d.last_seen;
                } catch(e) {}
            }, 1000);
        </script>
    </body>
    """, node_ip=NODEMCU_IP, cam_ip=ESP32_IP)

@app.route('/status')
def get_status(): return jsonify(trap_status)

@app.route('/trigger')
def manual_trigger():
    if not trap_status["is_closed"]:
        if send_wireless_trigger():
            trap_status["is_closed"] = True
            threading.Thread(target=sync_status_with_hardware, daemon=True).start()
            return "Triggered"
    return "NodeMCU Busy/Offline", 500

@app.route('/reset')
def reset_trap():
    trap_status["is_closed"] = False
    return "Reset"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None: continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# =======================
# AI Core & Cloud Worker
# =======================
def background_upload(local_path, label, ts, confidence):
    try:
        file_name = f"{ts.strftime('%Y%m%d_%H%M%S')}_{label}.jpg"
        storage_path = f"trap_1/{file_name}"
        
        # Upload image
        with open(local_path, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=storage_path, 
                file=f,
                file_options={"content-type": "image/jpeg"}
            )
        
        # Get Public URL
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
        
        # Insert metadata
        data = {
            "trap_id": 1,
            "image_url": public_url,
            "is_rodent": True if label == "rat" else False,
            "captured_at": ts.isoformat(),
            "confidence": confidence,
            "status": "detected" if label == "rat" else "false_trigg"
        }
        
        supabase.table("captures").insert(data).execute()
        print(f"[CLOUD] Detection logged successfully.")
    except Exception as e:
        print(f"[CLOUD] Upload error: {e}")

# ... (Keep all previous imports and configuration as they were)

def run_ai_logic():
    global trap_status, output_frame, lock
    model = YOLO(MODEL_PATH)
    
    while True:
        cap = cv2.VideoCapture(ESP32_URL)
        while True:
            ret, frame = cap.read()
            if not ret:
                break 

            results = model(frame, stream=True, verbose=False)
            for result in results:
                annotated = result.plot()
                with lock:
                    output_frame = annotated.copy()

                for box in result.boxes:
                    conf = float(box.conf[0])
                    # Get the class ID and the class name
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id].lower() # Converts "Rat" or "Others" to lowercase
                    
                    trap_status["last_conf"] = round(conf, 2)

                    # MODIFIED LOGIC: Check if it's a "rat" AND meets confidence
                    # This ignores the "Others" class even if confidence is high
                    if not trap_status["is_closed"] and label == "rat" and conf >= REAL_RAT_CONF:
                        if send_wireless_trigger():
                            trap_status["is_closed"] = True
                            trap_status["last_seen"] = f"Rat ({conf})"
                            
                            ts = datetime.now(timezone.utc)
                            path = os.path.join(SAVE_DIR, "capture.jpg")
                            cv2.imwrite(path, frame)
                            
                            # Log to Supabase (keeping your existing logic)
                            threading.Thread(target=background_upload, args=(path, "rat", ts, conf), daemon=True).start()
                            threading.Thread(target=sync_status_with_hardware, daemon=True).start()
                    
                    # Optional: Log 'Others' to UI status without triggering the trap
                    elif label != "rat":
                        trap_status["last_seen"] = f"Ignored: {label} ({conf})"

# ... (Keep the rest of the script exactly as it was)
if __name__ == '__main__':
    threading.Thread(target=run_ai_logic, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)