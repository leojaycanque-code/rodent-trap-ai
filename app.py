import os
import cv2
import numpy as np
import threading
from flask import Flask, request
from supabase import create_client, Client

app = Flask(__name__)

# --- CONFIGURATION ---
SUPABASE_URL = "https://fiygbkzcuepatjykwqns.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZpeWdia3pjdWVwYXRqeWt3cW5zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0MzA2NDcsImV4cCI6MjA2NDAwNjY0N30.yir3D_TXcEYoOdtsDy2kNRcPaX2r7rtGMnaYA7vQ9dE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- LIGHTWEIGHT AI LOADING ---
MODEL_PATH = "model/best.onnx"
# Load the model once during startup
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

def get_max_confidence(img):
    """Parses YOLOv8 ONNX output for the highest confidence score"""
    # 1. Resize and normalize image for the 160x160 model
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (160, 160), swapRB=True, crop=False)
    net.setInput(blob)
    
    # 2. Run Inference
    outputs = net.forward() # Shape is usually [1, 5, 525]
    
    # 3. Parse YOLOv8 format: Transpose to [525, 5] (where 5 is x, y, w, h, conf)
    predictions = np.squeeze(outputs).T
    
    # 4. Get the highest confidence score from the 5th column (index 4)
    # If you have multiple classes, you'd check columns 4, 5, 6...
    confidences = predictions[:, 4]
    max_conf = np.max(confidences)
    
    return max_conf

@app.route('/')
def health_check():
    return "AI Node: ACTIVE", 200

@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = request.data
    if not img_bytes: return "no_data", 400

    # Decode image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return "invalid_image", 400

    # Run AI logic
    try:
        max_score = get_max_confidence(img)
        print(f"AI Confidence: {max_score:.2f}")

        # TRIGGER THRESHOLD: 50% confidence
        if max_score > 0.50:
            threading.Thread(target=trigger_supabase).start()
            return "trigger"
            
    except Exception as e:
        print(f"Inference Error: {e}")
        return "error", 500

    return "monitoring"

def trigger_supabase():
    """Updates Supabase in a separate thread to keep the ESP32 connection fast"""
    try:
        supabase.table("trap_control").update({"is_triggered": True}).eq("id", 1).execute()
        print("--- RAT DETECTED: TRAP TRIGGERED ---")
    except Exception as e:
        print(f"Supabase Error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)