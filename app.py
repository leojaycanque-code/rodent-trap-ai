import os
import cv2
import numpy as np
import threading
from datetime import datetime, timezone
from flask import Flask, request
from supabase import create_client, Client

app = Flask(__name__)   

# --- CONFIG ---
SUPABASE_URL = "https://fiygbkzcuepatjykwqns.supabase.co" 
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZpeWdia3pjdWVwYXRqeWt3cW5zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDg0MzA2NDcsImV4cCI6MjA2NDAwNjY0N30.yir3D_TXcEYoOdtsDy2kNRcPaX2r7rtGMnaYA7vQ9dE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL_PATH = "model/best.onnx"
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

def save_to_supabase(img_bytes, confidence):
    """Fulfills Analytics & Image Viewing Objectives"""
    try:
        # 1. Create timestamped filename
        ts = datetime.now(timezone.utc)
        filename = f"trap_1_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"

        # 2. Upload to Storage (For Website Image Viewing)
        supabase.storage.from_("detection-images").upload(
            path=filename, 
            file=img_bytes, 
            file_options={"content-type": "image/jpeg"}
        )
        
        # 3. Get Public URL
        img_url = supabase.storage.from_("detection-images").get_public_url(filename)

        # 4. Insert Record (For Website Analytics & App Notifications)
        # Note: 'captured_at' handles your 'Frequency Over Time' objective
        capture_data = {
            "trap_id": 1,
            "image_url": img_url,
            "confidence": round(float(confidence), 2),
            "captured_at": ts.isoformat(),
            "status": "captured"
        }
        supabase.table("captures").insert(capture_data).execute()

        # 5. Flip the control switch for NodeMCU
        supabase.table("trap_control").update({"is_triggered": True}).eq("id", 1).execute()
        
        print(f"COMPLETE: Data synced to Supabase. Image: {img_url}")

    except Exception as e:
        print(f"Data Pipeline Error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = request.data
    if not img_bytes: return "no_data", 400

    # Decode image for AI
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 160x160 Inference
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (160, 160), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    predictions = np.squeeze(outputs).T
    max_conf = np.max(predictions[:, 4])

    print(f"AI Check: {max_conf:.2f}")

    if max_conf >= 0.50:
        # Run cloud tasks in background so ESP32 stays fast (Performance Objective)
        threading.Thread(target=save_to_supabase, args=(img_bytes, max_conf)).start()
        return "trigger"

    return "monitoring"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)