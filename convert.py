from ultralytics import YOLO

# This loads your model and prepares it for the "Diet"
model = YOLO("model/best.pt")

# Export to ONNX (Lightweight format for Render)
# imgsz=160 makes the internal math 4x smaller than 320px
path = model.export(format="onnx", imgsz=160, simplify=True)

print(f"--- DONE! Model saved at: {path} ---")