from ultralytics import YOLO

# Load your custom model
model = YOLO("model/best.pt")

# Export to ONNX with optimizations for Render's 512MB RAM
model.export(format="onnx", imgsz=160, simplify=True)

print("--- SUCCESS! ---")
print("Your new model is saved at: model/best.onnx")