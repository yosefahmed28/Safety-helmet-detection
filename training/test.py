from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path("runs/detect/runs/helmet_v8s/weights/best.pt")

model = YOLO(MODEL_PATH)
model.val(data="dataset/data.yaml", split="test", imgsz=416, device="cpu",
          project="runs", name="test_eval", exist_ok=True)