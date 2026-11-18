from ultralytics import YOLO

MODEL = "yolov8n.pt"
DATA = "dataset/data.yaml"


def main():
    model = YOLO(MODEL)

    # the parameters ran on gpu
    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=416,
        batch=32,
        device=0,
        patience=15,
        project="runs",
        name="helmet_v8s",
        exist_ok=True,
        seed=42,
    )


if __name__ == "__main__":
    main()