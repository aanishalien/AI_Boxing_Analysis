from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="datasets/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"
)