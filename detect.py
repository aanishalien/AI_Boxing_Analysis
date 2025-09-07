from ultralytics import YOLO

#model your trained model
model =  YOLO("runs/detect/train/weights/best.pt")

#Run interface on a video

results = model.predict(
    source="source.mp4",
    conf=0.4,
    imgsz=320,
    show=True,
    save=True
)

print("Done? Results saved in:", results[0].save_dir)