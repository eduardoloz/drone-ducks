from ultralytics import YOLO

model = YOLO("yolo11n.pt")

for result in model.track(
    source="video_h264.mp4",
    show=False,
    save=True,
    stream=True,
    persist = True,
    tracker="bytetrack.yaml",
):
    print(f"Detections: {len(result.boxes)}")