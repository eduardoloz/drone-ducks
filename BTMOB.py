from ultralytics import YOLO
import cv2
import threading
import time

# 1. Load YOLO
model = YOLO("yolov8n.pt")

# 2. Connect to the phone camera
url = "http://10.26.59.21:8080/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    raise RuntimeError("Could not connect to phone camera")

# Request a small capture buffer (some backends ignore this)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Shared latest-frame storage
lock = threading.Lock()
latest_frame = None
camera_running = True


def read_camera():
    global latest_frame, camera_running

    while camera_running:
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.01)
            continue

        # Replace the old frame instead of queuing frames
        with lock:
            latest_frame = frame


# 3. Start the camera reader in the background
reader = threading.Thread(target=read_camera, daemon=True)
reader.start()

try:
    while True:
        # Get the newest frame available
        with lock:
            frame = (
                latest_frame.copy()
                if latest_frame is not None
                else None
            )

        if frame is None:
            time.sleep(0.005)
            continue

        # 4. Reduce inference resolution to improve speed
        height, width = frame.shape[:2]
        target_width = 640

        if width > target_width:
            target_height = int(height * target_width / width)
            frame = cv2.resize(
                frame, (target_width, target_height)
            )

        # 5. Detect and track
        results = model.track(
            source=frame,
            tracker="bytetrack.yaml",
            persist=True,
            imgsz=640,
            verbose=False,
            conf=0.25
        )

        # 6. Display the annotated frame
        annotated_frame = results[0].plot()
        cv2.imshow(
            "YOLO + ByteTrack Phone Camera",
            annotated_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera_running = False
    reader.join(timeout=1)
    cap.release()
    cv2.destroyAllWindows()
