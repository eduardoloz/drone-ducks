import cv2
from ultralytics import YOLO

# 1. Load a pre-trained YOLO model (using 'n' for nano to get maximum FPS)
model = YOLO("yolov8n.pt") 

# ... Keep your existing imports and model definition ...

cap = cv2.VideoCapture(0)  # Or your working camera index

# =============== ADD THESE THREE LINES HERE ===============
# Force the webcam backend to capture using the MJPEG codec format
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# ==========================================================

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ... Keep the rest of your while True tracking loop exactly the same ...


if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # 3. Run YOLO tracking on the current frame
    # persist=True ensures the model retains tracking IDs across frames
    results = model.track(source=frame, persist=True, show=False)

    # 4. Get the annotated frame (contains bounding boxes, labels, and track IDs)
    annotated_frame = results[0].plot()

    # Display the live tracked stream
    cv2.imshow("YOLO Live Webcam Tracking", annotated_frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()