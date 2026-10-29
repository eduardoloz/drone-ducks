#Final Bytetrack Script (Annotated)

#Imports
from ultralytics import YOLO #YOLO from Ultralytics
import cv2 #Python Computer Vision Library
import threading #Similar to task.spawn() in Lua (Parallel Tasks)
import time #Delays

model = YOLO("yolov8n.pt")

#Create a Video Capture Object
url = "http://10.26.59.21:8080/video" #Change to Provided Camera Phone App URL
cap = cv2.VideoCapture(url)

#If Starting Stream is Unsuccessful
if not cap.isOpened():
    raise RuntimeError("Could not connect to phone camera")

#Force Processing on the Newest Frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

lock = threading.Lock() #Allows only one thread to access frame data at a time.
latest_frame = None #The most recent frame
camera_running = True

#Function for Reading Camera Frames
def read_camera():
    global latest_frame, camera_running #Global: Allows variables to be modified outside of their current scope.

    while camera_running:
        #Return values from .read()
        ret, frame = cap.read() #ret = Frame Retrieved, frame = The Image

        #If frame retrevial fails, try again after a delay.
        if not ret:
            time.sleep(0.01)
            continue

        #Acquires the lock, sets latest_frame to the frame acquired, and then automatically releases the lock.
        with lock:
            latest_frame = frame

#Creates and then starts a thread for the read_camera function.
#Threads are non-daemonic by default, meaning that Python won't shut down until the thread finishes.
#A daemonic thread will terminate upon Python shutdown.
reader = threading.Thread(target = read_camera, daemon = True)
reader.start()

#Infinite loop that continuously retrieves the newest frame.
try:
    while True:
        #Acquires the lock, sets latest_frame to a copy of the frame acquired, and then automatically releases the lock.
        with lock:
            frame = (
                latest_frame.copy()
                if latest_frame is not None #is not None - latest_frame has a value
                else None #Set frame to none.
            )

        #If frame retrevial fails, try again after a delay.
        if frame is None:
            time.sleep(0.005)
            continue

        #Performs YOLO Object Detection
        results = model.track(
            source = frame, #Uses frame as the image.
            tracker = "bytetrack.yaml", #Uses ByteTrack
            persist = True, #Preserves object identites while tracking.
            verbose = False,
            conf = 0.25 #Confidence Threshold
        )

        annotated_frame = results[0].plot() #results[0] is the results for the frame, and .plot() draws the boxes onto the image.
        cv2.imshow( #Shows the annotated frame to a window named "YOLO + ByteTrack Phone Camera"
            "YOLO + ByteTrack Phone Camera",
            annotated_frame
        )

        #Stops the program when q is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally: #Runs when the program leaves the try block.
    camera_running = False #Ends the background thread.
    reader.join(timeout = 1)
    cap.release() #Releases the camera stream.
    cv2.destroyAllWindows() #Closes all open windows.