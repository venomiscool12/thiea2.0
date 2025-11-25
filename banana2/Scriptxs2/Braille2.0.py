import os
import pyttsx3
import argparse
import cv2
from collections import defaultdict
import numpy as np
import sys
import time
from threading import Thread
import importlib.util


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='C:\Virtualtest\thiea2.0\banana2\yolov8n.pt', 
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")',
                    required=True)
parser.add_argument('--thresh', help='0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='640x480''), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')


args = parser.parse_args()




# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record


# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)


class VideoStream:
    """Camera object that controls video streaming from a webcam or video source.


    `source` accepts an integer device index (0,1,...) or a filename/URL.
    On Windows numeric device indices use the DirectShow backend for better USB camera support.
    """
    def __init__(self, source=0, resolution=(640,480), framerate=30):
        # Initialize the camera/video stream based on provided source
        # Accept strings like '0' (device index), int 0, or file/URL paths
        try:
            # Normalize source: if it's a digit string, convert to int
            if isinstance(source, str) and source.isdigit():
                src = int(source)
            else:
                src = source


            # On Windows prefer DirectShow for USB cameras when numeric index used
            if isinstance(src, int):
                self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                # file path or URL
                self.stream = cv2.VideoCapture(src)


            # Try to set common parameters (some backends ignore these)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3, resolution[0])
            ret = self.stream.set(4, resolution[1])
        except Exception as e:
            print('Error initializing video stream:', e)
            self.stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
           
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()


    # Variable to control when the camera is stopped
        self.stopped = False


    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self


    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return


            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()


    def read(self):
    # Return the most recent frame
        return self.frame


    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True


#speak function
def speak(text, engine):
    engine.say(text)
    engine.runAndWait()


#initialization of the TTS engine
def init_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)
    engine.setProperty('volume',1.0)
    return engine


# Get path to current working directory
CWD_PATH = os.getcwd()


# Use the model path provided by the user as the checkpoint path
PATH_TO_CKPT = model_path


# Determine label file path: try several sensible candidates and pick the first that exists.
# If none found, leave as None and continue without an explicit label map (model may provide names).
label_candidates = [
    os.path.splitext(model_path)[0] + ".txt",
    os.path.join(os.path.dirname(model_path), "labels.txt"),
    os.path.join(os.path.dirname(model_path), "label.txt"),
    os.path.join(CWD_PATH, "labels.txt"),
    os.path.join(CWD_PATH, "label"),
]
PATH_TO_LABELS = None
for p in label_candidates:
    if os.path.exists(p):
        PATH_TO_LABELS = p
        break


if PATH_TO_LABELS is None:
    print('Warning: label file not found. Continuing without explicit label map; class names may come from the model if available.')


# Have to do a weird fix for label map if using the COCO "starter model" from


# Initialize video stream using the `--source` argument (camera index, file, or URL)
videostream = VideoStream(source=img_source, resolution=(640,480),framerate=30).start()
time.sleep(1)
engine = init_engine()
# Parse resolution argument into width/height (fallback to 640x640)
try:
    if user_res:
        width, height = map(int, str(user_res).lower().split('x'))
    else:
        width, height = 640, 640
except Exception:
    width, height = 640, 640


# Minimum confidence threshold (float)
try:
    min_conf_threshold = float(min_thresh)
except Exception:
    min_conf_threshold = 0.5


# Frame rate calc initial value
frame_rate_calc = 0.0


# Try to load an Ultralytics YOLO model if available; otherwise continue without detection
model = None
try:
    from ultralytics import YOLO
    try:
        model = YOLO(PATH_TO_CKPT)
        print('Model loaded from', PATH_TO_CKPT)
    except Exception as e:
        print('Warning: failed to initialize YOLO model:', e)
        model = None
except Exception:
    print('Ultralytics YOLO not installed; running without detection.')
#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:


    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()
    frame = videostream.read()
    if frame is None:
        continue


    # Grab frame from video stream
    h, w, _ = frame.shape
    part_width = w // 3
   
    frame1, frame2, frame3, = frame[:, :part_width], frame[:, part_width:2 * part_width], frame[:, 2 * part_width:]
    frames = [frame1, frame2, frame3]


    # Acquire frame slices and preprocess each one
    for i, f in enumerate(frames):
        try:
            frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)
        except Exception as e:
            print('Frame preprocessing error:', e)
            continue


        # Run detection only if a model was successfully loaded
        if model is not None:
            try:
                res = model(frame_resized)[0]
                for box in res.boxes:
                    try:
                        confidence = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                    except Exception:
                        confidence = float(box.conf)


                    if confidence < min_conf_threshold:
                        continue


                    try:
                        cls_id = int(box.cls[0]) if hasattr(box.cls, '__len__') else int(box.cls)
                    except Exception:
                        cls_id = int(box.cls)


                    object_name = str(cls_id)
                    if hasattr(model, 'names') and model.names is not None:
                        try:
                            object_name = model.names[cls_id]
                        except Exception:
                            pass


                    try:
                        coords = box.xyxy[0] if hasattr(box.xyxy, '__len__') else box.xyxy
                        xmin, ymin, xmax, ymax = map(int, coords)
                    except Exception:
                        continue


                    # Draw bounding box on the slice
                    cv2.rectangle(f, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)


                    # Label text
                    label = f"{object_name}: {int(confidence * 100)}%"
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(
                        f,
                        (xmin, ymin - label_size[1] - 10),
                        (xmin + label_size[0], ymin + baseline - 10),
                        (255, 255, 255),
                        cv2.FILLED
                    )
                    cv2.putText(f, label, (xmin, ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


                    # Speak object if confidence above 0.95
                    if 0.95 <= confidence <= 1.0:
                        speak(f"A {object_name} has been detected", engine)


                    # Navigation guidance based on global x coordinate
                    global_x = xmin + i * part_width
                    if global_x < part_width:
                        navigation_text = f"{object_name} detected on the left"
                    elif global_x < 2 * part_width:
                        navigation_text = f"{object_name} detected in the center"
                    else:
                        navigation_text = f"{object_name} detected on the right"
                    speak(navigation_text, engine)
            except Exception as e:
                print('Detection error:', e)


    def redundant_object():
        # Control variables
        object_counts = defaultdict(int)
        cooldown_timers = {}


        # Cooldown duration (seconds)
        COOLDOWN_TIME = 15


        # Run detection loop
        for detected_object in combined_frame():
            current_time = time.time()


        # Check cooldown: skip if still in cooldown
        if detected_object in cooldown_timers and current_time < cooldown_timers[detected_object]:
            print(f"Skipping '{detected_object}' (cooldown active)")
           
        # Update detection count
        object_counts[detected_object] += 1


        # If object detected 3 times, trigger cooldown
        if object_counts[detected_object] >= 3:
            print(f"'{detected_object}' detected 3 times — entering 15s cooldown.")
            cooldown_timers[detected_object] = current_time + COOLDOWN_TIME
            object_counts[detected_object] = 0  # reset count


    #combine the frames after drawing all bounding boxes and labels
    combined_frame = np.hstack((frame1, frame2, frame3))
   
    # Draw framerate in corner of frame
    cv2.putText(combined_frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
   
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', combined_frame)


    # Get current tick
    t2 = cv2.getTickCount()


    # Calculate elapsed time in seconds
    time_elapsed = (t2 - t1) / cv2.getTickFrequency()


    # Calculate framerate
    frame_rate_calc = 1 / time_elapsed


    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break


# Clean up
cv2.destroyAllWindows()
videostream.stop()


#apllpelpelpwleaplepawle