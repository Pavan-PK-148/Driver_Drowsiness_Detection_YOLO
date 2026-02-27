import cv2
import torch
import time
import simpleaudio as sa
from collections import deque

# ==========================
# LOAD MODEL
# ==========================
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.5

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# ==========================
# DROWSINESS PARAMETERS
# ==========================
FRAME_THRESHOLD = 20          # Closed eye frames
BLINK_TIME_THRESHOLD = 0.8
HEAD_TILT_THRESHOLD = 1.2

closed_frame_count = 0
blink_start_time = None
blink_count = 0
tilt_score = 0
play_obj = None

print("🚗 Advanced Driver Monitoring System Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    sleepy_detected = False

    for (x, y, w, h) in faces:

        # HEAD TILT DETECTION
        aspect_ratio = h / w
        if aspect_ratio > HEAD_TILT_THRESHOLD:
            tilt_score += 1
        else:
            tilt_score = max(0, tilt_score - 1)

        roi_y1 = y + int(h * 0.2)
        roi_y2 = y + int(h * 0.55)
        eye_region = frame[roi_y1:roi_y2, x:x+w]

        if eye_region.size > 0:
            results = model(eye_region)
            detections = results.xyxy[0]

            for *box, conf, cls in detections:
                label = model.names[int(cls)]

                if conf > 0.55 and label in ["sleepy", "closed_eye", "closed"]:
                    sleepy_detected = True

    # ==========================
    # BLINK & SLEEP LOGIC
    # ==========================
    if sleepy_detected:
        closed_frame_count += 1

        if blink_start_time is None:
            blink_start_time = time.time()

        blink_duration = time.time() - blink_start_time

        if blink_duration > BLINK_TIME_THRESHOLD:
            cv2.putText(frame, "MICROSLEEP DETECTED", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    else:
        if closed_frame_count > 2:
            blink_count += 1

        closed_frame_count = 0
        blink_start_time = None

    # ==========================
    # FINAL DROWSINESS DECISION
    # ==========================
    drowsiness_score = closed_frame_count + tilt_score

    if drowsiness_score > FRAME_THRESHOLD:
        cv2.putText(frame, "!!! DROWSINESS ALERT !!!",
                    (80, 120),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1.5,
                    (0, 0, 255),
                    4)

        if play_obj is None or not play_obj.is_playing():
            try:
                wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
                play_obj = wave_obj.play()
            except:
                print("Alarm error")

    cv2.putText(frame, f"Blinks: {blink_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Advanced Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()