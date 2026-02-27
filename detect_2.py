import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ==============================
# INITIALIZE MEDIAPIPE
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# DROWSINESS PARAMETERS
# ==============================
EAR_THRESHOLD = 0.23
CONSEC_FRAMES = 20
HEAD_TILT_THRESHOLD = 15  # degrees

closed_frames = 0
blink_count = 0
alarm_playing = False

# ==============================
# GET ALARM PATH (MP3 VERSION)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
alarm_path = os.path.join(BASE_DIR, "alarm.mp3")

# ==============================
# EAR FUNCTION
# ==============================
def calculate_EAR(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)

print("🚗 Advanced Driver Monitoring System Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # LEFT EYE LANDMARKS
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]

            left_eye = []
            right_eye = []

            for idx in left_eye_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append([x, y])

            for idx in right_eye_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append([x, y])

            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)

            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # DRAW EYES
            cv2.polylines(frame, [left_eye], True, (0,255,0), 1)
            cv2.polylines(frame, [right_eye], True, (0,255,0), 1)

            # ==============================
            # DROWSINESS LOGIC
            # ==============================
            if avg_EAR < EAR_THRESHOLD:
                closed_frames += 1
            else:
                if closed_frames >= 3:
                    blink_count += 1
                closed_frames = 0

            # Microsleep detection
            if closed_frames > CONSEC_FRAMES:
                cv2.putText(frame, "!!! DROWSINESS ALERT !!!",
                            (50,100),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            1.2,
                            (0,0,255),
                            3)

                if not alarm_playing:
                    try:
                        os.system(f"afplay '{alarm_path}' &")
                        alarm_playing = True
                    except Exception as e:
                        print("Alarm error:", e)
            else:
                alarm_playing = False

            # DISPLAY INFO
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.putText(frame, f"Blinks: {blink_count}", (30,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()