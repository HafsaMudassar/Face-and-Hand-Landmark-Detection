import cv2
import mediapipe as mp
import os
import json
import time

# ----------------------- Setup -----------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create folders if not exist
os.makedirs("data", exist_ok=True)
os.makedirs("landmarks", exist_ok=True)

# Initialize Holistic model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

# FPS calculation
prev_time = 0

# Frame counter for saving images
frame_counter = 0

print("Press 'C' to capture image, 'S' to save landmarks, 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (800, 600))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Draw landmarks
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(255,0,255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"{int(fps)} FPS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Craniofacial Landmark Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # ----------------- Controls -----------------
    if key == ord('c'):
        # Capture image
        img_name = f"data/frame_{frame_counter}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")

    if key == ord('s'):
        # Save landmarks
        landmarks = {
            "face": [],
            "left_hand": [],
            "right_hand": []
        }

        # Face
        if results.face_landmarks:
            landmarks['face'] = [{"id": i, "x": lm.x, "y": lm.y, "z": lm.z} 
                                 for i, lm in enumerate(results.face_landmarks.landmark)]
        # Left hand
        if results.left_hand_landmarks:
            landmarks['left_hand'] = [{"id": i, "x": lm.x, "y": lm.y, "z": lm.z} 
                                      for i, lm in enumerate(results.left_hand_landmarks.landmark)]
        # Right hand
        if results.right_hand_landmarks:
            landmarks['right_hand'] = [{"id": i, "x": lm.x, "y": lm.y, "z": lm.z} 
                                       for i, lm in enumerate(results.right_hand_landmarks.landmark)]

        # Check if nothing detected
        if not any([landmarks['face'], landmarks['left_hand'], landmarks['right_hand']]):
            print("No landmarks detected for this frame.")
        else:
            lm_name = f"landmarks/frame_{frame_counter}.json"
            with open(lm_name, "w") as f:
                json.dump(landmarks, f, indent=4)
            print(f"Saved landmarks to {lm_name}")

        frame_counter += 1

    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
holistic.close()
