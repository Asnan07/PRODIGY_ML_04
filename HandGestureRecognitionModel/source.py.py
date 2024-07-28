import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
try:
    model = load_model('mp_hand_gesture')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load class names
try:
    with open('gesture.names', 'r') as f:
        classNames = f.read().split('\n')
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()
print(classNames)

def get_landmarks(result, x, y):
    landmarks = []
    for handslms in result.multi_hand_landmarks:
        for lm in handslms.landmark:
            lmx = int(lm.x * x)
            lmy = int(lm.y * y)
            landmarks.append([lmx, lmy])
        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
    return landmarks

def predict_gesture(landmarks):
    if len(landmarks) == 0:
        return ''
    landmarks = np.array(landmarks).flatten()  # Flattening to match model input shape
    landmarks = landmarks / np.linalg.norm(landmarks)  # Normalization
    prediction = model.predict([landmarks])
    classID = np.argmax(prediction)
    return classNames[classID]

# Initialize the webcam
cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    x, y, _ = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Process result and predict gesture
    className = ''
    if result.multi_hand_landmarks:
        landmarks = get_landmarks(result, x, y)
        className = predict_gesture(landmarks)

    # Show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
