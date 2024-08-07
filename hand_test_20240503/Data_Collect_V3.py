# -*- coding: utf-8 -*-
# 收集資料
import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


def compute_distances(face_landmarks,landmarks):
    distances = []

    # Define pairs for distance calculation
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8),
             (0, 9), (0, 10), (0, 11), (0, 12),
             (0, 13), (0, 14), (0, 15), (0, 16),
             (0, 17), (0, 18), (0, 19), (0, 20),
             (4, 8), (8, 12), (12, 16), (16, 20)]

    reference_pair = (0, 9)


    p_ref1 = np.array([landmarks.landmark[reference_pair[0]].x, landmarks.landmark[reference_pair[0]].y])
    p_ref2 = np.array([landmarks.landmark[reference_pair[1]].x, landmarks.landmark[reference_pair[1]].y])

   
    f_ref1 = np.array([face_landmarks[0].x, face_landmarks[0].y])
    f_ref2 = np.array([face_landmarks[1].x, face_landmarks[1].y]) 

    reference_distance = np.linalg.norm(p_ref1 - p_ref2)
    reference_face_distance = np.linalg.norm(f_ref1 - f_ref2)

    # for face_lm in face_landmarks:
    #     # Calculate the distance between each face landmark and the hand landmarks
    #     face_to_hands_X = (face_lm.x - landmarks.landmark[0].x) / reference_face_distance
    #     face_to_hands_Y = (face_lm.y - landmarks.landmark[0].y) / reference_face_distance
    #     # Append the calculated distances to the list
    #     distances.append(face_to_hands_X)
    #     distances.append(face_to_hands_Y)

    # for face_lm in face_landmarks:
    #     # Calculate the distance between each face landmark and the hand landmarks
    #     distance_x = face_lm.x - landmarks.landmark[0].x
    #     distance_y = face_lm.y - landmarks.landmark[0].y
    #     distance = np.sqrt(distance_x**2 + distance_y**2) / reference_face_distance
    #     # Append the calculated distance to the list
    #     distances.append(distance)

    # Calculate the distance between each face landmark and the hand landmarks
    face_to_hands_X = (face_landmarks[2].x - landmarks.landmark[0].x) / reference_face_distance
    face_to_hands_Y = (face_landmarks[2].y - landmarks.landmark[0].y) / reference_face_distance
    # Append the calculated distances to the list
    distances.append(face_to_hands_X)
    distances.append(face_to_hands_Y)

    # Calculate the distance between each face landmark and the hand landmarks
    distance_x = face_landmarks[2].x - landmarks.landmark[0].x
    distance_y = face_landmarks[2].y  - landmarks.landmark[0].y
    distance = np.sqrt(distance_x**2 + distance_y**2) / reference_face_distance
    # Append the calculated distance to the list
    distances.append(distance)

    for pair in pairs:
            p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
            p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
            distance = np.linalg.norm(p1 - p2) / reference_distance
            distances.append(distance)

 

    return distances

# Ask user for filename
filename = input("Please enter the filename for data: ")
# 重要～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～～下面這行記得改喔！！！
path="source_1"
save_path = "dataset/"+path+"/" + filename+".npy"
if not os.path.exists( f"dataset/{path}"):
    os.makedirs( f"dataset/{path}")

# Check if the 'dataset' directory exists, if not, create it
if not os.path.exists("dataset"):
    os.makedirs("dataset")

cap = cv2.VideoCapture(0)
data_collection = []

# Load existing data if the file exists
if os.path.exists(save_path):
    existing_data = np.load(save_path)
    data_collection.extend(existing_data.tolist())

collecting = False
start_time = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if not collecting:
            cv2.putText(frame, "Press SPACE to start data collection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            elapsed_time = int(time.time() - start_time)
            remaining_time = 15 - elapsed_time
            cv2.putText(frame, f"Time left: {remaining_time} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if elapsed_time >= 15:
                break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        results_face=face_detection.process(rgb_frame)
    

        # if results.multi_hand_landmarks and collecting and results_face.detections:
        #     for landmarks in results.multi_hand_landmarks:
        #         for detection_face in results_face.detections:
        #             face_landmarks = detection_face.location_data.relative_keypoints
        #             distances = compute_distances(face_landmarks,landmarks)
        #             data_collection.append(distances)
       
        if results.multi_hand_landmarks and collecting and results_face.detections:
            for landmarks in results.multi_hand_landmarks:
                for detection_face in results_face.detections:
                    face_landmarks = detection_face.location_data.relative_keypoints
                    distances = compute_distances(face_landmarks,landmarks)
                    data_collection.append(distances)
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    for face_lm in face_landmarks:
                        face_lm_x = int(face_lm.x * frame.shape[1])
                        face_lm_y = int(face_lm.y * frame.shape[0])
                        cv2.circle(frame, (face_lm_x, face_lm_y), 5, (0, 255, 0), -1)
                        
        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1)
    
        if key == 32 and not collecting:
            collecting = True
            start_time = time.time()
        elif key == ord('q'):
            break
except KeyboardInterrupt:
    print("中止程式")
cap.release()
cv2.destroyAllWindows()

# Convert the data_collection list to numpy array and save
np.save(save_path, np.array(data_collection))
print(f"Data saved to {save_path}")