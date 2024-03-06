import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize mediapipe Hands
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
datalist=[]
hands = mp_hands.Hands()
face_detection = mp_face.FaceMesh()

def compute_distances(landmarks):
    distances = []
    
    # Define pairs for distance calculation
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8),
             (0, 9), (0, 10), (0, 11), (0, 12),
             (0, 13), (0, 14), (0, 15), (0, 16),
             (0, 17), (0, 18), (0, 19), (0, 20),
             (4, 8), (8, 12), (12, 16), (16, 20)]

    reference_pair = (0, 9)#ladmark[1]
    p_ref1 = np.array([landmarks.landmark[reference_pair[0]].x, landmarks.landmark[reference_pair[0]].y])
    p_ref2 = np.array([landmarks.landmark[reference_pair[1]].x, landmarks.landmark[reference_pair[1]].y])
    reference_distance = np.linalg.norm(p_ref1 - p_ref2)#設定標準化的參數 亦手掌兩點間距離
    
    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        #標準化 避免因距離產生座標誤差
        distance = np.linalg.norm(p1 - p2) / reference_distance
        distances.append(distance)

    return distances

#臉部偵測標準化
def compute_distances_face(landmarks):
    distances = []
    
    # Define pairs for distance calculation
    pairs = [(0, 37), (0, 39), (0, 40), (0, 185),
             (0, 61), (0, 267), (0, 269), (0, 270),
             (0, 409), (0, 291), (0, 375), (0, 321),
             (0, 405), (0, 314), (0, 17), (0, 146),
             (0, 91), (0, 181), (0, 84), (80, 310),
             (88, 318) ]

    reference_pair = (4, 6)#ladmark[1]
    p_ref1 = np.array([landmarks.landmark[reference_pair[0]].x, landmarks.landmark[reference_pair[0]].y])
    p_ref2 = np.array([landmarks.landmark[reference_pair[1]].x, landmarks.landmark[reference_pair[1]].y])
    reference_distance = np.linalg.norm(p_ref1 - p_ref2)#設定標準化的參數 亦手掌兩點間距離
    
    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        #標準化 避免因距離產生座標誤差
        distance = np.linalg.norm(p1 - p2) / reference_distance
        distances.append(distance)

    return distances


# Ask user for filename
filename = input("Please enter the filename for data: ")
save_path = "dataset_V3/" + filename

# Check if the 'dataset' directory exists, if not, create it
if not os.path.exists("dataset_V3"):
    os.makedirs("dataset_V3")

cap = cv2.VideoCapture(0)
data_collection = []

collecting = False
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if not collecting:
        cv2.putText(frame, "Press SPACE to start data collection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        elapsed_time = int(time.time() - start_time)
        remaining_time = 10 - elapsed_time
        cv2.putText(frame, f"Time left: {remaining_time} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if elapsed_time >= 10:
            break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb_frame)
    results_face = face_detection.process(rgb_frame)
    
    if results_hands.multi_hand_landmarks and collecting:
        for landmarks in results_hands.multi_hand_landmarks:
            distances = compute_distances(landmarks)
            datalist=distances
    
    #以下新增.................................................
    if results_face.multi_face_landmarks and collecting:
        for landmarks in results_face.multi_face_landmarks:
            distances = compute_distances_face(landmarks)
            datalist=datalist+distances
            if len(datalist)==45:
               data_collection.append(datalist)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1)
    
    if key == 32 and not collecting:
        collecting = True
        start_time = time.time()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert the data_collection list to numpy array and save
np.save(save_path, np.array(data_collection))
print(f"Data saved to {save_path}")