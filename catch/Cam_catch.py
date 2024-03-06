import cv2
import mediapipe as mp
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# #詢問使用者輸入文件名稱。
# filename = input("Please enter the filename for data: ")
# #構建保存文件的路徑。
# save_path = "dataset_V3/" + filename 
# 初始化視訊捕獲
cap = cv2.VideoCapture(0)

# 初始化 Mediapipe Hands 模型
mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_draw_hands = mp.solutions.drawing_utils
hand_lms_style = mp_draw_hands.DrawingSpec(color=(0, 0, 255), thickness=3)
hand_con_style = mp_draw_hands.DrawingSpec(color=(0, 255, 0), thickness=5)
pTime = 0
cTime = 0

# 初始化 Mediapipe FaceMesh 模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,       
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw_face_mesh = mp_draw_hands

# 開始主循環
while True:
    ret, img = cap.read()
    if not ret:
        print("無法接收影格")
        break

    # 手部處理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(img_rgb)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_draw_hands.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS, hand_lms_style, hand_con_style)

    # 臉部處理
    img_rgb_face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_mesh_results = face_mesh.process(img_rgb_face)
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            # 繪製網格
            mp_draw_face_mesh.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_hands.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))
            # 繪製輪廓
            mp_draw_face_mesh.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_hands.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
            # 繪製眼睛
            mp_draw_face_mesh.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw_hands.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

    # 顯示合併後的圖像
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS : {int(fps)}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(5) == ord('q'):
        break    # 按下 q 鍵停止

# 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
