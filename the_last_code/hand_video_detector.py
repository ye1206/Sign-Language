import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from django.core.cache import cache
from .new_put_text import cv2_chinese_text
from django.conf import settings
from django.core.signals import request_finished#
from django.dispatch import receiver#
import os
import logging
import json
import gc
from PIL import ImageFont, ImageDraw, Image    # 載入 PIL 相關函式庫
fontpath = 'GenSekiGothic-R.ttc'          # 設定字型路徑
logger = logging.getLogger(__name__)
font = ImageFont.truetype(fontpath, 20)  

def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# PIL RGB 影像轉 OpenCV BGR 影像
def pil_to_cv2(image):
    return cv2.flip(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), 1)

# 使用 PIL 在影像上繪製文字
def draw_text(img, text, position, color=(0, 255, 0)):
    img=cv2.flip(img, 1)
    pil_img = cv2_to_pil(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return pil_to_cv2(pil_img)

def load_svm_models(topic, current_question_model):
    cache_key = f"model_{topic}_{current_question_model}"
    cached_model = cache.get(cache_key)

    if cached_model is None:
        try:
            model_filename = os.path.join(
                settings.BASE_DIR, f"script/model/topic_{topic}/question_{current_question_model}/svm_model_topic.pkl")
            scaler_filename = os.path.join(
                settings.BASE_DIR, f"script/model/topic_{topic}/question_{current_question_model}/scaler_topic.pkl")
            # Load labels file
            label_file = os.path.join(
                settings.BASE_DIR, f"script/model/topic_{topic}/question_{current_question_model}/labels_topic.txt")

            # Load SVM Model and scaler
            clf = joblib.load(model_filename)
            scaler = joblib.load(scaler_filename)

            with open(label_file, 'r', encoding="UTF-8") as f:
                labels = f.readlines()
            labels = [label.strip() for label in labels]

            # Package the model, scaler, and labels into a tuple
            cached_model = (clf, scaler, labels)
            # Store the model in cache
            cache.set(cache_key, cached_model, timeout=None)
            print(
                f"Model loaded for topic {topic}, question {current_question_model}")
        except FileNotFoundError as e:
            print(
                f"File not found: {e}")
            return None, None, None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None
    return cached_model

def format_answers_list(answer_list, group_size=3):
    formatted_text = ""
    for i in range(0, len(answer_list), group_size):
        group = answer_list[i:i + group_size]  # 每三個元素作為一組
        formatted_text += ", ".join(group) + "\n"  # 將這一組元素用逗號連接，並換行
    return formatted_text

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
    # 9/25
hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)

face = mp_face.FaceDetection(min_detection_confidence=0.5)

question_timer_limit = 60
question_timer_start = time.time()
current_question_index = 0
current_question_index_display = 1
current_answer_list_index = 0
max_wrong_answers = 3
current_wrong_answers = 0
answer_list = []
correct_questions_no = []
wrong_questions_no = []
# this method is used to deduce the hand_video method
# plz refer to the hand_video method accordingly


def compute_distances(face_landmarks, landmarks):
    distances = []

    # Define pairs for distance calculation
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8),
             (0, 9), (0, 10), (0, 11), (0, 12),
             (0, 13), (0, 14), (0, 15), (0, 16),
             (0, 17), (0, 18), (0, 19), (0, 20),
             (4, 8), (8, 12), (12, 16), (16, 20)]

    reference_pair = (0, 9)

    p_ref1 = np.array([landmarks.landmark[reference_pair[0]].x,
                       landmarks.landmark[reference_pair[0]].y])
    p_ref2 = np.array([landmarks.landmark[reference_pair[1]].x,
                       landmarks.landmark[reference_pair[1]].y])

    f_ref1 = np.array([face_landmarks[0].x, face_landmarks[0].y])
    f_ref2 = np.array([face_landmarks[1].x, face_landmarks[1].y])

    reference_distance = np.linalg.norm(p_ref1 - p_ref2)
    reference_face_distance = np.linalg.norm(f_ref1 - f_ref2)

    # Calculate the distance between each face landmark and the hand landmarks
    face_to_hands_X = (
        face_landmarks[2].x - landmarks.landmark[0].x) / reference_face_distance
    face_to_hands_Y = (
        face_landmarks[2].y - landmarks.landmark[0].y) / reference_face_distance
    distances.append(face_to_hands_X)
    distances.append(face_to_hands_Y)

    distance_x = face_landmarks[2].x - landmarks.landmark[0].x
    distance_y = face_landmarks[2].y - landmarks.landmark[0].y
    distance = np.sqrt(distance_x**2 + distance_y**2) / reference_face_distance
    distances.append(distance)

    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x,
                       landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x,
                       landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2) / reference_distance
        distances.append(distance)

    # ensure the number of features returned is 27
    if len(distances) != 27:
        print(f"Unexpected number of features: {len(distances)}")
    return distances

def cleanup():
    global question_timer_limit, question_timer_start, current_question_index, current_question_index_display
    global current_answer_list_index, max_wrong_answers, current_wrong_answers, answer_list, correct_questions_no, wrong_questions_no
    
    question_timer_limit = 60
    question_timer_start = time.time()
    current_question_index = 0
    current_question_index_display = 1
    current_answer_list_index = 0
    max_wrong_answers = 3
    current_wrong_answers = 0
    answer_list = []
    correct_questions_no = []
    wrong_questions_no = []

def cleanup_view():
    cleanup()  # 執行清理操作

@receiver(request_finished)     
def cleanup_resources(sender, **kwargs):
    # 在每個請求結束時執行的清理操作
    cleanup()

def hand_video(flag, frame, topic):
    # Define the list of questions
    # 主題一：上課日常1
    global question_timer_start,answer_list,current_question_index,current_question_index_display,current_answer_list_index,max_wrong_answers,current_wrong_answers,answer_list,correct_questions_no,wrong_questions_no
    if topic == '1':
        questions = [
            {"考試過了嗎": [["考試1_left", "考試2_left", "及格1_left", "及格2_left", "empty", "empty"], 
                ["考試1_right", "考試2_right", "及格1_right", "及格2_right", "有沒有1_right", "有沒有2_right"],
                ["考試1", "考試2", "及格1", "及格2", "有沒有1", "有沒有2"]]},
            {"今天上課的內容好難": [["今天1_left", "今天2_left", "empty", "empty", "讀書_left", "讀書_left", "empty", "empty", "empty"], [
                "今天1_right", "今天2_right", "去1_right", "去2_right", "讀書1_right", "讀書2_right", "這個_right", "難1_right", "難2_right"],
                ["今天1", "今天2", "去1", "去2", "讀書1", "讀書2", "這個","難1","難2"]]},
            {"你知道這怎麼解嗎？": [["empty", "empty", "會不會_left", "會不會_left"], [
                "這個_right", "你_right", "會不會1_right", "會不會2_right"],
                ["這個", "你", "會不會1", "會不會2"]]},
            {"作業寫了嗎？": [["作業1_left", "作業2_left", "empty", "empty", "empty", "empty"], [
                "作業1_right", "作業2_right", "寫1_right", "寫2_right", "有沒有1_right", "有沒有2_right"],
                ["作業1", "作業2", "寫1", "寫2", "有沒有1", "有沒有2"]]},
            {"筆記你有紀錄到嗎？": [["empty", "筆記_left", "筆記_left", "筆記_left", "筆記_left", "empty", "empty", "empty", "empty"], [
                "你_right", "筆記1_right", "筆記2_right", "筆記3_right", "筆記4_right", "寫1_right", "寫2_right", "有沒有1_right", "有沒有2_right"],
                ["你", "筆記1", "筆記2", "筆記3", "筆記4", "寫1","寫2","有沒有1","有沒有2"]]},
            {"下禮拜要期末考": [["empty", "empty", "要1_left", "要2_left", "考試1_left", "考試2_left", "末_left"], [
                "下禮拜1_right", "下禮拜2_right", "要1_right", "要2_right", "考試1_right", "考試2_right", "末_right"],
                ["下禮拜1", "下禮拜2", "要1", "要2", "考試1", "考試2", "末"]]},
            {"老師剛才上到哪裡了？": [["empty", "empty", "empty", "empty", "教_left", "教_left", "empty", "empty", "empty", "empty"], [
                "剛剛1_right", "剛剛2_right", "老師1_right", "老師2_right", "教1_right", "教2_right", "什麼1_right", "什麼2_right", "地方1_right", "地方2_right"],
                ["剛剛1", "剛剛2", "老師1", "老師2", "教1", "教2", "什麼1", "什麼2", "地方1", "地方2"]]},
            {"期中考我考得不太好": [["empty", "考試1_left", "考試2_left", "中_left", "empty"], [
                "我_right", "考試1_right", "考試2_right", "中_right", "不好_right"],
                ["我", "考試1", "考試2", "中", "不好"]]},
            {"今天的課好無聊": [["今天1_left", "今天2_left", "empty", "empty", "讀書_left", "讀書_left", "無聊1_left", "無聊2_left"], [
                "今天1_right", "今天2_right", "去1_right", "去2_right", "讀書1_right", "讀書2_right", "無聊1_right", "無聊2_right"],
                ["今天1", "今天2", "去1", "去2", "讀書1", "讀書2", "無聊1", "無聊2"]]},
            {"這堂課的期末形式是報告": [["empty", "empty", "empty", "讀書_left", "讀書_left", "末_left", "要1_left", "要2_left", "讀書_left", "讀書_left"], [
                "這個_right", "去1_right", "去2_right", "讀書1_right", "讀書2_right", "末_right", "要1_right", "要2_right", "報告1_right", "報告2_right"],
                ["這個", "去1", "去2", "讀書1", "讀書2", "末", "要1", "要2", "報告1", "報告2"]]}
        ]
    # 主題二：日常問候
    if topic == '2':
        questions = [
            {"同學早安": [["同學1_left", "同學2_left", "同學3_left", "同學4_left", "empty", "empty", "安1_left", "安2_left"], [
                "同學1_right", "同學2_right", "同學3_right", "同學4_right", "早1_right", "早2_right", "安1_right", "安2_right"]]},
            {"你看起來很累": [["empty", "empty", "empty", "empty", "empty"], [
                "看起來1_right", "看起來2_right", "你_right", "累1_right", "累2_right"]]},
            {"今天天氣好熱": [["今天1_left", "今天2_left", "empty", "empty", "empty", "empty", "熱1_left", "熱2_left"], [
                "今天1_right", "今天2_right", "天氣1_right", "天氣2_right", "天氣3_right", "天氣4_right", "熱1_right", "熱2_right"]]},
            {"你要回家嗎？": [["empty", "回_left", "家_left", "要1_left", "要2_left", "不要1_left", "不要2_left"], [
                "你_right", "回_right", "家_right", "要1_right", "要2_right", "不要1_right", "不要2_right"]]},
            {"很高興見到你": [["empty", "開心1_left", "開心2_left", "見到_left"], [
                "你我_right", "開心1_right", "開心2_right", "見到_right"]]},
            {"今天天氣不錯": [["今天1_left", "今天2_left", "empty", "empty", "empty", "empty", "empty", "empty"], [
                "今天1_right", "今天2_right", "天氣1_right", "天氣2_right", "天氣3_right", "天氣4_right", "不錯1_right", "不錯2_right"]]},
            {"最近忙嗎": [["最近_left", "忙1_left", "忙2_left", "empty"],
                      ["最近_right", "忙1_right", "忙2_right", "你_right"]]},
            {"好久不見": [["見1_left", "見2_left", "empty", "empty"], [
                "見1_right", "見2_right", "很久沒有1_right", "很久沒有2_right"]]},
            {"你今天好嗎?": [["empty", "今天1_left", "今天2_left", "empty", "empty"], [
                "你_right", "今天1_right", "今天2_right", "好_right", "不好_right"]]},
            {"午安你好": [["empty", "安1_left", "安2_left", "empty", "empty"], [
                "午_right", "安1_right", "安2_right", "你_right", "好(打招呼)_right"]]}
        ]
    # 主題三：吃飯
    elif topic == '3':
        questions = [
            {"你吃飯了嗎": [["empty", "吃飯_left", "吃飯_left", "empty", "empty"], [
                "你_right", "吃飯1_right", "吃飯2_right", "有沒有1_right", "有沒有2_right"]]},
            {"要一起吃飯嗎": [["一起1_left", "一起2_left", "吃飯_left", "吃飯_left", "empty", "empty", "要1_left", "要2_left", "不要1_left", "不要2_left"], [
                "一起1_right", "一起2_right", "吃飯1_right", "吃飯2_right", "去1_right", "去2_right", "要1_right", "要2_right", "不要1_right", "不要2_right"]]},
            {"學餐好吃嗎": [["學校_left", "學校_left", "學校_left", "吃飯_left", "吃飯_left", "empty", "empty", "empty", "empty", "empty", "empty", "empty"], [
                "學校1_right", "學校2_right", "學校1_right", "吃飯1_right", "吃飯2_right", "地方1_right", "地方2_right", "好吃1_right", "好吃2_right", "好吃3_right", "有沒有1_right", "有沒有2_right"]]},
            {"這飯好吃嗎": [["empty", "empty", "empty", "empty", "empty"], [
                "這個_right", "吃飯1_right", "吃飯2_right", "好_right", "不好_right"]]},
            {"學餐真的好貴": [["學校_left", "學校_left", "學校_left", "吃飯_left", "吃飯_left", "empty", "empty", "貴_left", "貴_left", "empty", "empty"], [
                "學校1_right", "學校2_right", "學校1_right", "吃飯1_right", "吃飯2_right", "地方1_right", "地方2_right", "貴1_right", "貴2_right", "很1_right", "很2_right"]]},
            {"你是吃素的嗎": [["empty", "吃飯_left", "吃飯_left", "素_left"],
                        ["你_right", "吃飯1_right", "吃飯2_right", "素_right"]]},
            {"我都自己在家煮飯": [["empty", "家_left", "裡_left", "裡_left", "煮_left", "煮_left", "煮_left", "煮_left", "菜1_left", "菜2_left"], [
                "我_right", "家_right", "裡1_right", "裡2_right", "煮1_right", "煮2_right", "煮1_right", "煮2_right", "菜1_right", "菜2_right"]]},
            {"我幾乎每天都外食": [["empty", "常常1_left", "常常2_left", "常常1_left", "每天_left", "每天_left", "外面_left", "外面_left", "吃飯_left", "吃飯_left"], [
                "我_right", "常常1_right", "常常2_right", "常常1_right", "每天1_right", "每天2_right", "外面1_right", "外面2_right", "吃飯1_right", "吃飯2_right"]]},
            {"你晚餐想吃什麼": [["empty", "晚上1_left", "晚上2_left", "empty", "empty", "empty", "empty", "empty", "empty"], [
                "你_right", "晚上1_right", "晚上2_right", "想1_right", "想2_right", "吃飯1_right", "吃飯2_right", "什麼1_right", "什麼2_right"]]},
            {"我還沒吃早餐": [["empty", "empty", "empty", "empty", "empty", "吃飯_left", "吃飯_left", "empty", "empty"], [
                "我_right", "吃飯1_right", "吃飯2_right", "早餐(早)1_right", "早餐(早)2_right", "吃飯1_right", "吃飯2_right", "沒有1_right", "沒有2_right"]]}
        ]
    # 主題四：假日休閒
    elif topic == '4':
        questions = [
            {"一起去運動吧!": [["一起1_left", "一起2_left", "運動1_left", "運動2_left", "運動1_left", "運動2_left", "empty", "empty", "要1_left", "要2_left", "不要1_left", "不要2_left"], [
                "一起1_right", "一起2_right", "運動1_right", "運動2_right", "運動1_right", "運動2_right", "去1_right", "去2_right", "要1_right", "要2_right", "不要1_right", "不要2_right"]]},
            {"要一起去學校嗎?": [["一起1_left", "一起2_left", "empty", "empty", "學校_left", "學校_left", "學校_left", "要1_left", "要2_left", "不要1_left", "不要2_left"], [
                "一起1_right", "一起2_right", "去1_right", "去2_right", "學校1_right", "學校2_right", "學校1_right", "要1_right", "要2_right", "不要1_right", "不要2_right"]]},
            {"那部電影好看嗎?": [["empty", "電影1_left", "電影2_left", "empty", "empty", "empty", "empty"], [
                "這個_right", "電影1_right", "電影2_right", "看起來1_right", "看起來2_right", "好_right", "不好_right"]]},
            {"我假日要去打工": [["放假1_left", "empty", "empty", "要1_left", "要2_left", "empty", "empty", "工作_left", "工作_left", "工作_left", "工作_left"], [
                "empty", "放假2_right", "我_right", "要1_right", "要2_right", "去1_right", "去2_right", "工作1_right", "工作2_right", "工作1_right", "工作2_right"]]},
            {"你假日有空嗎?": [["放假1_left", "empty", "empty", "時間_left", "時間_left", "empty", "empty"], [
                "empty", "放假2_right", "你_right", "時間1_right", "時間2_right", "有沒有1_right", "有沒有2_right"]]},
            {"你有參加學校的那個活動嗎?": [["empty", "參加1_left", "參加2_left", "學校_left", "學校_left", "學校_left", "empty", "活動1_left", "活動2_left", "empty", "empty"], [
                "你_right", "參加1_right", "參加2_right", "學校1_right", "學校2_right", "學校1_right", "那個人_right", "活動1_right", "活動2_right", "有沒有1_right", "有沒有2_right"]]},
            {"學校今年有舉辦健行": [["今天1_left", "今天2_left", "今年3_left", "今年3_left", "今年3_left", "學校_left", "學校_left", "學校_left", "empty", "empty", "empty", "empty", "empty", "empty"], [
                "今天1_right", "今天2_right", "今年3_right", "今年4_right", "今年3_right", "學校1_right", "學校2_right", "學校1_right", "有1_right", "有2_right", "健行1_right", "健行2_right", "健行1_right", "健行2_right"]]},
            {"我等一下要去運動": [["empty", "empty", "empty", "empty", "empty", "運動1_left", "運動2_left", "運動1_left", "運動2_left"], [
                "我_right", "等一下1_right", "等一下2_right", "去1_right", "去2_right", "運動1_right", "運動2_right", "運動1_right", "運動2_right"]]},
            {"我們去划船": [["empty", "empty", "empty", "划船1_left", "划船2_left"], [
                "你我_right", "去1_right", "去2_right", "划船1_right", "划船2_right"]]},
            {"我會參加運動會": [["empty", "會不會_left", "參加1_left", "參加2_left", "運動1_left", "運動2_left", "empty"], [
                "我_right", "會不會1_right", "參加1_right", "參加2_right", "運動1_right", "運動2_right", "會_right"]]}
        ]
    # 主題五：人際關係
    elif topic == '5':
        questions = [
            {"你認識那個人嗎?": [["empty", "empty", "empty", "empty", "empty"], [
                "那個人_right", "你_right", "知道1_right", "知道2_right", "嗎_right"]]},
            {"他跟我是同一個小組的": [["empty", "一起1_left", "一起2_left", "工作_left", "工作_left", "組1_left", "組2_left"], [
                "他我_right", "一起1_right", "一起2_right", "工作1_right", "工作2_right", "組1_right", "組2_right"]]},
            {"那個人好帥": [["empty", "empty", "empty", "empty", "empty"], [
                "那個人_right", "看起來1_right", "看起來2_right", "不錯1_right", "不錯2_right"]]},
            {"你要跟我一組嗎?": [["empty", "empty", "一起1_left", "一起2_left", "要1_left", "要2_left", "不要1_left", "不要2_left"], [
                "你_right", "我_right", "一起1_right", "一起2_right", "要1_right", "要2_right", "不要1_right", "不要2_right"]]},
            {"他們倆之間有發生什麼事嗎?": [["empty", "發生1_left", "發生2_left", "發生3_left", "empty", "empty"], [
                "他們倆_right", "發生1_right", "發生2_right", "發生3_right", "什麼1_right", "什麼2_right"]]},
            {"他們在一起了": [["empty", "empty", "變2_left", "情侶1_left", "情侶2_left", "empty", "empty"], [
                "他們倆_right", "變1_right", "empty", "情侶1_right", "情侶2_right", "情侶3_right", "情侶4_right"]]},
            {"我跟你說一則八卦": [["告訴你_left", "告訴你_left", "empty", "八卦_left"], [
                "告訴你1_right", "告訴你2_right", "八卦1_right", "八卦2_right"]]},
            {"我們之間有誤解": [["empty", "empty", "empty", "誤會1_left", "誤會2_left", "empty", "empty"], [
                "我們1_right", "我們2_right", "我們3_right", "誤會1_right", "誤會2_right", "有1_right", "有2_right"]]},
            {"我討厭他": [["empty", "empty", "empty", "empty"], [
                "那個人_right", "我_right", "討厭1_right", "討厭2_right"]]},
            {"我喜歡他": [["empty", "empty", "empty", "empty"], [
                "那個人_right", "我_right", "喜歡1_right", "喜歡2_right"]]}
        ]

    current_question_model = 0
    # Load the SVM model and scaler file
    # model_filename = f"model/topic_{topic}/question_{current_question_model}/svm_model_topic.pkl"
    # scaler_filename = f"model/topic_{topic}/question_{current_question_model}/scaler_topic.pkl"
    # # Load labels file
    # label_file = f"model/topic_{topic}/question_{current_question_model}/labels_topic.txt"
    # print('model_filename:', model_filename)
    # print('scaler_filename:', scaler_filename)
    # print('label_file:', label_file)
    
#9/20 這段改一下 如果有了就不加載試試
    clf, scaler, labels = load_svm_models(topic, current_question_model)

    # 檢查是否成功加載模型和縮放器
    if clf is None or scaler is None or labels is None:
        print("Failed to load model or scaler. (The First Time)")
        return frame

    # For static images:
    # parameters for the detector
    
    # Initialize variables
    
    num_questions = len(questions)


    # Maximum number of wrong answers allowed


    # Time delay for switching to the next question after correct answer (in seconds)
    switch_delay = 2
    switch_timer_started = False
    last_switch_time = time.time()

    # Question timer setup


    # Alert timer setup
    alert_timer_limit = 3
    alert_timer_start = time.time()
    alert_message = ""

    # flip it along y axis
    image = cv2.flip(frame, 1)
    image = cv2.flip(image, 1)



    # color format conversion
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_hand = hands.process(rgb_frame)
    results_face = face.process(rgb_frame)
    display_text = ''
    Answers_text = ''
    
    key, values = list(questions[current_question_index].items())[0]
    value_left, value_right, answer_already = values

    print("Key:", key)
    print("Value Left:", value_left)
    print("Value Right:", value_right)
    print("Answer Already:", answer_already)

    # if not results_hand.multi_hand_landmarks and not results_face.detections:
    #     hands.close()
    #     face.close()
    #     return frame

    answered_all_questions = False

    if (current_question_index != current_question_model):
        clf, scaler, labels = load_svm_models(topic, current_question_index)
        current_question_model = current_question_index
        if clf is None or scaler is None or labels is None:
            print("Failed to load model or scaler. (The Second Time)")
            return frame

    # check if the model and scaler are loaded successfully
    if clf is None or scaler is None or labels is None:
        print("Failed to load model or scaler.")
        return frame

    if results_hand.multi_hand_landmarks and results_face.detections:
        left_hand_landmarks = None
        right_hand_landmarks = None

        for index, landmarks in enumerate(results_hand.multi_hand_landmarks):
            hand_label = "Right" if results_hand.multi_handedness[
                index].classification[0].label == "Left" else "Left"
            if hand_label == "Left":
                left_hand_landmarks = landmarks
            else:
                right_hand_landmarks = landmarks

        if left_hand_landmarks and right_hand_landmarks:  # 這塊要再拓寬else 只有單手的情況
            for detection_face in results_face.detections:
                face_landmarks = detection_face.location_data.relative_keypoints

                distances_left = compute_distances(
                    face_landmarks, left_hand_landmarks)
                distances_right = compute_distances(
                    face_landmarks, right_hand_landmarks)

                # check if the scaler is None
                if scaler is None:
                    print("Scaler is None, cannot transform distances.")
                    return frame

                if len(distances_left) == 27 and len(distances_right) == 27:
                    distances_left = scaler.transform([distances_left])
                    distances_right = scaler.transform([distances_right])
                else:
                    print(
                        f"Unexpected number of features: {len(distances_left)}, {len(distances_right)} two hand")
                    return frame

                prediction_left = clf.predict(distances_left)
                prediction_right = clf.predict(distances_right)

                confidence_left = np.max(clf.predict_proba(distances_left))
                confidence_right = np.max(clf.predict_proba(distances_right))

                label_left = labels[prediction_left[0]]
                label_right = labels[prediction_right[0]]

                if confidence_left >= 0.70 and confidence_right >= 0.70:
                    if current_answer_list_index < len(value_left):
                        expected_label_left = value_left[current_answer_list_index]
                        expected_label_right = value_right[current_answer_list_index]

                        if label_left == expected_label_left and label_right == expected_label_right:
                            if not answered_all_questions:
                                answer_list.append(answer_already[current_answer_list_index])
                                current_answer_list_index += 1
                        elif current_answer_list_index == len(value_right) and answer_list != value_left:
                            if not answered_all_questions:
                                answer_list = []
                                current_answer_list_index = 0
                    elif current_answer_list_index > len(value_left):
                        answer_list = []
                        current_answer_list_index = 0
                display_text = f"Left Hand: {label_left}, Right Hand: {label_right}"
                Answers_text = f"Answers:{format_answers_list(answer_list)}"
        else:
            if left_hand_landmarks or right_hand_landmarks:
                for detection_face in results_face.detections:
                    face_landmarks = detection_face.location_data.relative_keypoints
                    if left_hand_landmarks:
                        distances = compute_distances(
                            face_landmarks, left_hand_landmarks)
                        print(f"distances: {distances}")

                        if len(distances) == 27:
                            distances = scaler.transform([distances])
                        else:
                            print(
                                f"Unexpected number of features: {len(distances)} one hand")
                            return frame
                        # distances = scaler.transform([distances])
                        prediction = clf.predict(distances)
                        confidence = np.max(clf.predict_proba(distances))
                        label_left = labels[prediction[0]]
                        label_right = "empty"  # 如果只有左手，右手标签为 "空"
                    else:
                        distances = compute_distances(
                            face_landmarks, right_hand_landmarks)
                        print(f"distances: {distances}")

                        if len(distances) == 27:
                            distances = scaler.transform([distances])
                        else:
                            print(
                                f"Unexpected number of features: {len(distances)} one hand")
                            return frame
                        # distances = scaler.transform([distances])
                        prediction = clf.predict(distances)
                        confidence = np.max(clf.predict_proba(distances))
                        label_left = "empty"  # 如果只有右手，左手标签为 "空"
                        label_right = labels[prediction[0]]

                    # 檢查 scaler 是否為 None
                    if scaler is None:
                        print("Scaler is None, cannot transform distances.")
                        return frame

                    if confidence >= 0.70:
                        if current_answer_list_index < len(value_left):
                            expected_label_left = value_left[current_answer_list_index]
                            expected_label_right = value_right[current_answer_list_index]

                            if (label_left == expected_label_left and label_right == expected_label_right):
                                if not answered_all_questions:
                                    answer_list.append(answer_already[current_answer_list_index])
                                    current_answer_list_index += 1
                            elif current_answer_list_index == len(value_right) and answer_list != value_left:
                                if not answered_all_questions:
                                    answer_list = []
                                    current_answer_list_index = 0
                    elif current_answer_list_index > len(value_left):
                        answer_list = []
                        current_answer_list_index = 0

                    display_text = f"Left Hand: {label_left}, Right Hand: {label_right}"
                    Answers_text = f"Answers:{format_answers_list(answer_list)}"

        print(display_text)
        print(Answers_text)
        cv2.putText(image, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, Answers_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2, cv2.LINE_AA)

        mp.solutions.drawing_utils.draw_landmarks(
            image, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(
            image, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 更新顯示文字
    correct_questions_no_display = f"正確題號: {correct_questions_no}"
    wrong_questions_no_display = f"錯誤題號: {wrong_questions_no}"
    if not answered_all_questions:
        key, values = list(questions[current_question_index].items())[0]
        value_left, value_right, answer_already = values
        question_display = f"問題 {current_question_index + 1}: 請比出「{key}」的手語"
        timer_display = f"剩餘時間: {question_timer_limit - (time.time() - question_timer_start):.1f} 秒"
        current_wrong_answers_display = f"當前累計錯誤次數: {current_wrong_answers} / {max_wrong_answers}"
    else:
        question_display = ""
        timer_display = ""
        current_wrong_answers_display = ""

    # 更新警告訊息
    if time.time() - alert_timer_start > alert_timer_limit:
        alert_message = ""
        alert_timer_start = time.time()

    image_hight, image_width, _ = image.shape
    annotated_image = image.copy()

    # 在影像上添加文字
        # Put text on the frame with green color
    annotated_image = draw_text(frame, display_text, (10, 50), (0, 0, 255))
    annotated_image = draw_text(annotated_image, Answers_text, (10, 70), (0, 0, 255))
    annotated_image = draw_text(annotated_image, question_display, (10, 30), (0, 255, 0))
    annotated_image = draw_text(annotated_image, timer_display, (400, 30), (0, 255, 0))
    annotated_image = draw_text(annotated_image, alert_message, (10, 80), (0, 255, 0))
    annotated_image = draw_text(annotated_image, correct_questions_no_display, (10, 350), (0, 255, 0))
    annotated_image = draw_text(annotated_image, wrong_questions_no_display, (10, 400), (0, 255, 0))
    annotated_image = draw_text(annotated_image, current_wrong_answers_display, (10, 450), (0, 255, 0))
    # annotated_image = cv2_chinese_text(
    #     annotated_image, question_display, 10, 30, (0, 255, 0))
    # annotated_image = cv2_chinese_text(
    #     annotated_image, timer_display, 400, 30, (0, 255, 0))
    # annotated_image = cv2_chinese_text(
    #     annotated_image, alert_message, 10, 80, (0, 255, 0))
    # annotated_image = cv2_chinese_text(
    #     annotated_image, correct_questions_no_display, 10, 350, (0, 255, 0))
    # annotated_image = cv2_chinese_text(
    #     annotated_image, wrong_questions_no_display, 10, 400, (0, 255, 0))
    # annotated_image = cv2_chinese_text(
    #     annotated_image, current_wrong_answers_display, 10, 450, (0, 255, 0))

    # Answer correct
    if answer_list == answer_already and answered_all_questions == False:
        alert_message = f">>>> 第 {current_question_index_display} 題答對了！ <<<<"
        alert_timer_start = time.time()
        current_question_index = (current_question_index + 1) % num_questions
        question_timer_start = time.time()  # Restart question timer for new question
        current_wrong_answers = 0
        answer_list.append(True)
        correct_questions_no.append(current_question_index_display)
        current_question_index_display += 1
        current_answer_list_index = 0
        answer_list = []
        print('answer_list:', answer_list)
        print('correct_questions_no:', correct_questions_no)
        # 檢查是否已回答完所有題目
        if current_question_index == 0:
            answered_all_questions = True
            print('answered_all_questions:', answered_all_questions)

    # Check for question timer expiry and switch question if time is up
    if time.time() - question_timer_start > question_timer_limit and answered_all_questions == False:
        question_timer_start = time.time()  # Restart question timer for new question
        current_wrong_answers += 1
        current_answer_list_index = 0
        answer_list = []
        print('current_wrong_answers:', current_wrong_answers)
        if current_wrong_answers == max_wrong_answers:
            # answer_list.append(False)
            wrong_questions_no.append(current_question_index_display)
            current_question_index_display += 1
            current_question_index = (
                current_question_index + 1) % num_questions
            current_wrong_answers = 0
            print('answer_list:', answer_list)
            print('wrong_questions_no:', wrong_questions_no)
            # 檢查是否已回答完所有題目
            if current_question_index == 0:
                answered_all_questions = True
                print('answered_all_questions:', answered_all_questions)

    # 檢查是否已回答完一輪所有題目
    if current_question_index == 0 and answered_all_questions:
        # annotated_image = cv2_chinese_text(
        #     annotated_image, "測驗結束", 10, 450, (0, 0, 255))
        # annotated_image = cv2_chinese_text(
        #     annotated_image, "本次測驗結果 :", 10, 500, (0, 0, 255))
        annotated_image = draw_text(frame, "測驗結束", (10, 450), (0, 0, 255))
        annotated_image = draw_text(frame, "本次測驗結果 :", (10, 500), (0, 0, 255))
    
    # flip it back and return
    return cv2.flip(annotated_image, 1)
# cv2.destroyAllWindows()
    
    