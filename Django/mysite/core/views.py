from django.shortcuts import render, redirect
from django.views.generic import TemplateView, ListView, CreateView
from django.core.files.storage import FileSystemStorage
from django.urls import reverse_lazy
from django.db import connection
from django.contrib import messages


from .forms import ImageForm

import urllib
import numpy as np
from script.hand_image_detector import hand_detection
import cv2
import mediapipe as mp
from utils.new_put_text import cv2_chinese_text
import joblib


from mysite.camera import VideoCamera, gen
from django.http import StreamingHttpResponse
from django.http import JsonResponse


class Home(TemplateView):
    template_name = 'index.html'


def image_upload_view(request):
    """Process images uploaded by users"""
    data = {"success": False}
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if request.FILES.get("image", None) is not None:
            image = _grab_image(stream=request.FILES["image"])
            # call the detection here
            annotated_image = hand_detection(image)
            # open cv window once it is done, to show the output image
            # alternatively one can put it in the html
            cv2.imshow("output", annotated_image)
            cv2.waitKey(0)

            form.save()
            img_obj = form.instance
            return render(request, 'image_upload.html', {'form': form, 'img_obj': img_obj})
    else:
        form = ImageForm()
    return render(request, 'image_upload.html', {'form': form})


# a helper function to convert img.url into a cv.img object
# for image upload and detection only
def _grab_image(path=None, stream=None, url=None):
    if path is not None:
        image = cv2.imread(path)
    else:
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

# for video input and detection
# the whole thing, video
# is returned as a streaming http response, or bytes


def video_stream(request):
    vid = StreamingHttpResponse(gen(VideoCamera(), False),
                                content_type='multipart/x-mixed-replace; boundary=frame')
    return vid


def video_save(request):
    vid = StreamingHttpResponse(gen(VideoCamera(), True),
                                content_type='multipart/x-mixed-replace; boundary=frame')
    return vid


def video_input(request):
    return render(request, 'video_input.html')


def cam(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            # 对每一帧进行处理
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Initialize mediapipe Hands Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Load the SVM model and scaler
model_filename = "svm_model_V3.pkl"
clf = joblib.load(model_filename)
scaler_filename = "scaler_V3.pkl"
scaler = joblib.load(scaler_filename)

# Load labels
label_file = "labels_V3.txt"
with open(label_file, 'r', encoding='utf-8') as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]


def compute_distances(landmarks):
    # Define pairs for distance calculation
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8),
             (0, 9), (0, 10), (0, 11), (0, 12),
             (0, 13), (0, 14), (0, 15), (0, 16),
             (0, 17), (0, 18), (0, 19), (0, 20),
             (4, 8), (8, 12), (12, 16), (16, 20)]

    distances = []
    reference_distance = np.linalg.norm(
        np.array([landmarks.landmark[0].x, landmarks.landmark[0].y]) -
        np.array([landmarks.landmark[9].x, landmarks.landmark[9].y]))

    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x,
                       landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x,
                       landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2)
        # Normalize the distance using the reference distance
        distances.append(distance/reference_distance)

    return distances


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        display_text = ''

        if results.multi_hand_landmarks:
            for index, landmarks in enumerate(results.multi_hand_landmarks):
                # Distinguish between left and right hand
                hand_label = "Right" if results.multi_handedness[
                    index].classification[0].label == "Left" else "Left"

                distances = compute_distances(landmarks)
                distances = scaler.transform([distances])

                prediction = clf.predict(distances)
                confidence = np.max(clf.predict_proba(distances))

                label = labels[prediction[0]]
                display_text = f"{hand_label} Hand: {label} ({confidence*100:.2f}%)"

                # To visualize the landmarks of the hand
                mp.solutions.drawing_utils.draw_landmarks(
                    image, landmarks, mp_hands.HAND_CONNECTIONS)

        img = cv2.putText(image, display_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()


def test(request):
    context = {
        'display_text': 'Placeholder text'  # 預設的文字，以防止首次載入時沒有文字
    }
    return render(request, 'test.html', context)


def tutorial(request):
    return render(request, 'tutorial.html')


def tutorialGreet(request):
    return render(request, 'tutorialGreet.html')


def tutorialMeat(request):
    return render(request, 'tutorialMeat.html')


def tutorialLeisure(request):
    return render(request, 'tutorialLeisure.html')


def tutorialRelation(request):
    return render(request, 'tutorialRelation.html')


def register(request):  # 註冊頁面
    if request.method == "POST":
        title = request.POST.get("title")  # get the title
        pwd = request.POST.get("pwd")  # get the password
        pwd = int(pwd)  # turn the password to int
        name = request.POST.get("name")  # get the name

        with connection.cursor() as cursor:  # 連接資料庫
            cursor.execute(
                # 將註冊資訊新增到資料庫
                "INSERT INTO cart_user (pwd, title, name) VALUES (%s, %s, %s)",
                [pwd, title, name],
            )
            cursor.execute("SELECT LAST_INSERT_ID()")  # select 出最後一筆資料的id
            result = cursor.fetchone()  # 取得最後一筆資料的id
            if result is not None:  # 註冊成功
                no = result[0]
            else:
                messages.error(request, "註冊失敗，請再試一次。")

        return redirect("/")  # 回傳登入頁面

    # 渲染context內容到register.html
    return render(request, "register.html", context={"title": "註冊"})


def testGreet(request):
    context = {
        'display_text': 'Placeholder text'  # 預設的文字，以防止首次載入時沒有文字
    }
    return render(request, 'testGreet.html', context)


def testMeat(request):
    context = {
        'display_text': 'Placeholder text'  # 預設的文字，以防止首次載入時沒有文字
    }
    return render(request, 'testMeat.html', context)


def testLeisure(request):
    context = {
        'display_text': 'Placeholder text'  # 預設的文字，以防止首次載入時沒有文字
    }
    return render(request, 'testLeisure.html', context)


def testRelation(request):
    context = {
        'display_text': 'Placeholder text'  # 預設的文字，以防止首次載入時沒有文字
    }
    return render(request, 'testRelation.html', context)
