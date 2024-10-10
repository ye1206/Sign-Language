from django.shortcuts import render, redirect
from django.db import connection
from .models import Login, User
from django.contrib import messages
from django.http import StreamingHttpResponse
from myWeb.camera import VideoCamera, gen
from django.core.cache import cache

# Create your views here.


def login(request):
    web_Login = Login(request.POST)  # get the form
    error = []  # return error, and alert
    if request.method == "POST":
        if web_Login.is_valid():
            cursor = connection.cursor()  # 連接資料庫
            cursor.execute(
                "select `id`, `username`, `password` from webs_User;")  # select all user from db
            all_users = cursor.fetchall()  # get all user in db

            name = request.POST["username"]  # get the username
            pwd = request.POST["password"]  # get the password

            correct = False  # 是否有找到帳密
            for i in range(len(all_users)):  # check all user
                if str(all_users[i][1]) == name and str(all_users[i][2]) == pwd:  # 帳密正確
                    correct = True  # 找到帳密
                    return redirect("/index/")  # 跳轉到首頁

            if not correct:  # 帳密錯誤
                error.append("wrong account or password, login failed")

    context = {"error": error, "webs_login": web_Login}  # 將錯誤訊息和表單傳到前端
    return render(request, "login.html", context)  # 渲染 context 內容到 login.html


def register(request):
    if request.method == "POST":
        username = request.POST.get("username")  # get the username
        password = request.POST.get("password")  # get the password

        if User.objects.filter(username=username).exists():  # 如果使用者名稱已經存在
            return render(request, "register.html", {"error": "Username already exists"})

        with connection.cursor() as cursor:  # 連接資料庫
            cursor.execute(
                "INSERT INTO webs_User (username, password) VALUES (%s, %s)",
                [username, password],)  # 將註冊資訊新增到資料庫
            cursor.execute("SELECT LAST_INSERT_ID()")  # select 出最後一筆資料的id
            result = cursor.fetchone()  # 取得最後一筆資料的id
            if result is not None:  # 註冊成功
                id = result[0]
            else:
                messages.error(request, "註冊失敗，請再試一次。")

        return redirect("/")  # 回傳登入頁面

    # 渲染context內容到register.html
    return render(request, "register.html", context={"title": "註冊"})


def index(request):
    return render(request, 'index.html')


def tutorial(request):
    return render(request, 'tutorial.html')


def test(request):
    return render(request, 'test.html')


def tutorialClass(request):
    return render(request, 'tutorialClass.html')


def tutorialGreet(request):
    return render(request, 'tutorialGreet.html')


def tutorialMeat(request):
    return render(request, 'tutorialMeat.html')


def tutorialLeisure(request):
    return render(request, 'tutorialLeisure.html')


def tutorialRelation(request):
    return render(request, 'tutorialRelation.html')


def stream(request):
    topic = request.GET.get('topic')
    width, height = 640, 480
    return StreamingHttpResponse(gen(VideoCamera(topic), width, height),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def testClass(request):
    return render(request, 'testClass.html')


def testGreet(request):
    return render(request, 'testGreet.html')


def testMeat(request):
    return render(request, 'testMeat.html')


def testLeisure(request):
    return render(request, 'testLeisure.html')


def testRelation(request):
    return render(request, 'testRelation.html')


def instruction(request):
    return render(request, 'instruction.html')
