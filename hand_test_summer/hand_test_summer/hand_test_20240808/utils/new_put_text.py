
"""
coding=utf-8
cv2解决绘制中文乱码的问题
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
 
 
def cv2_chinese_text(img, text, left, top, text_color=(0, 255, 0)):
    text_size = 25
    # 判断是否OpenCV图片类型
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    font_style = ImageFont.truetype(r"./font/GenSekiGothic-R.ttc",
                                    text_size, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, text_color, font=font_style)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
