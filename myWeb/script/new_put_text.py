
"""
coding=utf-8
cv2解决绘制中文乱码的问题
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from django.conf import settings


def cv2_chinese_text(img, text, left, top, text_color=(0, 255, 0)):
    text_size = 25
    # 判断是否OpenCV图片类型
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 字体的格式
    font_style = ImageFont.truetype(settings.FONT_DIR,
                                    text_size, encoding="utf-8")

    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    text_img = Image.new('RGBA', img.size, (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((left, top), text, text_color, font=font_style)

    text_img = text_img.transpose(Image.FLIP_LEFT_RIGHT)
    img.paste(text_img, (0, 0), text_img)

    # 绘制文本
    # draw.text((left, top), text, text_color, font=font_style)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
