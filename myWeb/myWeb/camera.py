import cv2
from script.hand_video_detector import hand_video
# import time

# basically take camera input and convert it into a cv object
# later to be processed by gen()


class VideoCamera(object):
	def __init__(self):  # Open the camera
		self.video = cv2.VideoCapture(0)  # 0 means the first camera

	def __del__(self):  # release the camera
		self.video.release()

	def get_frame(self):  # get the frame from the camera
		success, image = self.video.read()
		if success:
			# call the detection here
			image = hand_video(success, image)

		return image


# generator that saves the video captured if flag is set
def gen(camera, width, height):
    while True:
        frame = camera.get_frame()  # 獲取frame
        if frame is None:  # 如果frame是空的，則跳出迴圈
            break
        resized_frame = cv2.resize(frame, (width, height))  # 重新設定frame大小
        ret, jpeg = cv2.imencode('.jpg', resized_frame)  # 將frame轉換成jpeg格式
        frame_bytes = jpeg.tobytes()  # 轉換成bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')  # 回傳frame_bytes
