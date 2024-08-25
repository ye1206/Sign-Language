import cv2
from script.hand_video_detector import hand_video
import numpy as np
# from script.hand_test_20240503.Demo_V3_Multi import recog
# import time

# basically take camera input and convert it into a cv object
# later to be processed by gen()


class VideoCamera(object):
	def __init__(self, topic):  # Open the camera
		self.video = cv2.VideoCapture(0)  # 0 means the first camera
		self.topic = topic

	def __del__(self):  # release the camera
		self.video.release()

	def get_frame(self):  # get the frame from the camera
		success, image = self.video.read()
		# if success:
			# call the detection here
		# image = hand_video(success, image, self.topic)
		# image = recog(image)

		if not success:
			print("Cannot get the frame")
			return None

		try:
			image = hand_video(success, image, self.topic)
			ret, jpeg = cv2.imencode('.jpg', image)
			return jpeg.tobytes()
		except Exception as e:
			print(f"Error when handling the frame: {str(e)}")
			return None

		# return image


# generator that saves the video captured if flag is set
def gen(camera, width, height):
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Failed to get frame, exiting loop")
            break
        try:
            # Decode the byte stream into a numpy array
            nparr = np.frombuffer(frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            resized_frame = cv2.resize(img, (width, height))
            ret, jpeg = cv2.imencode('.jpg', resized_frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        except Exception as e:
            print(f"Error generating frame: {str(e)}")
            break
