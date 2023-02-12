import numpy as np
import cv2 as cv

class Camera:
	def __init__(self, device_id: int):
		self.device = cv.VideoCapture(device_id)
		assert self.device.isOpened(), 'Cannot open capture device.'
		
	def capture_one_frame(self, show=False):
		# Capture frame-by-frame
		ret, frame = self.device.read()
		
		# if frame is read correctly ret is True
		self.frame = frame if ret else None
		
		if show and ret:
			cv.imshow('frame', self.frame)
			
	def draw_bounding_box(self, center, height, width):
		low_corner = int(center[0] - width / 2), int(center[1] - height / 2)
		high_corner = int(center[0] + width / 2), int(center[1] + height / 2)
		bbox = cv.rectangle(self.frame, low_corner, high_corner, (0, 255 ,0), 2)
		cv.imshow('frame', bbox)
	
if __name__ == "__main__":
	cam = Camera(0)
	while True:
		cam.capture_one_frame(show=False)
		if cam.frame is not None:
			cam.draw_bounding_box(np.random.randint((500,500)), np.random.randint(40), np.random.randint(40))
		if cv.waitKey(1) == ord('q'):
			break
	cam.device.release()
	cv.destroyAllWindows()
