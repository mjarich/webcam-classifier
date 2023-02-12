import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2 as cv

from camera import Camera
from labels import get_labels
label_names = get_labels()

# Best available weights (currently alias for IMAGENET1K_V2)
# Note that these weights may change across versions
# resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = models.resnet18(pretrained=True)
net.eval()
net.to(device)

def transform(image):
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	return transform(image)

def classify_loop():
	cam = Camera(0)
	while True:
		cam.capture_one_frame(show=True)
		frame = cam.frame
		if frame is not None:
			frame = transform(frame)[None].to(device)
			label = int(net(frame).argmax())
			print(label_names[label])
			
		if cv.waitKey(1) == ord('q'):
			break
	cam.device.release()
	cv.destroyAllWindows()

if __name__ == '__main__':
	classify_loop()
