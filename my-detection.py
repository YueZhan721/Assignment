from jetson.inference import detectNet
from jetson.utils import loadImage, saveImage, videoSource, videoOutput

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
img = loadImage("/home/nvidia/Desktop/ZhanYue/Assignment3/detection/ssd/data/fruit/test/ad9a41a222faef4c.jpg")

detections = net.Detect(img)
for detection in detections:
    print(detection)

saveImage("/home/nvidia/Desktop/ZhanYue/Assignment3/detection/ssd/data/fruit/output.jpg",img)


