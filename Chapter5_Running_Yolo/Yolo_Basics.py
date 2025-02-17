from ultralytics import YOLO
import cv2

model = YOLO('../Yolo_Weights/yolov8l.pt')
results = model("Images/1.jpg",show=True)
cv2.waitKey(0)