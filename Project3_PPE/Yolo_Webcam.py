from sympy import ceiling
from torch.cuda import device
from ultralytics import YOLO
import cv2
import cvzone
import math
import torch
# for webcam
# cap=cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

cap = cv2.VideoCapture('../Videos/motorbikes-1.mp4')

model = YOLO('../Yolo_Weights/yolov8n.pt').to('cuda')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success,frame=cap.read()
    if not success:
        break
    results = model(frame,stream=True,device='cuda')
    for result in results:
        # 通常单帧只有1个result
        # result单帧的所有检测结果,results一个或多个result,
        boxes = result.boxes
        # boxes单帧中所有检测到的边框信息
        # 每个 box 对应一个目标的坐标和置信度等信息。
        for box in boxes:
            # 在目标检测模型中，一个 box 可能包含多个候选框（尽管大多数情况下每目标只保留一个最优框），
            # 因此 box.xyxy 返回的可能是二维张量（如形状 (N,4)，N 为候选框数量）。
            # [0] 表示取第一个（即置信度最高的）边界框的坐标。
            # xyxy：边界框的绝对坐标（左上角 (x1,y1) 和右下角 (x2,y2)）
            x1,y1,x2,y2 = map(int,box.xyxy[0].cpu().numpy())
            # 将张量坐标转换为整数像素值，确保cv2.rectangle正确绘制。
            # cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),2)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(frame,(x1,y1,w,h))

            # confidence置信度
            conf=math.ceil(box.conf[0]*100)/100
            # 保留两位小数
            # class类别
            cls=int(box.cls[0])
            cvzone.putTextRect(frame,f'{classNames[cls]} {conf}',(max(x1,0),max(y1+20,0)),scale=2,thickness=2)


    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
