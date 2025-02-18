import cv2
import cvzone
import math
from ultralytics import YOLO

# for webcam
# cap=cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

cap = cv2.VideoCapture('../Videos/ppe-1-1.mp4')
# ../当前目录的上一级查找

model = YOLO('best.pt').to('cuda')

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask',
              'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
              'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck',
              'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
              'trailer', 'truck and trailer', 'truck', 'van', 'vehicle',
              'wheel loader'
              ]
myColor = (255, 0, 0)

while True:
    success,frame=cap.read()
    if not success:
        break
    results = model(frame,stream=True,device='cuda')
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0].cpu().numpy())
            w, h = x2-x1, y2-y1
            cv2.rectangle(frame,(x1,y1),(x2,y2),myColor,3)
            conf=math.ceil(box.conf[0]*100)/100
            # 保留两位小数
            # class类别
            cls=int(box.cls[0])
            currentClass=classNames[cls]
            if conf>0.5:
                if currentClass=="Safety Vest" or currentClass=="Hardhat" or currentClass=="Mask":
                    myColor = (0, 255, 0)
                elif currentClass=="NO-Hardhat" or currentClass=="NO-Mask" or currentClass=="NO-Safety Vest":
                    myColor = (0, 0, 255)
                else:
                    myColor = (255, 0, 0)

                cv2.rectangle(frame,(x1,y1),(x2,y2),myColor,3)
                cvzone.putTextRect(frame,f'{currentClass} {conf}',
                                   (max(x1,0),max(y1+20,0)),scale=2,thickness=2,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor)


    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
