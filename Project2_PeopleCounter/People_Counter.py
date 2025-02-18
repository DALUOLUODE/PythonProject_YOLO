
from ultralytics import YOLO
import cv2
import cvzone
import math
import sort

cap = cv2.VideoCapture('../Videos/people.mp4')

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

mask=cv2.imread('mask.png')
# tracking
tracker = sort.Sort(max_age= 20,min_hits= 3,iou_threshold= 0.3)
limitsUp=[103,161,296,161]
limitsDown=[527,489,735,489]
totalCountUp=[]
totalCountDown=[]

while True:
    success,frame=cap.read()
    frameRegion=cv2.bitwise_and(frame,mask)
    frameGraphics=cv2.imread('graphics.png',cv2.IMREAD_UNCHANGED)
    frame=cvzone.overlayPNG(frame,frameGraphics,(730,260))
    if not success:
        break
    results = model(frameRegion,stream=True,device='cuda')
    detections=sort.np.empty((0,5))
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1,y1,x2,y2 = map(int,box.xyxy[0].cpu().numpy())
            w, h = x2-x1, y2-y1
            conf=math.ceil(box.conf[0]*100)/100
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if currentClass== "person" and conf>0.5:
                # cvzone.putTextRect(frame,f'{currentClass} {conf}',(max(x1,0),max(y1+20,0)),
                #                    scale=1,thickness=1,offset=6)
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=6,rt=5)
                currentArray=sort.np.array([x1,y1,x2,y2,conf])
                detections=sort.np.vstack((detections,currentArray))

    resultsTacker=tracker.update(detections)

    cv2.line(frame,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)
    cv2.line(frame,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,255),5)

    for result in resultsTacker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=6, rt=2,colorR=(255,0,255))
        cvzone.putTextRect(frame, f'{int(id)}', (max(x1, 0), max(y1 + 20, 0)),
                           scale=2, thickness=2, offset=6)

        cx,cy=x1+w//2,y1+h//2
        cv2.circle(frame,(cx,cy),2,(255,0,255),2,cv2.FILLED)

        if limitsUp[0]<cx<limitsUp[2] and limitsUp[1]-20<cy<limitsUp[1]+20:
            if totalCountUp.count(id)==0:
                totalCountUp.append(id)
                cv2.line(frame, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0]<cx<limitsDown[2] and limitsDown[1]-20<cy<limitsDown[1]+20:
            if totalCountDown.count(id)==0:
                totalCountDown.append(id)
                cv2.line(frame, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(frame, f'Count:{len(totalCount)}', (50,50),)
    cv2.putText(frame,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(frame,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)


    cv2.imshow("frame",frame)
    # cv2.imshow("frameRegion",frameRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()