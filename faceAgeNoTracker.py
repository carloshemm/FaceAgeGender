import numpy as np
import cv2
from faceDet import FaceLandmarks
from agegender import AgeGender

ageGender = AgeGender()
faces = FaceLandmarks()

cap = cv2.VideoCapture(0)
#cap.set(3,1920) #definitions to logitech FHD camera
#cap.set(4,1080)
#cap.set(15,1.2)

while cap.isOpened():
    cap.read()
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frameCopy = frame.copy()
    if ret:
        faceboxes = faces.run_model(frame)
        if faceboxes is not None:
            for box in faceboxes:
                x1,y1,x2,y2 = box
                crop = frame[y1:y2, x1:x2]
                age, gender = ageGender.run_model(crop)
                label = str(gender)+"  "+str(age)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                top = max(y1, labelSize[1])
                cv2.rectangle(frameCopy,(x1,y1),(x1+labelSize[0],y1-labelSize[1]),(0,0,230), -1)
                cv2.rectangle(frameCopy, (x1,y1), (x2,y2), (0,0,230),3)
                cv2.putText(frameCopy, label, (x1, top), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),2)
                
        frameCopy = cv2.resize(frameCopy, (1280,640))
        cv2.imshow("teste", frameCopy)
        
        if cv2.waitKey(1) == 27:
            break
    
