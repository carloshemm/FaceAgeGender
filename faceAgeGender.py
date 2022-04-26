import numpy as np
import cv2
from facedetector import FaceLandmarks
from agegender import AgeGender

from bytetracker.byte_tracker import BYTETracker
from bytetracker.byte_tracker import get_config

ageGender = AgeGender()
faces = FaceLandmarks()

cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)

# Load ByteTracker
cfg = get_config()
cfg.merge_from_file("bytetracker/byte.yaml")
byte = BYTETracker(cfg)

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def plot_tracking(image, tlwhs, obj_ids, ids2=None):
    # image = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        cv2.rectangle(image, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(image, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)

while cap.isOpened():
    cap.read()
    ret, frame = cap.read()
    frameCopy = frame.copy()
    if ret:
        try:
            faceboxes, confidences = faces.run_model(frame)
            if faceboxes is not None:
                test_size = (float(cfg.BYTETRACK.TEST_SIZE.split(',')[0]), float(cfg.BYTETRACK.TEST_SIZE.split(',')[1]))
                outputs = byte.update(np.array(faceboxes), np.array(confidences), frame.shape, test_size)
                scale = min(test_size[0]/frame.shape[0],test_size[1]/frame.shape[1] )
                for t in outputs:
                    tlwh = t.tlwh*scale
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > cfg.BYTETRACK.MIN_BOX_AREA and not vertical:
                        plot_tracking(frame, [tlwh], [tid])
                
                for track in byte.tracked_stracks:
                    x1 = int(track.tlwh[0])
                    y1 = int(track.tlwh[1])
                    x2 = int(track.tlwh[0] + track.tlwh[2])
                    y2 = int(track.tlwh[1] + track.tlwh[3])
                    crop = frame[y1:y2, x1:x2]
                    age, gender = ageGender.run_model(crop)
                    if track.sexo == "TBD" or track.idade == "TBD":
                        track.sexo = gender
                        track.idade = age
                    label = str(track.sexo)+"  "+str(track.idade)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                    top = max(y1, labelSize[1])
                    cv2.rectangle(frameCopy,(x1,y1),(x1+labelSize[0],y1-labelSize[1]),(0,0,230), -1)
                    cv2.rectangle(frameCopy, (x1,y1), (x2,y2), (0,0,230),3)
                    cv2.putText(frameCopy, label, (x1, top), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),2)
            frameCopy = cv2.resize(frameCopy, (1600,900))
            cv2.imshow("teste", frameCopy)
        
            if cv2.waitKey(1) == 27:
                break
        except:                
            frameCopy = cv2.resize(frameCopy, (1600,900))
            cv2.imshow("teste", frameCopy)
        
            if cv2.waitKey(1) == 27:
                break
    