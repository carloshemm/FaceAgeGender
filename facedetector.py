import cv2
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

path = config.get("MODELS", "facedet_model")
reidpath = config.get("MODELS", "facereid_model")
embeddingpath = config.get("MODELS", "reid_model")

tshd = eval(config.get("PARAMETERS", "faceTSHD"))

class FaceLandmarks:
    def __init__(self):
        self.detnet = cv2.dnn.readNet(config=cv2.samples.findFile(path+".xml"), model=cv2.samples.findFile(path+".bin"))
        self.detnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.detnet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
      
        self.result = []
        self.faces = []
        self.confidences = []
        
        
    def run_model(self,crop):
        self.result = []
        
        try:
            self.faces = []
            self.confidences = []
            result = []    
            blob = cv2.dnn.blobFromImage(crop, size=(256, 256),ddepth=cv2.CV_8U)
            self.detnet.setInput(blob)
            outs = self.detnet.forward()
                
            self.result = outs
            self.postprocess()
            for box in self.faces:
                x1,y1,x2,y2 = int(box[0]*crop.shape[1]*0.9),int(box[1]*crop.shape[0]*0.9),int(box[2]*crop.shape[1]*1.1),int(box[3]*crop.shape[0]*1.1)
                result.append([x1,y1,x2-x1,y2-y1])
            return result, self.confidences

        except:
            print("Erro em face embeddings")
            return None       
        
        
    def postprocess(self):
        for faces in self.result:
            for detection in faces[0]:
                confidence = detection[2]
                if confidence > tshd:
                    x1,y1,x2,y2 = detection[3], detection[4], detection[5], detection[6]
                    self.faces.append([x1,y1,x2,y2])
                    self.confidences.append(confidence)