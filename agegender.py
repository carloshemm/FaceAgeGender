import numpy as np
import cv2
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

path = config.get("MODELS", "ageGender_model")


class AgeGender:
    def __init__(self):
        self.net = cv2.dnn.readNet(config=cv2.samples.findFile(path+".xml"), model=cv2.samples.findFile(path+".bin"))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.outNames = self.net.getUnconnectedOutLayersNames()
        
    def run_model(self, crop):
        self.result = []
        
        try:
            crop = cv2.resize(crop, (62,62))
                
            blob = cv2.dnn.blobFromImage(crop, size=(62, 62),ddepth=cv2.CV_8U)
            self.net.setInput(blob)
            gender,age = self.net.forwardAndRetrieve(['prob', 'age_conv3'])
            
            if gender[0][0][1][0][0] > 0.8:
                gender = "Homem"
            else:
                gender = "Mulher"
            
            age = int(age[0][0][0][0][0]*100)
                
            self.result = (age,gender)
                           
            return self.result
        except:
            print("Erro em AgeGender")
            return ("TBD","TBD")