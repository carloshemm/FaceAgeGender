import json
import requests
from random import randint,seed
import numpy as np

class CreateRequest:
    def __init__(self):
        dict = {'Embeddings' : []}
        with open('embeddings2.json', 'w') as jfile:
            json.dump(dict, jfile, indent=4)
            
    def makeRequest(self,track):
        if len(track.passedAreas) > 0 and (track.encoding_corpo is not None or track.encoding_face is not None):
            timestamp = track.timestamp
            idade = track.idade 
            humor = track.humor 
            sexo = track.sexo 
            encoding_corpo = track.encoding_corpo
            encoding_face = track.encoding_face
            area_id = track.passedAreas
            direcao = track.direcao
            timeInArea = list(np.subtract(track.timeInArea,track.timestamp))
            timeInArea = [int(i) for i in timeInArea]
            
            values = {
                'area_id': area_id,
                'idade': idade,
                'sexo': sexo,
                'humor': humor,
                'timestamp':timestamp,
                'timeInArea' : timeInArea,
                'encoding_corpo': encoding_corpo,
                'encoding_face': encoding_face
            }
            
            with open('embeddings2.json', 'r+') as f:
                data = json.load(f)
                data['Embeddings'].append(values)
                f.seek(0)        # <--- should reset file position to the beginning.
                json.dump(data, f, indent=4)
                f.truncate()
            
            #r = requests.post('http://172.40.3.236:3000/api/v1/visitantes/', json=values)
            #print(f'Response === {r}')
            #print(f'r json === {r.json()}')
        
            #for keys,item in values.items():
            #    print(f'{keys} -----> {item}')