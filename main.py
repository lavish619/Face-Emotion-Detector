import cv2
import keras
import numpy as np
from models import trial_model, alexnet_model

label_indices ={ 0:'anger', 1:'contempt', 2:'disgust', 3:'fear',
                4:'happiness', 5:'neutral',6:'sadness',7:'surprise'}

num_classes = len(label_indices)

model =  trial_model((350,350,3),num_classes)
model.load_weights("trial_model.h5")

vid = cv2.VideoCapture(0)
while True:
    ret, frame1 = vid.read()
    
    frame1 = cv2.resize(frame1,(350,350))
    
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame = frame/255.0
    frame = np.expand_dims(frame, axis=0)
    indice = np.argmax(model.predict(frame))
    prediction = label_indices[indice]
    #print(prediction)
    cv2.putText(frame1, "Emotion: "+ prediction, (5,20), cv2.FONT_HERSHEY_SIMPLEX , 0.8 , (0,0,255),2)
    cv2.imshow('frame', frame1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        
vid.release()
cv2.destroyAllWindows()