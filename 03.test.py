from utils.network import get_model

import numpy as np 
import os 
import cv2 
import mediapipe as mp
from utils.mediapipetools import *

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


threshold = 0.90



# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['nothing', 'ready', 'stop'])

# Videos are going to be 30 frames in length
sequence_length = 15

model = get_model(actions, sequence_length, 88)


model.load_weights('action.h5')

# 1. New detection variables
sequence = []
sentence = []


cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]

        print('keypoints:',len(keypoints))
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]
        
        if len(sequence) == sequence_length:
            try:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                select = np.argmax(res)
                print(actions[select])
                print(res[select]*100)
            except:
                sequence = []
                continue
            
            
        #3. Viz logic
            if res[select] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
            
            else:
                sentence = []
            

            if len(sentence) > 1: 
                sentence = sentence[-1:]

            # Viz probabilities
            image = prob_viz(res, actions, image)
            
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()