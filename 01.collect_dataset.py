import cv2
import numpy as np
import os
import mediapipe as mp


from utils.mediapipetools import *

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['nothing'])

# Thirty videos worth of data
no_sequences = 60

# Videos are going to be 30 frames in length
sequence_length = 15

for action in actions: 
    for sequence in range(90, 90+no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
        
        
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(30, 30+no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, f'STARTING COLLECTION : {action}', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Video Number {sequence}/30', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6, cv2.LINE_AA)
                    # Show to screen
                    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(1000)
                else: 
                    cv2.putText(image, f'Video Number {sequence}/30', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6, cv2.LINE_AA)
                    # Show to screen
                    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()