import mediapipe as mp
import cv2
import time
import pandas as pd
#from pose.py import *

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('video1.mp4')
pTime = 0

output_filename = 'myoutput.mp4'
output_frames_per_second = 20.0
file_size = (1920,1080)

'''while cap.isOpened():

    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        #img = frame.copy()
    
        #ret, frame = cap.read()
    
        # Resize image
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
        input_img = tf.cast(img, dtype=tf.int32)
    
        # Detection section

        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

        # Render keypoints 
        loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)
        cv2.imshow('Movenet Multipose', frame)
    
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

cap.release()
cv2.destroyAllWindows()'''
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
result = cv2.VideoWriter(output_filename, fourcc, output_frames_per_second, file_size) 
landmarks_df = pd.DataFrame()
keypoints = []
frame = 0
while cap.isOpened():
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        #print(results.pose_landmarks)
        #cv2.imshow(img)
        frame = frame + 1

        print("Frame in video: " + str(frame))

        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                print("Landmark ID: " + str(id))
                h, w,c = img.shape
                #print(id, lm)
                print(lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
                print("cx: " + str(cx))
                print("cy: " + str(cy))


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
        cv2.imshow("Image", img)
        result.write(img)
        cv2.waitKey(1)
        print("Frame: " + str(frame) + " has cTIME: "+ str(cTime))
        print("Frame: " + str(frame) + " has fps: "+ str(fps))

cap.release()
#landmarks_df.append(keypoints, ignore_index=True)
result.release()
