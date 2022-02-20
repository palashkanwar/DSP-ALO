import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For static images:
'''with mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(file_list):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
    )
    # Draw pose landmarks on the image.
    annotated_image = image.copy()
    # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
    # upper_body_only is set to True.
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)'''
# global variables
pose_estimator = []
pose_estimator_dim = []
# For each object detected
# WHICH POSE ESTIMATOR TO USE.
selected_pose_idx=0
                                                     
if(len(pose_estimator)==0): 
   pose = mp_pose.Pose(min_detection_confidence=0.6,     
             min_tracking_confidence=0.6)
   pose_estimator.append(pose)    
   pose_estimator_dim.append(<detected object's boundary>)
   selected_pose_idx = len(pose_estimator)-1                            
elif(<object_id>>len(pose_estimator)):
   thresholdForNew = 100
   prev_high_score = 0
   selected_pose_idx_high =0
   prev_low_score = 1000000000
   selected_pose_idx_low =0
   pose_idx = 0
   for dim in pose_estimator_dim:
      score = compareDist(dim,<detected object's boundary>)
      if(score > prev_high_score):
         selected_pose_idx_high  =  pose_idx
         prev_high_score = score
      if(score < prev_low_score):                                        
         selected_pose_idx_low  =  pose_idx
         prev_low_score = score
      pose_idx+=1
   if prev_high_score > thresholdForNew:
      pose = mp_pose.Pose(min_detection_confidence=0.6,                   
min_tracking_confidence=0.6)
      pose_estimator.append(pose)    
      pose_estimator_dim.append(<detected object's boundary>)
      selected_pose_idx = len(pose_estimator)-1 
   else:
      selected_pose_idx = selected_pose_idx_low
   pose_estimator_dim[selected_pose_idx]=<detected object's boundary>
                                    
else:
   pose_idx = 0
   prev_score = 1000000000                                
   for dim in pose_estimator_dim:
      score = compareDist(dim,[x_min, y_min, box_width, box_height])
      if(score < prev_score):                                        
         selected_pose_idx  =  pose_idx
         prev_score = score   
      pose_idx+=1
   pose_estimator_dim[selected_pose_idx]=<detected object's boundary>
   
# For webcam input:
cap = cv2.VideoCapture("video1.mp4")
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()