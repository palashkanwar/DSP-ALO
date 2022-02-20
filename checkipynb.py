#from tensorflow.keras.models import Sequential
#from keras.layers.normalization import layer_normalization
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

#os.environ['TFHUB_CACHE_DIR'] = '/Desktop/Pose_estimation-main/'
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

cap = cv2.VideoCapture('Camera1Clip1.MP4')
while cap.isOpened():

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
cv2.destroyAllWindows()

print("works")