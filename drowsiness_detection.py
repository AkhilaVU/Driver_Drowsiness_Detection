'''
		DRIVER DROWSINESS MONITORING SYSTEM USING 
		VISUAL BEHAVIOUR & MACHINE LEARNING
Versions:
	Python2.7
	Ubuntu14.04 (uname -a ) 
	
Used Modules:
	*dlib
	 https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/
	*imutils
	 pip install imutils
	*OpenCV
	 sudo apt-get install python-opencv
	*math
	*os

'''

import cv2 #computer vision
import os
import dlib
from imutils import face_utils
from math import sqrt

print('Initialising...')

print('Setting up camera...')
cam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_DUPLEX
bottomLeftCornerOfText = (10, 400)
fontScale = .5
fontColor = (255, 255, 255)
lineType = 1

def minus(p1,p2):
	x1 = p1[0]
	y1 = p1[1]
	x2 = p2[0]
	y2 = p2[1]
	out = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
	return out

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cam.read()
    #cv2.imshow("test", frame)

    k = cv2.waitKey(1)

    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    	
	# loop over the face detections
    for rect in rects:
    	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape1 = predictor(gray, rect)
	shape1 = face_utils.shape_to_np(shape1)
	
	print('__________________________-')
	
	left_eye = shape1[36:42]
	right_eye = shape1[42:48]
	nose = shape1[27:31]
	mouth = shape1[48:60]
	
		
	# loop over the (x, y)-coordinates for the left eye landmarks
	# and draw them on the image
	for (x, y) in left_eye:
        	#print(x,y)
		cv2.circle(frame, (x, y), 2, (255, 0, 0), -10)

	# loop over the (x, y)-coordinates for the right eye landmarks
	# and draw them on the image
	for (x, y) in right_eye:
        	#print(x,y)
		cv2.circle(frame, (x, y), 2, (255, 0, 0), -10)
	
	# loop over the (x, y)-coordinates for the nose landmarks
	# and draw them on the image
	for (x, y) in nose:
        	#print(x,y)
		cv2.circle(frame, (x, y), 2, (255, 255, 0), -10)
		
  	# loop over the (x, y)-coordinates for the mouth landmarks
	# and draw them on the image
	for (x, y) in mouth:
        	#print(x,y)
		cv2.circle(frame, (x, y), 2, (0, 0, 255), -10)
		

           
    # show the frame
    cv2.imshow("Frame", frame)
    
    cv2.putText(
        frame, 'Driver',
        tuple(p[17]),
        font,
        fontScale,
        fontColor,
        lineType
        )
        
    cv2.imshow("Faces found", frame)    
    key = cv2.waitKey(1) & 0xFF
    
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
	break
 
# do a bit of cleanup
cam.release()
cv2.destroyAllWindows()

