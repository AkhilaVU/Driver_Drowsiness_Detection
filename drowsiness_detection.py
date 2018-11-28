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

import cv2
import os
import dlib
from imutils import face_utils
from math import sqrt
#import matplotlib.pyplot as plt

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
num_setup_frames = 90
frame_count = 0
alert_count = 0
EAR_vals = []
NLR_vals = []
MOR_max = 0

while True:
    if alert_count > 45:
        print('\n\n\t\t\tDANGER!!!\n\t\tDrowsiness Detected')
        break
    frame_count += 1
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
    #print(rects)
    if len(rects) == 0:
        continue
    elif len(rects) == 1:
        rect = rects[0]
    else:
        rect = rects[0]
        
        
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape1 = predictor(gray, rect)
    shape1 = face_utils.shape_to_np(shape1)
    print(' ')
    print(str(frame_count) + '__________________________'+str(alert_count))

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
        

        
    p = shape1
        
    # Calculate Eye Aspect Ratio
    EAR = (minus(p[37],p[41]) + minus(p[38],p[40]))/(2*minus(p[39],p[36]))
    print("EAR: "+ str(EAR))  
        
     
        # Calculate MOuth Opening Ratio
    MOR = (minus(p[50],p[58]) + minus(p[51],p[57]) + minus(p[52],p[56]))/(3*minus(p[54],p[48])) 
    print("MOR: "+ str(MOR)+ "\t"+ str(MOR_max))

    #Nose Length Ratio
    nose_len = minus(p[30],p[27])
    print("Nose Length: " + str(nose_len))
    
    if frame_count < num_setup_frames:
        EAR_vals.append(EAR)
        NLR_vals.append(nose_len)
        if MOR > MOR_max :
            MOR_max = MOR
    elif frame_count == num_setup_frames:
        print('\n\nSetup phase completed.')
        
        EAR_vals.sort()
        EAR_thres = sum(EAR_vals[0:len(EAR_vals)/2])/(len(EAR_vals)/2)

        avg_nose_len = sum(NLR_vals)/len(NLR_vals)
        print('EAR Threshold: ' + str(EAR_thres))
        print('Average Nose Length: ' + str(avg_nose_len))
        print('Maximum MOR: ' + str(MOR_max))
        
    else:
        print("NLR: " + str(nose_len/avg_nose_len))
        if EAR < EAR_thres:
            alert_count +=1
            print('\n\tAlert!!! Eyes Closing!!!\n')
            continue
        elif MOR > MOR_max:
            alert_count +=1
            print('\n\tAlert!!! Yawning!!!\n')
            continue
        elif not(.9 < (nose_len/avg_nose_len) < 1.1):
            alert_count +=1
            print('\n\tAlert!!! Head Bending!!!\n')
            continue
        

    
    cv2.putText(
        frame, 'Driver',
        tuple(p[17]),
        font,
        fontScale,
        fontColor,
        lineType
        )
    # show the frame
    cv2.imshow("Frame", frame)        
    #cv2.imshow("Faces found", frame)    
    key = cv2.waitKey(1) & 0xFF
    
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

#plt.plot(EAR_vals,'--r^')
# do a bit of cleanup
cam.release()
cv2.destroyAllWindows()

