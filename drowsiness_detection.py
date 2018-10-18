#initial file
import cv2
import os

print('Initialising...')
myPath = os.getcwd()#Current Working Dir
cascPath  = myPath + '\haarcascades\haarcascade_frontalface_default.xml'

print('Setting up camera...')
cam = cv2.VideoCapture(0)

# Create the haar cascade
print('Creating the haar cascade...')
faceCascade = cv2.CascadeClassifier(cascPath)   # haar cascade

# cv2.namedWindow("test")
font                   = cv2.FONT_HERSHEY_DUPLEX
bottomLeftCornerOfText = (10,400)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 1

while True:
    ret, frame = cam.read()
     #cv2.imshow("test", frame)

    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
