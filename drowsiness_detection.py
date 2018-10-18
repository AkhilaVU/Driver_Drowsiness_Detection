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