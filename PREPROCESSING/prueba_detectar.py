import numpy as np
import os
import time
import cv2
from detectar import Detector

start_time = time.clock()


videoFile = 'MesesLSM.mp4'
trackFile = 'MesesLSM_faces.csv'
c_videos = 'C:/Users/Fernando/Documents/GitHub/DIRECT-SIGN/PREPROCESSING/Videos/'
c_track = 'C:/Users/Fernando/Documents/GitHub/DIRECT-SIGN/PREPROCESSING/Track/'
c_anot = 'C:/Users/Fernando/Documents/GitHub/DIRECT-SIGN/PREPROCESSING/csvs/'
videoFile = os.path.join(c_videos, videoFile)
trackFile = os.path.join(c_track, trackFile)

# READ VIDEO

video = cv2.VideoCapture(videoFile)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

detect = Detector(video, face_cascade, trackFile)
caras = detect.detectar()

print("--- %s seconds ---" % (time.clock() - start_time))