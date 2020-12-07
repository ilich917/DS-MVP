import numpy as np
import cv2
import os
import sys
import pdb
import csv

MIN_VIDEO_LENGTH = 2 # in seconds. face segments smaller than this are ignored

TRACK_SMOOTH_NUMFRAMES = 10 # number of past frames to look at for smoothing tracking

ROI_SCALE = 1.3;

SKIP_FRAMES_NUM = 5;

def show(img, wait):
    cv2.imshow('image',img)
    cv2.waitKey(wait)

def PV(arr):
    for x in arr:
        print (x);

def PNN(ss):
     print(ss,end=', ',flush=True)


def getFrame(cap,i,show):
    cap.set(1, i)
    ret, frame = cap.read()
    if (show==1):
        cv2.imshow('frame',frame);
        cv2.waitKey(0);
    return frame;

def readFile(fil):
    lines = [];
    with open(fil) as f:
        lines = f.readlines()
    return lines;

def viewVideo(vid):
    
    frameNo = 0;
    
    while(True):
      ret, frame = vid.read()
     
      if ret == True: 
        frameNo+=1;

        # Write the frame into the file 'output.avi'
        # out.write(frame)
        # PNN(frameNo);

        # Display the resulting frame    
        cv2.imshow('frame',frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
     
      # Break the loop
      else:
        break 


def cropFrame(vid, time, roi):
    # roi is in [1, num_pixels]

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    total_frames = int(vid.get(7))
    FPS = vid.get(cv2.CAP_PROP_FPS)
    
    # pdb.set_trace();

    frameNo = int(time*FPS)+1

    if (frameNo<0 or frameNo>total_frames):
        print("ERROR : frameNo:",frameNo, " is invalid.")
        return None;

    frame  = getFrame(vid, frameNo, 0);

    roi = list(map(int,roi))

    xmin = roi[0];
    xmax = roi[2];
    ymin = roi[1];
    ymax = roi[3];


    if (xmin<0 or xmax>frame_width or ymin<0 or ymax>frame_height):
        print("ERROR : ymin,ymax, xmin,xmax : ", ymin,", ",ymax,", ", xmin,", ",xmax);   
        return None

    cropped = frame[ymin:ymax, xmin:xmax, :]

    return cropped;


# START PROGRAM

#if (len(sys.argv) < 3):
    print("Usage: python faceCrop <video.mp4> <trackFile.txt> <rootFolder>")



#####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#      ESTO ES LO QUE TENGO QUE CAMBIAR
#####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#'MesesLSM.mp4'
videoFile = 'SemanasLSM.MOV'
trackFile = 'faceTrack.csv'
c_videos = 'C:/Users/Fernando/Documents/DIRECT-SIGN/PREPROCESSING/Videos/'
c_track = 'C:/Users/Fernando/Documents/DIRECT-SIGN/PREPROCESSING/Track/'
videoFile = os.path.join(c_videos, videoFile)
trackFile = os.path.join(c_track, trackFile)

print(videoFile, "\n", trackFile);

# READ VIDEO

vid = cv2.VideoCapture(videoFile)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
total_frames = int(vid.get(7))
FPS = vid.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

print("height:",frame_height,", width:",frame_width, ", total_frames:",total_frames, ", FPS:", FPS)


### Desde aquí el detector de caras

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

num = total_frames
print(num)
faceTrack = np.zeros(shape=(num, 300, 300))
shape = 300,300

c = 0
with open(trackFile, "w", newline="") as f:
    wr = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
    while c < num:
        
        
        _, frame = vid.read()
        time = vid.get(cv2.CAP_PROP_POS_MSEC)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(150,150))
        print("faces : ",list(faces))
        #print(faces)
        #dist1 = cv2.convertScaleAbs(faces)
        #dist2 = cv2.normalize(faces, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
        #cv2.imshow('cara',dist2)
        for (x, y, w, h) in faces:
            print('x,y,w,h : ',x,y,w,h)
            if (x+w>=frame_width) or (y+h>=frame_height):
                print("te pasaste")
            
            
            linea = map(float,[time, 0,  x, y, x+w, y+h])
            #resized = cv2.resize(frame[x:x+w, y:y+h], (300,300), cv2.INTER_AREA)
            wr.writerow(list(linea))
            crop = gray[y:y+h,x:x+w]
            print('crop shape : ',crop.shape)
            resized = cv2.resize(crop,shape)
            
            faceTrack[c] = resized.reshape(1,300,300)
            
        
        c += 1   
        if c % 1== 0:
            print('frame: ', c)
           # print(faceTrack[c-1].shape)
        
        #print(c)
f.close()

#
print("Se han detectado las caras en el video")

#hasta aquí el crop

########################
################### Grabar video:



#print(type(faceTrack))

import skvideo
import skvideo.io

output = "out.mp4"

writer = skvideo.io.FFmpegWriter(output)
        


#fourcc = cv2.VideoWriter_fourcc(*'DIVX')

#out = cv2.VideoWriter(output,fourcc, FPS, shape)

#skvideo.io.write(output,faceTrack)

for frame in faceTrack:
    
    writer.writeFrame(frame)
    #out.write(frame)
    
writer.close()

#out.realease()

# READ TRACK
faceTrack = []
#print(trackFile)
with open(trackFile, "rt") as f:
    rd = csv.reader(f, quoting = csv.QUOTE_NONNUMERIC)
    for lista in rd:
        faceTrack.append(lista)
        
        
        
# dict containing all detection for each face. indexed by face number
faces = {} 
print(len(faceTrack)==total_frames)

for row in faceTrack:
    #x = (x.split(','))
    #x = list(map(float,x[0:-1]))
    if (row[1] not in faces):
        faces[row[1]] = []   # agrega una nueva cara al diccionario de caras, tipo faces[2], para la segunda cara. 
                            # por lo tanto separa cada dimensión por cada cara.

    faces[row[1]].append([row[0],row[2],row[3],row[4],row[5]])   # junta el tiempo y los 4 puntos de la roi en una lista, anexa a la cara. ejemplo:
                                                    # ...1.0: [[0.04, 0.514, 0.19, 0.667, 0.463], [0.08, 0.521, 0.192, 0.678, 0.471]],...

    # break;
#print(faces)
"""
croppedFacesPath = 'croppedFaces/'

def convertToPixels(roi, vid):
    
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    xmin = int(roi[0]*frame_width)
    xmax = int(roi[2]*frame_width)
    ymin = int(roi[1]*frame_height)
    ymax = int(roi[3]*frame_height)

    roi = [xmin,ymin,xmax,ymax];
    return roi

def getMeanRoi(rois,vid):
    rois = np.array(rois);
    rois = rois[:,1:]; # remove time

    meanROI = np.mean(rois,0)
    meanROI = convertToPixels(meanROI, vid);

    return meanROI

def checkROI(roi, vid):
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    if (roi[0]<0 or roi[2]>frame_width or roi[1]<0 or roi[3]>frame_height):
        return 0

    return 1;

def getROIPixelCenter(roi,vid):
    roi = convertToPixels(roi,vid);

    cx = int((roi[0]+roi[2])/2)
    cy = int((roi[1]+roi[3])/2)

    return cx,cy


def getROIRelativeCenter(roi,vid):
    
    cx = (roi[0]+roi[2])/2.0
    cy = (roi[1]+roi[3])/2.0

    return cx,cy

def resizeROI(roi, meanROI,smoothCenters, roi_scale, vid):
    # roi is in [0,1]
    # meanROI is in [1,num_pixels]

    roi = np.array(roi);
    meanROI = np.array(meanROI);

    MW = int((meanROI[2]-meanROI[0])*roi_scale); # mean width
    MH = int((meanROI[3]-meanROI[1])*roi_scale); # mean height

    roi = convertToPixels(roi,vid);

    # cx,cy = getROIPixelCenter(roi,vid)
    SC = np.array(smoothCenters)
    # print ("SC:",SC)

    # get smoothened center
    cx = int(np.mean(SC,0)[0]);
    cy = int(np.mean(SC,0)[1]);

    # print (cx,",",cy)

    xmin = cx-MW/2
    ymin = cy-MH/2
    xmax = cx+MW/2
    ymax = cy+MH/2

    # resizedROI = [cx-MW/2 , cy-MH/2, cx+MW/2, cy+MH/2 ];
    resizedROI = [xmin, ymin, xmax, ymax];

    if (checkROI(resizedROI,vid) == 0):
        print("Dropping ROI:",resizedROI);
        resizedROI = roi # this should be resized later

    return resizedROI;


for f in faces.items():
    print("Face :",f[0], "Time:", f[1][0][0]," to ",f[1][-1][0]);
    
    if ( ( f[1][-1][0] - f[1][0][0] ) < MIN_VIDEO_LENGTH ):
        continue;

    # PV(f[1]);

    # print(f[1])

    fourcc = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc('M','J','4','2')
    #videoName = croppedFacesPath+str(f[0])+'.avi';
    videoName = croppedFacesPath+"/"+videoFile.split('/')[1]+"_"+str(f[0])+'.avi';
    
    meanROI = getMeanRoi(f[1],vid);

    print ("meanROI: ",meanROI);
    oMW = int(meanROI[2]-meanROI[0]); # mean width
    oMH = int(meanROI[3]-meanROI[1]); # mean height

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    # Scale ROI region
    MW = int( oMW*ROI_SCALE )
    MH = int( oMH*ROI_SCALE )

    roi_scale = ROI_SCALE;

    if (MW<0 or MH<0 or MW>frame_width or MH>frame_height):
        roi_scale = 1.0;
        MW = oMW;
        MH = oMH;

    outVideoSize = (MW,MH)
    out = cv2.VideoWriter(videoName,fourcc, FPS, outVideoSize)


    smoothCenters = [];
    

    segmentFrames = len(f[1]);
    frameNum = 0;
    for fr in f[1]:
        frameNum+=1;
        # skip first few and last few frames
        if (frameNum<SKIP_FRAMES_NUM or frameNum>(segmentFrames- SKIP_FRAMES_NUM )):
            continue;

        # ROI in fr[1:]. format: [xmin,ymin,xmax,ymax] as in tracking txt file

        cx,cy = getROIPixelCenter(fr[1:],vid);

        if (len(smoothCenters)<1 ):
            for ind in range(0,TRACK_SMOOTH_NUMFRAMES):
                smoothCenters.append([cx,cy])

        # remove oldest center and add current
        temp = smoothCenters.pop(0);
        smoothCenters.append([cx,cy]);
        # print(len(smoothCenters));

        # resize roi to crop region of mean width and height
        resizedROI = resizeROI(fr[1:], meanROI, smoothCenters,roi_scale, vid);

        # if (checkROI(resizedROI,vid) == 0):
        #     print("Dropping ROI:",resizedROI);
        #     continue;

        # print ("resizedROI: ",resizedROI);

        cropped = cropFrame(vid,fr[0], resizedROI);
        
        # check if roi or frameNo is invalid. If yes, remove its file
        if (cropped is None):
            print("Face:",f[0], "bounds error(roi or frameNo).")
            print("deleting file:", videoName)
            out.release()
            os.remove(videoName);
            break;

        cropped = cv2.resize(cropped, outVideoSize) # REMOVE THIS and crop single size roi from video 

        # print("Cropping frame:",f[0]," - ", fr);
        # show(cropped,2);

        out.write(cropped)

    cv2.destroyAllWindows()
    out.release()

# viewVideo(vid);


for i in range(0,200):
    PNN(str(i))
    ff = getFrame(vid, i, 1);

# When everything done, release the video capture and video write objects
vid.release()
# out.release()

# Closes all the frames
cv2.destroyAllWindows() 
"""