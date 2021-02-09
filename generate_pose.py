import os
"""
Antes de correr este fichero se debe crear y activar un env en conda con los siguientes requerimientos: python 3.9, moviepy.py
Además, debe descargarse openpose
Por último, se deben cambiar la ruta de OpenPoseDemo.exe.
"""

from moviepy.video.VideoClip import VideoClipos
from moviepy.video.io.VideoFileClip import VideoFileClip

from moviepy.video.fx.resize import resize
VideoFileClip.resize = resize

#from moviepy.video.fx import resize
dataset_path = 'PREPROCESSING/Videos'
dataset_path2 = 'PREPROCESSING/Videos_raw'

for sub in os.listdir(dataset_path):
    for file in os.listdir(dataset_path + '/' + sub):
            video = dataset_path + '/' + sub + '/' + file
            #clip = VideoFileClip(video)
            #clip = clip.resize(0.3)
            
            
            #os.remove(video)
            #video2 = dataset_path2 + '/' + sub + '/' + file
            #clip.write_videofile(video2)
            #clip.close()
            if os.path.isdir('PREPROCESSING/Pose/' + sub + '/' + file[:-4]):
                pass
            else:
                os.makedirs('PREPROCESSING/Pose/' + sub + '/' + file[:-4])
            command = r'C:\Users\Fernando\Documents\GitHub\Openpose-gpu\openpose\bin\OpenPoseDemo.exe --video {} --display 0 --model_pose BODY_25 --render_pose 0 --number_people_max 1 --model_folder C:\Users\Fernando\Documents\GitHub\Openpose-gpu\openpose\models --hand --face --write_json "PREPROCESSING/Pose/{}/{}"'.format(video, sub, file[:-4])
            print("executing ...", command)
            os.system(command)

# TENGO QUE VOLVER A GENERAR LOS VIDEOS. PODRÍA HACER RESIZE AL ARCHIVO ORIGINAL..."""