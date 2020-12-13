"""
Script para generar los archivos json desde un set de videos
Editar los path
"""

import os
dataset_path = 'Bible_j316_new'

for sub in os.listdir(dataset_path):
    for file in os.listdir(dataset_path + '/' + sub):
        for f in os.listdir(dataset_path+'/'+sub+'/'+file):
            # print(f)
            video = dataset_path + '/' + sub + '/' + file + '/' + f
            #command = 'bin\OpenPoseDemo.exe --video {} --hand --write_json ./output/{}'.format(video, sub)
            command = r'bin\OpenPoseDemo.exe' --video "{}" --model_ppose COCO --hand --write_json "./new_bible_out/{}{}"'.format(video, sub, file)
            print("executing ...", command)
            os.system(command)