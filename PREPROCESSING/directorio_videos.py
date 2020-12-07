import os
import csv

path = 'C:/Users/Fernando/Documents/GitHub/DIRECT-SIGN/PREPROCESSING/Videos/'
directorio = path + 'directorio_videos.csv'

videos = os.listdir(path)

for video in videos:
    if video == "directorio_videos.csv":
        videos.remove(video)

with open(directorio, "w", newline="") as f:
    wr = csv.writer(f, quoting=csv.QUOTE_NONE)
    wr.writerow(videos)
f.close()
print("Directorio creado")