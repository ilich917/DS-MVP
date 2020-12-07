import csv
import numpy as np

import cv2

# Una clase que toma como argumentos un haarcascade, un archivo log y un video, luego devuelve un array
# con el resultado (coordenadas de un cuadrado con la cara):


class Detector():
    """
    Uso: 
    detect = Detector(video, face_cascade, trackFile)
    caras = detect.detectar() 
    """
    
    def __init__(self, video, detector, track_file):
        self.video = video
        self.detector = detector
        self.total_frames = int(video.get(7))
        self.file = track_file
        self.tracker = np.zeros(shape=(self.total_frames, 300, 300))

    def detectar(self):
        
        with open(self.file, "w", newline="") as f:
            wr = csv.writer(f, quoting = csv.QUOTE_NONNUMERIC)
            c = 0
            while c < self.total_frames:
                
                _, frame = self.video.read()
                time = self.video.get(cv2.CAP_PROP_POS_MSEC)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Ahora usa el detector
                faces = self.detector.detectMultiScale(gray, 1.1, 4, minSize=(150,150))

                
                for (x, y, w, h) in faces:
                    linea = map(float,[time, 0,  x, y, x+w, y+h])
                    wr.writerow(list(linea))
                    crop = gray[y:y+h,x:x+w]
                    resized = cv2.resize(crop,(300,300))
                   # print(np.array(crop).shape)
                    self.tracker[c] = resized.reshape(1,300,300)
                    
                
                c += 1   
                if c % 100== 0:
                    print('frame: ', c)
                
        f.close()
        return self.tracker