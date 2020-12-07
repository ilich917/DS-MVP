from csv_reader import leer_csv

file1 = 'csvs/MesesLSM.csv'
file2 = 'Videos/directorio_videos.csv'

directorio = leer_csv(file2)
init = directorio[0].

###
lector = leer_csv(init)

for video in directorio:
    lector.__init__(video)
    lector.quitar_extension()
print(anotacion)

