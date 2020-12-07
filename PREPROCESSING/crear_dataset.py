"""
formato final csv:

Se puede crear un dataset con frames espaciados uniformemente, entonces, 
asignar null para todos los campos excepto para aquellos en que se encuentre una seña. En ese caso, puede ser esto:

-> Crear temporal-dataset uniformemente distribuido y vacío para cada video, de longitud ceiling la longitud del video (en hh:mm:ss:ms ? o solo msec)
-> ir rellenando las filas (cada frame)

Qué pasa con el tiempo de los frames?

-> Se puede hacer esto: # es necesario??? sí, porque en realidad ahora no tengo un dataset.

anotacion = csv elan depurado
tdataset = [] #vacio, índice temporal
for glosa in anotacion:
    for fila in tdataset:
        while fila.index < glosa[1]:
            if fila.index > glosa[0]:
                fila[:] = glosa[2:]    # rellena la fila con los datos de la anotación



# probando cómo abrir más de un archivo a la vez. No me funcionó. Será mejor crear muchos
with fileinput.input(files=('textfile1.txt',
    'textfile2.txt')) as f:
    line = f.readline()
    print('f.filename():', f.filename())
    print('f.isfirstline():', f.isfirstline())
    print('f.lineno():', f.lineno())
    print('f.filelineno():', f.filelineno())
    for line in f:
    print(line, end='')
    
"""

## crear dataset temporal vacío

import pandas as import pd

for filename in directorio:
    with open(filename) as f:
        for chunk in pd.read_csv(f, chunksize=chunksize):
            process(chunk)


