import csv

class leer_csv():
    """
    Esta clase recibe el path de un .csv y devuelve un objeto array con 
    los valores leídos.
    
    Uso: 
    
    lector = leer_csv(archivo.csv)
    archivo = lector.abrir()
    """
    
    def __init__(self, archivo_csv):
        self.csv_ = []
        self.archivo = archivo_csv
        self.titulo = []
        
    def no_es_head(self, row):
        
        if '#' in str(row):
            return False
        elif 'hh:mm:ss.ms' in str(row):
            return False
        else:
            return True
        
           
    def __iter__(self):
        
        with open(self.archivo, 'r', newline="", encoding="utf-8") as file:
            
            reader = csv.reader(file, dialect='excel', delimiter=',')
    
            for row in reader:
                c=0
                if self.no_es_head(row):    #debo optimizar esto más adelante
                    self.csv_.append(row)
                    c+=1
                    
        file.close()
        return self.csv_
    
    def __getitem__(self):
        pass
        
    
    def quitar_extension(self):
        """
        solo debería funcionar y usarse para el directorio de videos
        puede que sea necesario migrar esta función a ese otro archivo
        pero por ahora me acomoda esto
        """
        
        for row in self.csv_:
            print(row)
            #separado = '.'.split(row)
