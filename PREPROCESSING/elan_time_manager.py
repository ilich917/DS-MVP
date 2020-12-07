import csv
from datetime import timedelta

from csv_reader import leer_csv

class elan_time_manager():
    """
    Esta clase recibe un csv crudo y convierte el string de tiempo
    ["hh:mm:ss.ms"] a un formato lista [h, min, s, ms] 
    de variables enteras, separado en t_inicial y t_final para 
    cada seña anotada en el video correspondiente al csv
    """
    
    def __init__(self, file):
        self.file = file
        self.delta = timedelta()
        self.lector = leer_csv(self.file)
        self.anotacion = self.lector.abrir()
        self.tiempo_i = []
        self.tiempo_f = []

    def to_int(self,lista_string):
        lista_int = []
        for elem in lista_string:
            lista_int.append(int(elem))
        return lista_int

    def separar_string(self, string):
        tiempo = list(str(string).split(':'))
        foo = tiempo[2].split('.')
        
        tiempo.pop(2)
        tiempo.append(foo[0])
        tiempo.append(foo[1])
        
        tiempo = self.to_int(tiempo)
        
        return tiempo

    def main(self):
        c=0
        for row in self.anotacion:
                self.tiempo_i.append(self.separar_string(row[0]))
                self.tiempo_f.append(self.separar_string(row[1]))
                #print(self.tiempo_f[c][:])
                c+=1
        return self.tiempo_i, self.tiempo_f