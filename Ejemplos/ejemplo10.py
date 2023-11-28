import pandas as pd
import numpy as np
from datetime import datetime

#Datos Categóricos
categorias = pd.Series(['rojo', 'verde', 'azul', 'rojo'],dtype='category')
print("Datos Categóricos")
print(categorias)

#Datos ordinales
ordinales = pd.Categorical(['bajo','medio','alto','medio'],categories=['bajo','medio','alto'],ordered=True)
print("Datos Ordinales")
print(ordinales)

#Datos Numericos
#continuos
continuos = pd.Series([25.5, 30.2, 35.8, 40.1])
print("Datos Continuos")
print(continuos)

#discretos
discretos = pd.Series([2, 3, 5, 7])
print("Datos Discretos")
print(discretos)

#Datos Temporales
fechas = pd.Series([datetime(2023,1,1),datetime(2023,1,2),datetime(2023,1,3)])
print("Datos Temporales")
print(fechas)
print()

#datos de texto
Textos = pd.Series(['Hola mundo','Mineria de datos en python','Tipos de datos'])
print("Datos de Texto")
print(Textos)

# Datos multidimensionales
data = {
    'categorias': ['A','B','C','D'],
    'numericas': [10,20,30,40]
}
df = pd.DataFrame(data)
print("Datos Multidimensionales")
print(df)