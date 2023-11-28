# Title: Práctica 2 - Visualización y Estructuras Multidimensionales
# Alumno: Daniel Michelle Tovar Ponce
# Boleta: 2021670120
# Grupo: 7CM1

import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Crear listas con valores posibles para cada columna
nombres = [
    "Daniel",
    "Marlene",
    "Cinthia",
    "Karla",
    "Isaac",
    "Fernando",
    "Mich",
    "Francisco",
    "Claudio",
    "Jesus",
]
materias = [
    "Sistemas Operativos",
    "Aplicaciónes Moviles",
    "Programación Web",
    "Diseño de Algoritmos",
    "Programación Orientada a Objetos",
]
semestre = [
    "Primero",
    "Segundo",
    "Tercero",
    "Cuarto",
    "Quinto",
    "Sexto",
    "Séptimo",
    "Octavo",
]

# Inicializar listas vacías para almacenar los datos generados
lista_nombres = []
lista_materias = []
lista_semestre = []
lista_calificación = []

# Generar 500 filas de datos
for _ in range(500):
    nombre = random.choice(nombres)
    materia = random.choice(materias)
    sem = random.choice(semestre)
    calif = random.randrange(0, 11)  # Calificación entre 0 y 100

    # Añadir los datos generados a las listas
    lista_nombres.append(nombre)
    lista_materias.append(materia)
    lista_semestre.append(sem)
    lista_calificación.append(calif)

# Crear un DataFrame de pandas con los datos
df = pd.DataFrame(
    {
        "Nombre": lista_nombres,
        "Materia": lista_materias,
        "Semestre": lista_semestre,
        "Calificación": lista_calificación,
    }
)

# Ver las primeras filas del DataFrame para asegurarnos de que se ha creado correctamente
print(df)

# Opcional: guardar el DataFrame en un archivo CSV
# df.to_csv("df_inventario.csv", index=False)

cubomultidim = (
    df.groupby(["Materia", "Semestre"])[["Calificación"]].mean().reset_index()
)
print(cubomultidim)

sns.barplot(x="Materia", y="Calificación", hue="Semestre", data=cubomultidim)
plt.show()
