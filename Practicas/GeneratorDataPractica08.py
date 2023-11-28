import pandas as pd
import random

# Generación del dataset
# Crear listas con valores posibles para cada columna
Edad = [i for i in range(18, 65)] * 42
Sexo = ["Hombre", "Mujer"]
EstadoCivil = ["Soltero", "Casado", "Divorciado", "Viudo"]
NivelEducativo = ["Primaria", "Secundaria", "Preparatoria", "Universidad"]
ExperienciaLaboral = [i for i in range(0, 60)]
Salud = ["Buena", "Mala"]
UsoTecnologia = ["Si", "No"]

# Inicializar listas vacías para almacenar los datos generados
lista_edad = []
lista_ingreso = []
lista_sexo = []
lista_estado_civil = []
lista_nivel_educativo = []
lista_experiencia_laboral = []
lista_salud = []
lista_uso_tecnologia = []

# Generar 500 filas de datos
for _ in range(1000):
    Edad = random.randint(18, 65)
    Ingresos = random.randint(10000, 100000)
    Sexo = random.choice(["Hombre", "Mujer"])
    EstadoCivil = random.choice(["Soltero", "Casado", "Divorciado", "Viudo"])
    NivelEducativo = random.choice(
        ["Primaria", "Secundaria", "Preparatoria", "Universidad"]
    )
    ExperienciaLaboral = random.randint(0, 60)
    Salud = random.choice(["Buena", "Mala"])
    UsoTecnologia = random.choice(["Si", "No"])

    # Añadir los datos generados a las listas
    lista_edad.append(Edad)
    lista_ingreso.append(Ingresos)
    lista_sexo.append(Sexo)
    lista_estado_civil.append(EstadoCivil)
    lista_nivel_educativo.append(NivelEducativo)
    lista_experiencia_laboral.append(ExperienciaLaboral)
    lista_salud.append(Salud)
    lista_uso_tecnologia.append(UsoTecnologia)

# Crear un DataFrame de pandas con los datos
df = pd.DataFrame(
    {
        "Edad": lista_edad,
        "Ingresos": lista_ingreso,
        "Sexo": lista_sexo,
        "Estado Civil": lista_estado_civil,
        "Nivel Educativo": lista_nivel_educativo,
        "Experiencia Laboral": lista_experiencia_laboral,
        "Salud": lista_salud,
        "Uso Tecnologia": lista_uso_tecnologia,
    }
)

# Ver las primeras filas del DataFrame para asegurarnos de que se ha creado correctamente
print(df.head())

# Opcional: guardar el DataFrame en un archivo CSV
df.to_csv("df_UsoTech.csv", index=False)
