import pandas as pd
import random
from datetime import datetime

# Crear listas con valores posibles para cada columna
fechas = [f"2021-{str(month).zfill(2)}" for month in range(1, 13)] * 42 
productos = ['Manzana', 'Leche', 'Pan', 'Platano', 'Jugo de Naranja']
categorias = ['Frutas', 'Lácteos', 'Panadería', 'Frutas', 'Bebidas']
municipios = ['Zacatecas', 'Guadalupe', 'Fresnillo','Tlaltenango','Jerez']
vendedores = ['Ana', 'Héctor', 'Francisco', 'Nora','Clauido']

# Inicializar listas vacías para almacenar los datos generados
lista_fechas = []
lista_productos = []
lista_categorias = []
lista_mun = []
lista_vendedores = []
lista_ventas = []
lista_unidades = []

# Generar 500 filas de datos
for _ in range(500):
    fecha = random.choice(fechas)
    producto = random.choice(productos)
    categoria = categorias[productos.index(producto)]  
    municipio = random.choice(municipios)
    vendedor = random.choice(vendedores)
    ventas = random.randint(50, 200)  # Ventas en dólares, entre 50 y 200
    unidades = random.randint(20, 100)  # Unidades vendidas, entre 20 y 100
    
    # Añadir los datos generados a las listas
    lista_fechas.append(fecha)
    lista_productos.append(producto)
    lista_categorias.append(categoria)
    lista_mun.append(municipio)
    lista_vendedores.append(vendedor)
    lista_ventas.append(ventas)
    lista_unidades.append(unidades)

# Crear un DataFrame de pandas con los datos
df = pd.DataFrame({
    'Fecha': lista_fechas,
    'Producto': lista_productos,
    'Categoría': lista_categorias,
    'Municipio': lista_mun,
    'Vendedor': lista_vendedores,
    'Ventas': lista_ventas,
    'UnidadesVendidas': lista_unidades
})

# Ver las primeras filas del DataFrame para asegurarnos de que se ha creado correctamente
print(df.head())

# Opcional: guardar el DataFrame en un archivo CSV
df.to_csv('df_inventario.csv', index=False)