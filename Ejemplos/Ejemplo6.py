import pandas as pd

#cargar los datos desde un archivo CSV
df = pd.read_csv("data_ejemplo06.csv")

# Ver los primeros datos del DataFrame
print("Datos Originales")
print(df)
print()

#Agregar ventas por producto(Simulando un cubo OLAP)
print("Ventas totales por producto: ")
ventas_por_producto = df.groupby('Producto')['Ventas'].sum()
print(ventas_por_producto)
print()

#Agregar ventas por region
print("Ventas totales por región: ")
ventas_por_region = df.groupby('Región')['Ventas'].sum()
print(ventas_por_region)
print()

#Agregar ventas por producto y region
print("Ventas totales por producto y region: ")
ventas_por_producto_y_region = df.groupby(['Producto','Región'])['Ventas'].sum().unstack()
print(ventas_por_producto_y_region)
