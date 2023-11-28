# pip install pandas

import pandas as pd

# Suponemos que tenemos un conjunto de datos de ventas en un DataFrame
data = {
    "Fecha": ["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"],
    "Producto": ["Manzana", "Banana", "Manzana", "Banana"],
    "Ciudad": ["New York", "New York", "Chicago", "Chicago"],
    "Ventas": [100, 150, 200, 50],
}

df = pd.DataFrame(data)

# Pivot para crear un cubo simple (Suma de ventas por fecha y producto)
cube = pd.pivot_table(
    df, values="Ventas", index="Fecha", columns="Producto", aggfunc="sum"
)

print("Cubo Simple: ")
print(cube)

# Se puede realizar un análisis más complejo agregando mas dimensiones.
# por ejemplo, podriamos querer saber las ventas por ciudad y producto

cube_multi_dimension = pd.pivot_table(
    df, values="Ventas", index=["Fecha", "Ciudad"], columns="Producto", aggfunc="sum"
)

print("\nCubo multi-dimension:")
print(cube_multi_dimension)

# podriamos querer "rodar" (roll-up) el cubo para tener las ventas por producto
cube_rollup = pd.pivot_table(df, values="Ventas", columns="Producto", aggfunc="sum")

print("\nCubo rodado (Roll-up):")
print(cube_rollup)
