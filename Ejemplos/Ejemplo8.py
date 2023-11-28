import pandas as pd

data = {
    "Fecha": ["2021-01", "2021-01", "2021-02", "2021-02", "2021-03"],
    "Producto": ["Manzana", "Banana", "Manzana", "Banana", "Manzana"],
    "Region": ["Norte", "Norte", "Sur", "Sur", "Oeste"],
    "Ventas": [100, 75, 90, 90, 110],
}

df = pd.DataFrame(data)
print("DataFrame Original")
print(df)

# Drill Down: Detallar las ventas por mes y producto
print("\nDrill-Down (Ventas por mes y producto)")
drill_down = df.groupby(["Fecha", "Producto"]).sum()
print(drill_down)

# Roll Up: Resumir las ventas por mes
print("\nRoll-Up (Ventas por mes)")
roll_up = df.groupby(["Fecha"]).sum()
print(roll_up)

# Slice: Filtrar las ventas por mes
print("\nSlice (Ventas de Enero)")
slice_op = df[df["Fecha"] == "2021-01"]
print(slice_op)

# Dice: Filtrar las ventas por mes y producto
print("\nDice (Ventas de Manzanas en enero 2021):")
dice_op = df[(df["Fecha"] == "2021-01") & (df["Producto"] == "Manzana")]
print(dice_op)

# Pivote: Cambiar las dimensiones de Fecha y Producto
print("\nPivote (Cambiar dimensiones):")
pivot_op = df.pivot_table(
    values="Ventas", index="Producto", columns="Fecha", aggfunc="sum"
)
print(pivot_op)

# Drill-through: mostrar los datos detallados que componen una suma de ventas
print("\nDrill-Through (Datos que componen la suma de ventas en Enero 2021):")
drill_through = df[(df["Fecha"] == "2021-01")]
print(drill_through)

# Drill-across: esto requiere dos DataFrames,
#   vamos a crear un segundo DataFrame para el ejemplo
data = {
    "Fecha": ["2021-01", "2021-01", "2021-02", "2021-02", "2021-03"],
    "Producto": ["Manzana", "Banana", "Manzana", "Banana", "Manzana"],
    "Ventas": [200, 150, 180, 160, 220],
}
df2 = pd.DataFrame(data)
# Aquí cruzamos datos de Ventas e Inventario para el producto 'Manzana'
print("\nDrill-Across (Cruzar datos de ventas e inventario para 'Manzana'):")
drill_across = pd.merge( df[df['Producto']=='Manzana'],
                         df2[df2['Producto']=='Manzana'],
                         on=['Fecha', 'Producto']
)
print(drill_across)

# Consolidación: Calcular métricas como la suma total, el promedio, etc.
print("\nConsolidación (Suma y Promedio de ventas):")
consolidacion = df['Ventas'].agg(['sum', 'mean'])
print(consolidacion)