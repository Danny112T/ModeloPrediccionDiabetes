import pandas as pd  # Se importa la libería pandas para poder trabajar con DataFrames

# Se leen los datos de los archivos csv
df = pd.read_csv("datos_ventas.csv")
df_inventario = pd.read_csv("df_inventario.csv")

# 1. Drill-Down
dd = df.groupby(["Fecha", "Categoría", "Producto"])["Ventas"].sum().reset_index()
# Se agrupa por fecha, categoría y producto y se suman las ventas
print("1. Drill-Down Result:")  # Se imprime el resultado
print(dd)

# 2. Roll-Up
ventas_region = (
    df.groupby("Municipio")["Ventas"].sum().reset_index()
)  # Se agrupa por municipio y se suman las ventas
ventas_totales = ventas_region["Ventas"].sum()  # Se suman las ventas totales
print("\n2. Roll-Up Result:")
print(ventas_region)  # Se imprime el resultado
print(f"Ventas Totales: {ventas_totales}")

# 3. Slice and Dice
slice_result = df[
    (df["Fecha"].str.startswith("2021-01")) & (df["Municipio"] == "Zacatecas")
]
# Se obtienen las ventas del municipio de Zacatecas en la fecha de enero de 2021
dice_result = slice_result[
    slice_result["Categoría"] == "Frutas"
]  # Se filtra por categoría
print("\n3. Slice and Dice Result:")  # Se imprime el resultado
print(dice_result)

# 4. Pivot (o Rotate)
pivot_result = pd.pivot_table(
    df,
    values="Ventas",
    index=["Vendedor"],
    columns=["Fecha"],
    aggfunc=sum,
    fill_value=0,
)
# Se crea una tabla pivote con los vendedores como renglones y las fechas como columnas
print("\n4. Pivot Result:")  # Se imprime el resultado
print(pivot_result)

# 5. Drill-Through
total_ventas_frutas_2021_01 = df[
    (df["Fecha"].str.startswith("2021-01")) & (df["Categoría"] == "Frutas")
]["Ventas"].sum()
# Se obtienen las ventas totales de frutas en enero de 2021
drill_through_result = df[
    (df["Fecha"].str.startswith("2021-01")) & (df["Categoría"] == "Frutas")
]  # Se obtienen las ventas de frutas en enero de 2021
print("\n5. Drill-Through Result:")  # Se imprime el resultado
print(drill_through_result)

# 6. Drill-Across (Simulación ya que no tenemos un DataFrame real de inventario)
# El df_inventario es el otro conjunto de datos.
drill_across_result = pd.merge(
    df, df_inventario, on=["Producto", "Fecha"]
)  # Se unen los dos DataFrames por producto y fecha
print("\n6. Drill-Across Result:")  # Se imprime el resultado
print(drill_across_result)


# 7. Consolidación
consolidation_result = (
    df.groupby("Vendedor")
    .agg({"Ventas": ["sum", "mean", "max"], "UnidadesVendidas": ["sum", "mean", "max"]})
    .reset_index()
)
# Se agrupa por vendedor y se obtienen las ventas y unidades vendidas totales, promedio y máximo
print("\n7. Consolidación Result:")  # Se imprime el resultado
print(consolidation_result)
