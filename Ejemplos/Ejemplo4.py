# Inicializamos un diccionario vacio para actuar como nuestro cubo de datos
cubo_de_datos = {}

# funcion para añadir datos al cubo
def añadir_datos(tiempo, ubicacion, producto, ventas):
    if tiempo not in cubo_de_datos:
        cubo_de_datos[tiempo] = {}
    if ubicacion not in cubo_de_datos[tiempo]:
        cubo_de_datos[tiempo][ubicacion] = {}
    cubo_de_datos[tiempo][ubicacion][producto] = ventas

# funcion para mostrar el cubo de datos
def mostrar_cubo():
    for tiempo, datos_tiempo in cubo_de_datos.items():
        print(f"Año: {tiempo}")
        for ubicacion, datos_ubicacion in datos_tiempo.items():
            print(f"    ubicación: {ubicacion}")
            for producto, ventas in datos_ubicacion.items():
                print(f"    Producto: {producto}, Ventas Totales: {ventas}")

# Interaccion con el usuario para llenar el cubo de datos
while True:
    print("---------- LLenar cubo de datos ---------- ")

    tiempo = input("Ingrese el año (por ejemplo, 2023): \n")
    ubicacion = input("Ingrese la ubicacion (por ejemplo, NY): \n")
    producto = input("Ingrese el tipo de producto (por ejemplo, Televisor): \n")
    ventas = int(input("Ingrese la cantidad total de ventas: \n"))

    añadir_datos(tiempo, ubicacion, producto, ventas)
    mostrar_cubo()

    continuar = input("¿Desea continuar añadiendo datos? (s/n): \n")
    if continuar.lower() != 's':
        break