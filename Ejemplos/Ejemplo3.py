cubo_de_datos = {
    "2023": {
        "New York": {
            "Televisor": 5000,
            "Telefono": 3000,
        },
        "San Francisco": {
            "Televisor": 4000,
            "Telefono": 3500,
        },
    },
    "2022": {
        "New York": {
            "Televisor": 4500,
            "Telefono": 2900,
        },
        "San Francisco": {
            "Televisor": 3900,
            "Telefono": 3400,
        },
    },
}


# funcion para añadir datos al cubo
def añadir_datos(tiempo, ubicacion, producto, ventas):
    if tiempo not in cubo_de_datos:
        cubo_de_datos[tiempo] = {}
    if ubicacion not in cubo_de_datos[tiempo]:
        cubo_de_datos[tiempo][ubicacion] = {}
    cubo_de_datos[tiempo][ubicacion][producto] = ventas


# Funcion para consultar el cubo de datos
def consultar_ventas(tiempo, ubicacion, producto):
    try:
        return cubo_de_datos[tiempo][ubicacion][producto]
    except KeyError:
        return "Información no disponible"


# Permitir al usuario añadir datos al cubo
tiempo = input("ingrese el año (por Ej, 2023): ")
ubicacion = input("ingrese la ubicacion (por Ej, New York): ")
producto = input("ingrese el producto (por Ej, Televisor): ")
ventas = int(input("ingrese las ventas (por Ej, 5000): "))
añadir_datos(tiempo, ubicacion, producto, ventas)

# Permitir al usuario hacer una consulta
tiempo_consulta = input("ingrese el año (por Ej, 2023): ")
ubicacion_consulta = input("ingrese la ubicacion (por Ej, New York): ")
producto_consulta = input("ingrese el producto (por Ej, Televisor): ")
print(
    "Ventas Totales: $",
    consultar_ventas(tiempo_consulta, ubicacion_consulta, producto_consulta),
)
