cubo_de_datos = {
    "2023": {
        "New York": {
            "Televisor":5000,
            "Telefono":3000,
        },
        "San Francisco": {
            "Televisor":4000,
            "Telefono":3500,
        },
    },
    "2022": {
        "New York": {
            "Televisor":4500,
            "Telefono":2900,
        },
        "San Francisco": {
            "Televisor":3900,
            "Telefono":3400,
        },
    },
}

def consultar_ventas(tiempo, ubicaciones, productos):
    try:
        return cubo_de_datos[tiempo][ubicaciones][productos]
    except KeyError:
        return "Información no disponible"
    
print("Ventas de televisores en NY en 2023: $", consultar_ventas("2023", "New York", "Televisor"))