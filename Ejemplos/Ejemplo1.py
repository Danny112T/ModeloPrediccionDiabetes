def elegir_planta(temperatura, tipo_suelo):
    if temperatura == "frio":
        if tipo_suelo == "acido":
            return "Rododendro"
        elif tipo_suelo == "neutro":
            return "Arándano"
        else:
            return "Enebro"
    elif temperatura == "moderado":
        if tipo_suelo == "acido":
            return "Camelia"
        elif tipo_suelo == "neutro":
            return "Rosa"
        else:
            return "Lavanda"
    else:
        if tipo_suelo == "acido":
            return "Azalea"
        elif tipo_suelo == "neutro":
            return "Hibisco"
        else:
            return "Olivo"


temperatura = input("¿Cómo es el clima (frio/moderado/caluroso?").lower()
tipo_suelo = input("¿Qué tipo de suelo tienes (acido/neutro/alcaino)?").lower()

planta_a_cultivar = elegir_planta(temperatura, tipo_suelo)

print(f"La mejor planta para que cultives es {planta_a_cultivar}.")
