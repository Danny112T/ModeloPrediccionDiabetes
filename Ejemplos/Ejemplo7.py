#Ejemplo Saas
import requests 
import json

def get_github_repo_info(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"Nombre del repositorio: {data['name']}")
        print(f"Descripción: {data['description']}")
        print(f"Estrellas; {data['stargazers_count']}")
        print(f"Lenguaje principal: {data['language']}")
    else:
        print(f"No se pudo obtener informacion para el repositorio {owner}/{repo}")

#Ejemplo de uso
get_github_repo_info("mouredev", "retos-programacion-2023")
