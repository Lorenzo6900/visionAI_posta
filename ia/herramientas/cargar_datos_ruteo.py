import csv
from collections import defaultdict

from ruteo import buscar_ruta_optima

# Los CSV son los exportados de Neon. Los nodos se identifican por su id (1 a 35).
with open('data/nodos.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    coordenadas = {}
    nombres = {}
    for fila in reader:
        coordenadas[fila['id']] = (float(fila['posicion_x']), float(fila['posicion_y']))
        nombres[fila['nombre']] = fila['id']

with open('data/aristas.csv', mode='r', encoding='utf-8') as file2:
    reader = csv.DictReader(file2)
    vecinos = defaultdict(list)

    for fila in reader:
        origen = fila['nodo_origen_id']
        destino = fila['nodo_destino_id']
        distancia = float(fila['distancia_m'])

        if origen not in coordenadas:
            raise ValueError(f"La arista menciona '{origen}', que no existe en nodos.csv")
        if destino not in coordenadas:
            raise ValueError(f"La arista menciona '{destino}', que no existe en nodos.csv")

        vecinos[origen].append((destino, distancia))
        vecinos[destino].append((origen, distancia))

    for nodo in coordenadas:
        if len(vecinos[nodo]) == 0:
            print("Nodo sin vecinos:", nodo)


por_visitar = [nombres['T']]
visitados = []

while len(por_visitar) > 0:
    nodo = por_visitar.pop()
    visitados.append(nodo)

    for vecino, distancia in vecinos[nodo]:
        if vecino not in visitados and vecino not in por_visitar:
            por_visitar.append(vecino)

for nodo in coordenadas:
    if nodo not in visitados:
        print("Nodo no alcanzable desde T:", nodo)


print(buscar_ruta_optima(vecinos, nombres['escaleras_1'], nombres['oficina_2d0_lib']))
print(buscar_ruta_optima(vecinos, nombres['aula_207'], nombres['lab_2']))
print(buscar_ruta_optima(vecinos, nombres['aula_201'], nombres['lab_3']))
