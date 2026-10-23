import heapq

#djkistra
# Grafo del mapa de rutas
grafo = {
    'A': [('B', 2), ('D', 7)],
    'B': [('A', 2), ('C', 2)],
    'C': [('B', 2), ('D', 2)],
    'D': [('C', 2), ('A', 7)],
    'Z': []
}

def buscar_ruta_optima(grafo, inicio, destino):

    if inicio not in grafo:
        raise ValueError(f"El nodo de inicio '{inicio}' no existe en el grafo")

    elif destino not in grafo:
        raise ValueError(f"El nodo destino '{destino}' no existe en el grafo")

    distancias = {nodo: float('inf') for nodo in grafo}
    distancias[inicio] = 0
    
    # DICCIONARIO CLAVE: Guarda de dónde vinimos (nodo_hijo: nodo_padre)
    padres = {nodo: None for nodo in grafo}
    
    cola_prioridad = [(0, inicio)]
    
    while cola_prioridad:
        distancia_actual, nodo_actual = heapq.heappop(cola_prioridad)
        
        if nodo_actual == destino:
            break  # Ya encontramos la ruta óptima al destino
            
        if distancia_actual > distancias[nodo_actual]:
            continue
            
        for vecino, peso_arista in grafo[nodo_actual]:
            distancia_nueva = distancia_actual + peso_arista
            
            if distancia_nueva < distancias[vecino]:
                distancias[vecino] = distancia_nueva
                padres[vecino] = nodo_actual # Registramos el camino
                heapq.heappush(cola_prioridad, (distancia_nueva, vecino))

    if distancias[destino] == float('inf'):
        return [], distancias[destino]
                
    # RECONSTRUCCIÓN DEL CAMINO (Desde el destino hacia atrás hasta el inicio)
    ruta = []
    nodo_actual = destino
    while nodo_actual is not None:
        ruta.append(nodo_actual)
        nodo_actual = padres[nodo_actual]
    
    ruta.reverse() # Invertimos para que vaya de Inicio a Destino
    return ruta, distancias[destino]


if __name__ == "__main__":
    inicio = 'A'
    destino = 'D'
    ruta_exacta, distancia_total = buscar_ruta_optima(grafo, inicio, destino)

    if distancia_total == float('inf'):
        print("No hay camino posible hasta ese destino.")

    else:
        print(f"La mejor ruta es: {' -> '.join(ruta_exacta)}")
        print(f"Distancia total recorrida: {distancia_total} metros.")
