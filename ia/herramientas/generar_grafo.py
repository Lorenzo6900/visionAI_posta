"""
Genera data/grafo.json a partir de las 3 tablas exportadas de Neon
(data/nodos.csv, data/aristas.csv, data/marcadores.csv).

Uso (desde la raíz del repo):  python ia/herramientas/generar_grafo.py
"""

import csv
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data"

# Aristas que todavía no están en Neon: los gemelos lab_1 (19) y aula_214 (29)
# quedaron sin conectar. Copian las aristas de su gemelo (sala_de_prof 20, aula_215 28).
# Cuando se carguen en Neon y se vuelva a exportar, esta lista se vacía.
ARISTAS_FALTANTES = [
    (33, 19, 12.00),
    (19, 21, 7.88),
    (19, 22, 7.88),
    (26, 29, 6.60),
    (29, 30, 3.00),
    (29, 31, 3.00),
]

# Código de cada cartel según la hoja carteles_ivan (las carpetas de la galería de embeddings)
CODIGOS = {
    "impacto_social": "M01", "dispo_moviles": "M02", "tecno_satelital": "M03",
    "int_de_las_cosas": "M04", "cartel_tic_vr": "M05", "RV": "M06", "impresion_3d": "M07",
    "IA": "M08", "arte_digital": "M09", "videojuegos": "M10", "hardware": "M11",
    "apps": "M12", "albert_einstein": "M13", "frontal": "M14", "horarios_ort": "M15",
    "visionai_blanco": "M16", "human_centric_data": "M17", "ai_neural_networks": "M18",
    "enfrente_a_horarios": "M19",
}


def leer(nombre):
    with open(DATA / nombre, encoding="utf-8-sig") as archivo:
        return list(csv.DictReader(archivo))


nodos = []
for fila in leer("nodos.csv"):
    nodos.append({
        "id": int(fila["id"]),
        "numero": fila["numero"],               # el N viejo de la planilla
        "nombre": fila["nombre"],
        "x": float(fila["posicion_x"]),
        "y": float(fila["posicion_y"]),
        "pared": fila["pared"],                 # de qué lado está la puerta
        "es_destino": fila["es_destino_final"] == "true",
    })
ids = {n["id"] for n in nodos}

aristas = [
    {"origen": int(f["nodo_origen_id"]), "destino": int(f["nodo_destino_id"]), "distancia": float(f["distancia_m"])}
    for f in leer("aristas.csv")
]
existentes = {frozenset((a["origen"], a["destino"])) for a in aristas}
for origen, destino, distancia in ARISTAS_FALTANTES:
    if frozenset((origen, destino)) not in existentes:
        aristas.append({"origen": origen, "destino": destino, "distancia": distancia})

# Un cartel físico puede estar repetido en Neon (una fila por cada nodo gemelo): se agrupa por imagen
carteles = {}
for fila in leer("marcadores.csv"):
    nombre = fila["imagen_referencia_url"]
    if nombre not in carteles:
        carteles[nombre] = {
            "nombre": nombre,
            "codigo": CODIGOS.get(nombre),
            "ancho_m": float(fila["ancho_cm"]) / 100,
            "alto_m": float(fila["alto_cm"]) / 100,
            "grados": float(fila["grados"]),    # hacia dónde mira el frente: este=0, norte=90, oeste=180, sur=270
            "pared": fila["pared"],
            "x": float(fila["pos_x"]),
            "y": float(fila["pos_y"]),
            "nodos": [],
            "marcador_ids": [],
        }
    carteles[nombre]["nodos"].append(int(fila["nodo_id"]))
    carteles[nombre]["marcador_ids"].append(int(fila["id"]))

# Chequeos
for a in aristas:
    assert a["origen"] in ids and a["destino"] in ids, f"arista con nodo inexistente: {a}"
for c in carteles.values():
    assert all(n in ids for n in c["nodos"]), f"cartel con nodo inexistente: {c['nombre']}"
    assert c["codigo"], f"cartel sin código M: {c['nombre']}"

conectados = {a["origen"] for a in aristas} | {a["destino"] for a in aristas}
for n in nodos:
    if n["id"] not in conectados:
        print(f"Aviso: el nodo {n['id']} ({n['nombre']}) no tiene aristas")

with open(DATA / "grafo.json", "w", encoding="utf-8") as archivo:
    json.dump({"nodos": nodos, "aristas": aristas, "carteles": list(carteles.values())},
              archivo, indent=2, ensure_ascii=False)

print(f"grafo.json: {len(nodos)} nodos, {len(aristas)} aristas, {len(carteles)} carteles")
