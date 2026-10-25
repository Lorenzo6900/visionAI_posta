"""
Pipeline de VisionAI en la raspi (~1 fps):
    cámara -> detectar -> identificar -> posicion -> ruta -> decidir -> voz

detectar(), ruta() y decidir() ya son reales. capturar, identificar y posicion todavía
devuelven datos falsos; su lógica está en ia/herramientas/ y se va conectando de a una pieza.

Uso (desde la raíz del repo):  python -m ia.main
"""

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO as ModeloYOLO

from ia.herramientas.ruteo import buscar_ruta_optima
from voz.cliente import hablar

RAIZ = Path(__file__).resolve().parent.parent
RUTA_GRAFO = RAIZ / "data" / "grafo.json"
RUTA_YOLO = RAIZ / "models" / "best.pt"

UMBRAL_DETECCION = 0.5   # confianza mínima de YOLO para aceptar un cartel
UMBRAL_NMS = 0.45        # cajas que se pisan más que esto se consideran el mismo cartel
CONFIANZA_MINIMA = 0.6   # de si hay o no confianza de que se sabe donde esta la persona.
REPETIR_CADA_S = 5.0     #esto para no repetir la misma instruccion x cada frame.
LLEGADA_M = 1.0          # a menos de esto de un nodo, se considera que ya llegó

# Ángulos: este=0, norte=90, oeste=180, sur=270 (x crece al este, y al norte).
# Positivo = antihorario, o sea que girar a la izquierda suma grados.
PARED_A_GRADOS = {"este": 0, "norte": 90, "oeste": 180, "sur": 270}


# ---------- Tipos ----------

@dataclass
class Deteccion:
    caja: tuple[int, int, int, int]     # x1, y1, x2, y2 en píxeles
    esquinas: np.ndarray                # 4x2: arriba-izq, arriba-der, abajo-der, abajo-izq
    confianza: float


@dataclass
class Posicion:
    x: float                            # dónde está la persona en el mapa (metros)
    y: float
    rumbo_grados: float                 # hacia dónde mira la cámara


@dataclass
class Estado:
    ultima_instruccion: tuple | None = None   # (acción, nodo objetivo) de lo último que se dijo
    ultima_vez: float = 0.0
    camino: list[int] = field(default_factory=list)


# ---------- Datos ----------

def cargar_grafo(ruta: Path = RUTA_GRAFO) -> dict:
    with open(ruta, encoding="utf-8") as archivo:
        datos = json.load(archivo)

    vecinos = {nodo["id"]: [] for nodo in datos["nodos"]}
    for arista in datos["aristas"]:
        vecinos[arista["origen"]].append((arista["destino"], arista["distancia"]))
        vecinos[arista["destino"]].append((arista["origen"], arista["distancia"]))

    return {
        "nodos": {n["id"]: n for n in datos["nodos"]},          # id de Neon -> nodo
        "vecinos": vecinos,
        "carteles": {c["nombre"]: c for c in datos["carteles"]},  # nombre de imagen -> cartel
    }


GRAFO = cargar_grafo()
YOLO = ModeloYOLO(RUTA_YOLO)


# ---------- Piezas del pipeline ----------

def capturar() -> np.ndarray:
    # TODO: Picamera2 a 1280x720 (ver herramientas/calibracioncamara.py)
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def detectar(frame: np.ndarray) -> list[Deteccion]:
    resultado = YOLO(frame, conf=UMBRAL_DETECCION, iou=UMBRAL_NMS, verbose=False)[0]

    detecciones = []
    for caja, confianza in zip(resultado.boxes.xyxy.tolist(), resultado.boxes.conf.tolist()):
        x1, y1, x2, y2 = (int(v) for v in caja)
        # TODO (Ivan): esquinas reales del cartel. Por ahora, las de la caja.
        esquinas = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        detecciones.append(Deteccion(caja=(x1, y1, x2, y2), esquinas=esquinas, confianza=confianza))
    return detecciones


def identificar(recorte: np.ndarray) -> tuple[str | None, float]:
    """Devuelve (nombre del cartel, confianza). El nombre es la clave de GRAFO["carteles"];
    es None si no se reconoce el cartel."""
    # TODO: embedding + galería (ver herramientas/compararEmbeddings.py)
    return "impacto_social", 0.8


def posicion(esquinas: np.ndarray, cartel: dict) -> Posicion:
    # TODO: solvePnP con ancho_m, alto_m, x, y y grados del cartel (ver herramientas/solvePNP.py)
    # Por ahora: parado a 2 m justo enfrente del cartel, mirándolo.
    frente = math.radians(cartel["grados"])
    return Posicion(
        x=cartel["x"] + 2.0 * math.cos(frente),
        y=cartel["y"] + 2.0 * math.sin(frente),
        rumbo_grados=(cartel["grados"] + 180) % 360,
    )


def ruta(nodo: int, destino: int) -> list[int]:
    """Camino más corto de nodo a destino (lista de ids). Vacía si no hay camino."""
    # Los gemelos (otro lado del pasillo) están en el mismo punto: se puede salir de cualquiera
    actual = GRAFO["nodos"][nodo]
    salidas = [n["id"] for n in GRAFO["nodos"].values() if (n["x"], n["y"]) == (actual["x"], actual["y"])]

    mejor_camino, mejor_distancia = [], math.inf
    for salida in salidas:
        camino, distancia = buscar_ruta_optima(GRAFO["vecinos"], salida, destino)
        if distancia < mejor_distancia:
            mejor_camino, mejor_distancia = camino, distancia
    return mejor_camino


def rumbo_hacia(pos: Posicion, nodo: dict) -> float:
    return math.degrees(math.atan2(nodo["y"] - pos.y, nodo["x"] - pos.x)) % 360


def diferencia(angulo: float, referencia: float) -> float:
    """Cuánto hay que girar desde referencia para mirar a angulo: -180..180, positivo = izquierda."""
    return (angulo - referencia + 180) % 360 - 180


def accion_de_giro(giro: float) -> str:
    if abs(giro) < 25:
        return "Seguí derecho"
    if abs(giro) > 155:
        return "Date la vuelta y avanzá"
    return "Girá a la izquierda y avanzá" if giro > 0 else "Girá a la derecha y avanzá"


def lado_de_la_puerta(destino: dict, pos: Posicion) -> str:
    if destino["pared"] not in PARED_A_GRADOS:
        return ""
    giro = diferencia(PARED_A_GRADOS[destino["pared"]], pos.rumbo_grados)
    if abs(giro) < 45:
        return " La puerta está adelante."
    if abs(giro) > 135:
        return " La puerta está atrás tuyo."
    return " La puerta está a tu izquierda." if giro > 0 else " La puerta está a tu derecha."


def decidir(camino: list[int], pos: Posicion, confianza: float, estado: Estado,
            ahora: float | None = None) -> str | None:
    """Devuelve el texto a decir, o None si conviene quedarse callado.
    Habla si la instrucción cambió; si no cambió, la repite como máximo cada REPETIR_CADA_S."""
    ahora = time.monotonic() if ahora is None else ahora

    if confianza < CONFIANZA_MINIMA:
        instruccion = ("no_seguro", None)
        texto = "No estoy seguro de dónde estás. Mové la cabeza despacio."
    elif not camino:
        instruccion = ("sin_camino", None)
        texto = "No encuentro un camino hasta el destino."
    else:
        # El objetivo es el primer nodo del camino que todavía está lejos
        nodos = [GRAFO["nodos"][n] for n in (camino[1:] or camino)]
        lejos = [n for n in nodos if math.dist((pos.x, pos.y), (n["x"], n["y"])) > LLEGADA_M]
        if not lejos:
            destino = GRAFO["nodos"][camino[-1]]
            instruccion = ("llegada", destino["id"])
            texto = f"Llegaste a {destino['nombre'].replace('_', ' ')}." + lado_de_la_puerta(destino, pos)
        else:
            objetivo = lejos[0]
            accion = accion_de_giro(diferencia(rumbo_hacia(pos, objetivo), pos.rumbo_grados))
            metros = max(1, round(math.dist((pos.x, pos.y), (objetivo["x"], objetivo["y"]))))
            instruccion = (accion, objetivo["id"])
            texto = f"{accion} {metros} metros."

    if instruccion == estado.ultima_instruccion and ahora - estado.ultima_vez < REPETIR_CADA_S:
        return None
    estado.ultima_instruccion = instruccion
    estado.ultima_vez = ahora
    return texto


def recortar(frame: np.ndarray, caja: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = caja
    return frame[y1:y2, x1:x2]


# ---------- Bucle principal ----------

def procesar_frame(frame: np.ndarray, destino: int, estado: Estado) -> str | None:
    detecciones = detectar(frame)
    if not detecciones:
        return None

    mejor = max(detecciones, key=lambda d: d.confianza)
    nombre, confianza = identificar(recortar(frame, mejor.caja))
    if nombre is None:
        return None

    cartel = GRAFO["carteles"][nombre]
    nodo = cartel["nodos"][0]      # si el cartel está en dos nodos gemelos, están en el mismo punto
    pos = posicion(mejor.esquinas, cartel)
    estado.camino = ruta(nodo, destino)
    return decidir(estado.camino, pos, min(mejor.confianza, confianza), estado)


def main(destino: int = 12, cantidad_frames: int = 3) -> None:
    print(f"Grafo: {len(GRAFO['nodos'])} nodos, {len(GRAFO['carteles'])} carteles")
    estado = Estado()

    for _ in range(cantidad_frames):
        inicio = time.perf_counter()
        texto = procesar_frame(capturar(), destino, estado)
        if texto:
            hablar(texto)
        print(f"frame procesado en {(time.perf_counter() - inicio) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
