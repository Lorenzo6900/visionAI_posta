"""
Servidor de la IA (FastAPI). Node le manda los frames de la raspi y lo que dijo la persona;
la IA contesta con el texto que Node pasa a audio. Contrato completo en CLAUDE.md.

Uso (desde la raíz del repo):  uvicorn ia.servidor:app --host 127.0.0.1 --port 8000
"""

import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import time

from ia.herramientas.destino import buscar_destino
from ia.main import GRAFO, Estado, procesar_frame

app = FastAPI()

# Hay una sola raspi y un solo usuario a la vez: un único estado en memoria, sin sesión.
ESTADO = Estado()
# Un pedido a la vez: si llega un frame mientras se procesa otro, espera su turno
# (así dos frames no pisan el estado ni usan YOLO al mismo tiempo).
CANDADO = threading.Lock()


class PedidoDestino(BaseModel):
    texto: str


@app.post("/frame")
def recibir_frame(imagen: UploadFile = File(...)):
    datos = np.frombuffer(imagen.file.read(), dtype=np.uint8)
    frame = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if frame is None:              # no era un JPEG válido: se calla y espera el siguiente
        return {"texto": None, "debug": None}

    with CANDADO:
        texto, debug = procesar_frame(frame, ESTADO)
    return {"texto": texto, "debug": debug}


@app.post("/destino")
def recibir_destino(pedido: PedidoDestino):
    frase, destino_id = buscar_destino(pedido.texto, GRAFO["nodos"])
    with CANDADO:
        ESTADO.callado = False         # la persona volvió a hablar, aunque no se le haya entendido
        if destino_id is not None:     # si no entendió, sigue el destino que tenía
            ESTADO.destino = destino_id
            ESTADO.ultima_instruccion = None       # destino nuevo: en el próximo frame habla seguro
            ESTADO.ultimo_cartel = time.monotonic()  # los 10 s sin cartel se cuentan desde acá
    return {"frase": frase, "destino_id": destino_id}