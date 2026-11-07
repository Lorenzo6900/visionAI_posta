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

from fastapi.responses import HTMLResponse, Response
from ia.herramientas.destino import buscar_destino
from ia.main import GRAFO, Estado, procesar_frame

app = FastAPI()

# Hay una sola raspi y un solo usuario a la vez: un único estado en memoria, sin sesión.
ESTADO = Estado()
# Un pedido a la vez: si llega un frame mientras se procesa otro, espera su turno
# (así dos frames no pisan el estado ni usan YOLO al mismo tiempo).
CANDADO = threading.Lock()

@app.get("/health")
def salud():
    # El hosting del colegio lo consulta para saber si la app arrancó bien
    return {"ok": True}

# Página para ver en vivo lo que ve la IA (se actualiza sola cada segundo).
# "ultima" va sin barra adelante: en el hosting la app vive en /visionai/ia/, y así el navegador
# pide /visionai/ia/ultima.
PAGINA_VER = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionAI en vivo</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font-family: system-ui, sans-serif; }
  header { padding: 8px 12px; display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  img { display: block; width: 100%; max-width: 1280px; margin: 0 auto; }
  small { color: #999; }
</style></head>
<body>
<header><strong>VisionAI en vivo</strong> <span id="estado">esperando la primera foto...</span>
<small>verde = cartel usado, amarillo = con esquinas, rojo = cortado, gris = sin esquinas</small></header>
<img id="foto" alt="Última foto que recibió la IA">
<script>
const foto = document.getElementById("foto"), estado = document.getElementById("estado");
async function actualizar() {
  try {
    const r = await fetch("ultima?t=" + Date.now(), { cache: "no-store" });
    if (r.status === 200) {
      const anterior = foto.src;
      foto.src = URL.createObjectURL(await r.blob());
      if (anterior.startsWith("blob:")) URL.revokeObjectURL(anterior);
      estado.textContent = "actualizado " + new Date().toLocaleTimeString();
    } else {
      estado.textContent = "todavía no llegó ninguna foto";
    }
  } catch (e) {
    estado.textContent = "sin conexión con la IA";
  }
  setTimeout(actualizar, 1000);
}
actualizar();
</script>
</body></html>"""


@app.get("/ver", response_class=HTMLResponse)
def ver():
    return PAGINA_VER


@app.get("/ultima")
def ultima_foto():
    # La última foto que llegó, con lo que encontró la IA dibujado encima
    if ESTADO.ultima_vista is None:
        return Response(status_code=204)
    return Response(ESTADO.ultima_vista, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


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