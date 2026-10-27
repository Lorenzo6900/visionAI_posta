"""
Marca a mano las 4 esquinas de los carteles que el recorte automático no encontró o eligió mal.
Lee <galeria>_recortes/a_mano.txt (una línea por foto, por ejemplo "M05/d1_clip_003_dia_1_0046.jpg")
y guarda cada recorte enderezado en <galeria>_recortes/M05/, igual que el automático.

Uso (desde la raíz del repo):
    python -m ia.herramientas.marcar_a_mano C:/VisionAI/embeddings/galeria

En cada foto, hacé 4 clicks en las esquinas DEL CARTEL QUE DICE EL TÍTULO, siempre en este orden:
    1 arriba-izq, 2 arriba-der, 3 abajo-der, 4 abajo-izq
Teclas: r = borrar clicks, s = saltear la foto, q = salir. Con 4 clicks se guarda solo.
Las fotos que ya tienen recorte se saltean, así que se puede cortar y seguir otro día.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from ia.embeddings import enderezar
from ia.main import GRAFO

galeria = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/VisionAI/embeddings/galeria")
recortes = galeria.parent / f"{galeria.name}_recortes"
nombre_de = {c["codigo"]: c["nombre"] for c in GRAFO["carteles"].values()}

pendientes = [l.strip() for l in (recortes / "a_mano.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
pendientes = [p for p in pendientes if not (recortes / p).exists()]
print(f"{len(pendientes)} fotos para marcar")

clicks = []


def al_clickear(evento, x, y, flags, param):
    if evento == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append([x, y])


cv2.namedWindow("marcar")
cv2.setMouseCallback("marcar", al_clickear)

for numero, pendiente in enumerate(pendientes, start=1):
    codigo = pendiente.split("/")[0]
    img = cv2.imread(str(galeria / pendiente))
    clicks.clear()
    titulo = f"{numero}/{len(pendientes)}  {codigo} = {nombre_de.get(codigo, '?')}   (s saltear, r borrar, q salir)"

    tecla = 0
    while len(clicks) < 4:
        vista = img.copy()
        cv2.putText(vista, titulo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(vista, titulo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        for i, (x, y) in enumerate(clicks, start=1):
            cv2.circle(vista, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(vista, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("marcar", vista)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla == ord("r"):
            clicks.clear()
        if tecla in (ord("s"), ord("q")):
            break

    if tecla == ord("q"):
        break
    if len(clicks) == 4:
        recorte = enderezar(img, np.array(clicks, dtype=np.float32))
        (recortes / codigo).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(recortes / pendiente), recorte)
        cv2.imshow("recorte", recorte)
        cv2.waitKey(400)

cv2.destroyAllWindows()
faltan = [p for p in pendientes if not (recortes / p).exists()]
print(f"Quedan {len(faltan)} sin marcar")
