"""
Prueba detectar() de main.py sobre fotos (sin cámara). Corre desde la raíz del repo:

    python -m ia.herramientas.probar_detectar C:/VisionAI/dataset_real/lote_test
    python -m ia.herramientas.probar_detectar C:/VisionAI/dataset_real/final/images/val

Guarda cada foto con lo detectado en <carpeta>_detectado/:
    caja azul = YOLO, esquinas verdes = refinadas, esquinas rojas = las de la caja (no se pudo refinar)
Si al lado de images/ hay labels/ (formato YOLO), cuenta aciertos contra las etiquetas.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ia.main import detectar


def leer_etiquetas(ruta_imagen: Path, ancho: int, alto: int) -> list[tuple[float, float, float, float]] | None:
    ruta_txt = Path(str(ruta_imagen.with_suffix(".txt")).replace("images", "labels"))
    if "images" not in str(ruta_imagen) or not ruta_txt.exists():
        return None
    cajas = []
    for linea in ruta_txt.read_text().split("\n"):
        if linea.strip():
            _, cx, cy, w, h = map(float, linea.split()[:5])
            cajas.append(((cx - w / 2) * ancho, (cy - h / 2) * alto, (cx + w / 2) * ancho, (cy + h / 2) * alto))
    return cajas


def iou(a, b) -> float:
    ancho = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    alto = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    interseccion = ancho * alto
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - interseccion
    return interseccion / union if union > 0 else 0


carpeta = Path(sys.argv[1])
salida = carpeta.parent / f"{carpeta.name}_detectado"
salida.mkdir(exist_ok=True)

imagenes = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
tiempos = []
total_detecciones = refinadas = 0
etiquetados = aciertos = falsos = 0

for ruta in imagenes:
    frame = cv2.imread(str(ruta))
    inicio = time.perf_counter()
    detecciones = detectar(frame)
    tiempos.append(time.perf_counter() - inicio)

    total_detecciones += len(detecciones)
    for d in detecciones:
        refinadas += d.esquinas_refinadas
        x1, y1, x2, y2 = d.caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{d.confianza:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        color = (0, 255, 0) if d.esquinas_refinadas else (0, 0, 255)
        cv2.polylines(frame, [d.esquinas.astype(np.int32)], True, color, 2)
        for numero, (x, y) in enumerate(d.esquinas.astype(int), start=1):
            cv2.putText(frame, str(numero), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(salida / ruta.name), frame)

    etiquetas = leer_etiquetas(ruta, frame.shape[1], frame.shape[0])
    if etiquetas is not None:
        etiquetados += len(etiquetas)
        acertadas = [any(iou(d.caja, e) > 0.5 for d in detecciones) for e in etiquetas]
        aciertos += sum(acertadas)
        falsos += sum(not any(iou(d.caja, e) > 0.5 for e in etiquetas) for d in detecciones)

print(f"{len(imagenes)} fotos, {total_detecciones} detecciones, {refinadas} con esquinas refinadas")
# El primer frame es más lento (onnxruntime se calienta), por eso se saltea
print(f"Tiempo de detectar(): {np.mean(tiempos[1:] or tiempos) * 1000:.0f} ms promedio")
if etiquetados:
    print(f"Carteles etiquetados: {etiquetados}, encontrados: {aciertos} ({aciertos / etiquetados:.0%}), "
          f"detecciones de más: {falsos}")
print(f"Fotos dibujadas en {salida}")
