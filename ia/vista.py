"""
Dibuja sobre la foto lo que encontró la IA, para verlo en vivo en /ver (ia/servidor.py).
Solo sirve para probar y apuntar mejor la cámara: no cambia ninguna decisión.

Colores: verde = el cartel que se usó, amarillo = cartel con esquinas, rojo = cortado por el borde,
gris = YOLO lo ve pero no encontró las esquinas.
"""

import unicodedata

import cv2
import numpy as np

VERDE, AMARILLO, ROJO, GRIS, BLANCO, NEGRO = (0, 200, 0), (0, 220, 255), (0, 0, 255), (170, 170, 170), (255, 255, 255), (0, 0, 0)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def sin_tildes(texto: str) -> str:
    """putText de OpenCV no sabe dibujar tildes ni la ñ: se las saca."""
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn").replace("ñ", "n").replace("Ñ", "N")


def escribir(imagen, texto, x, y, color, escala=0.7):
    # Con un borde negro se lee sobre cualquier fondo
    cv2.putText(imagen, sin_tildes(texto), (x, y), FUENTE, escala, NEGRO, 4, cv2.LINE_AA)
    cv2.putText(imagen, sin_tildes(texto), (x, y), FUENTE, escala, color, 2, cv2.LINE_AA)


def dibujar(frame, detecciones, cortadas, mejor, nombre, motivo, texto, debug=None) -> bytes:
    """Devuelve la foto con las cajas, las esquinas, el motivo y lo que se dijo, como JPEG."""
    vista = frame.copy()
    ancho = vista.shape[1]

    for d in detecciones:
        x1, y1, x2, y2 = d.caja
        if d is mejor:
            color, etiqueta = VERDE, nombre
        elif d in cortadas:
            color, etiqueta = ROJO, "cortado por el borde"
        elif d.esquinas_refinadas:
            color, etiqueta = AMARILLO, "con esquinas"
        else:
            color, etiqueta = GRIS, "sin esquinas"
        cv2.rectangle(vista, (x1, y1), (x2, y2), color, 2)
        escribir(vista, f"{etiqueta}  YOLO {d.confianza:.2f}", x1, max(y1 - 10, 95), color, 0.6)

        if d.esquinas_refinadas:
            puntos = d.esquinas.astype(np.int32)
            cv2.polylines(vista, [puntos], True, color, 2)
            for i, (x, y) in enumerate(puntos, start=1):
                cv2.circle(vista, (int(x), int(y)), 6, ROJO, -1)
                escribir(vista, str(i), int(x) + 8, int(y) - 8, BLANCO, 0.7)

    # Banda de arriba: el motivo, lo que se dijo y la posición
    cv2.rectangle(vista, (0, 0), (ancho, 80), NEGRO, -1)
    escribir(vista, motivo, 12, 30, VERDE if motivo == "ok" else AMARILLO, 0.8)
    linea = f"dice: {texto}" if texto else "dice: (nada)"
    if debug and debug.get("x") is not None:
        linea += f"   |   x={debug['x']} y={debug['y']} rumbo={debug['rumbo']} conf={debug['confianza']}"
    escribir(vista, linea, 12, 64, BLANCO, 0.6)

    _, jpeg = cv2.imencode(".jpg", vista, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return jpeg.tobytes()