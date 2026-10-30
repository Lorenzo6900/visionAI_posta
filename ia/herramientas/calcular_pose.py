"""
Calculo de distancia y angulo a un cartel con solvePnP (Clase 12).
 
Este modulo SOLO hace la cuenta: no abre fotos, ni videos, ni ventanas.
Lo usan probar_pose_foto.py (clicks sobre una foto) y pose_en_video.py (YOLO + esquinas).
 
La calibracion (calibracion.npz) es la de la Pi Camera en el modo 1640x1232 -> 1280x720.
Solo vale mientras no se toque el foco del lente ni cambie ese modo de la camara.
"""
 
from pathlib import Path
 
import numpy as np
import cv2
 
# Resolucion (ancho, alto) con la que se calibro. Las fotos y los videos tienen que ser asi.
TAMANO_CALIBRACION = (1280, 720)
 
# Por defecto, la calibracion del repo (data/calibracion.npz), la misma que usa main.py
RUTA_CALIBRACION = Path(__file__).resolve().parents[2] / "data" / "calibracion.npz"
 
 
def cargar_calibracion(ruta=RUTA_CALIBRACION):
    """Devuelve la matriz de la camara (K) y los coeficientes de distorsion."""
    d = np.load(ruta)
    return d["matriz_intrinseca"], d["coef_distorsion"]
 
 
def puntos_del_cartel(ancho, alto):
    """
    Esquinas del cartel en el mundo real (metros), con el origen en su centro
    y Z = 0 (el cartel es plano). Mismo orden que las esquinas de la imagen:
    1 arriba-izq, 2 arriba-der, 3 abajo-der, 4 abajo-izq.
    """
    return np.array([
        [-ancho / 2, -alto / 2, 0],
        [ ancho / 2, -alto / 2, 0],
        [ ancho / 2,  alto / 2, 0],
        [-ancho / 2,  alto / 2, 0],
    ], dtype=np.float64)
 
 
def calcular_pose(puntos_foto, K, distorsion, ancho, alto, detalle=False):
    """
    puntos_foto: 4 esquinas en pixeles de la imagen COMPLETA, en el orden
                 arriba-izq, arriba-der, abajo-der, abajo-izq.
    ancho, alto: medidas reales del cartel, en metros.
 
    Devuelve (distancia_m, angulo_deg, error_reproyeccion_px).
      - distancia: de la camara al centro del cartel, en el piso (sin la altura).
      - angulo: 0 = de frente; + = camara a la derecha del cartel mirandolo de frente.
        Cerca de 0 (menos de ~12 grados) es ambiguo: el signo puede cambiar.
    Con detalle=True devuelve ademas las 4 esquinas reproyectadas.
    """
    puntos_foto = np.asarray(puntos_foto, dtype=np.float64).reshape(4, 2)
    puntos_pared = puntos_del_cartel(ancho, alto)
 
    # IPPE: metodo para objetos planos. La distorsion se pasa aca (no hacer undistort ademas).
    ok, rvec, tvec = cv2.solvePnP(puntos_pared, puntos_foto, K, distorsion,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP no pudo resolver la pose")
 
    # Posicion de la camara vista desde el cartel (origen en el centro del cartel)
    R, _ = cv2.Rodrigues(rvec)
    cx, cy, cz = (-R.T @ tvec).ravel()
 
    distancia = float(np.sqrt(cx**2 + cz**2))
    angulo = float(np.degrees(np.arctan2(cx, -cz)))
 
    # Chequeo: las esquinas reales proyectadas tienen que caer cerca de las marcadas
    reproyectados, _ = cv2.projectPoints(puntos_pared, rvec, tvec, K, distorsion)
    reproyectados = reproyectados.reshape(4, 2)
    error = float(np.linalg.norm(reproyectados - puntos_foto, axis=1).mean())
 
    if detalle:
        return distancia, angulo, error, reproyectados
    return distancia, angulo, error