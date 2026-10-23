"""
Ejemplo Clase 12: distancia y ángulo a un cartel con solvePnP.
 
Uso:  python pose_solvepnp.py m19_3m_30.jpg
Hacé 4 clicks en las esquinas, SIEMPRE en este orden:
    1 arriba-izq, 2 arriba-der, 3 abajo-der, 4 abajo-izq
Tecla r = borrar clicks, q = salir.
"""
 
import sys
import numpy as np
import cv2
 
# ---------- 1. Medidas del cartel (en metros, medidas con cinta) ----------
ANCHO = 1.20   # M19
ALTO = 0.90
 
# Lista 1: esquinas en la pared, desde el centro del cartel (mismo orden que los clicks)
puntos_pared = np.array([
    [-ANCHO / 2, -ALTO / 2, 0],   # 1 arriba-izq
    [ ANCHO / 2, -ALTO / 2, 0],   # 2 arriba-der
    [ ANCHO / 2,  ALTO / 2, 0],   # 3 abajo-der
    [-ANCHO / 2,  ALTO / 2, 0],   # 4 abajo-izq
], dtype=np.float64)
 
# ---------- 2. Cámara (K nominal a 1280x720 hasta que Ivan calibre) ----------
K = np.array([[1060, 0, 640],
              [0, 1060, 360],
              [0,    0,   1]], dtype=np.float64)
distorsion = np.zeros(5)
 
# ---------- 3. Clicks (Lista 2) ----------
img = cv2.imread(sys.argv[1])
clicks = []
 
def al_clickear(evento, x, y, flags, param):
    if evento == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append([x, y])
 
cv2.namedWindow("foto")
cv2.setMouseCallback("foto", al_clickear)
 
while True:
    vista = img.copy()
    for i, (x, y) in enumerate(clicks):
        cv2.circle(vista, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(vista, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("foto", vista)
    tecla = cv2.waitKey(20) & 0xFF
    if tecla == ord("r"):
        clicks.clear()
    if tecla == ord("q") or len(clicks) == 4:
        break
 
cv2.waitKey(500)
cv2.destroyAllWindows()
if len(clicks) < 4:
    sys.exit("Faltan clicks.")
 
puntos_foto = np.array(clicks, dtype=np.float64)
 
# ---------- 4. solvePnP ----------
ok, rvec, tvec = cv2.solvePnP(puntos_pared, puntos_foto, K, distorsion,
                              flags=cv2.SOLVEPNP_IPPE)   # IPPE: método para objetos planos
 
# ---------- 5. Dónde está la cámara, visto desde el cartel ----------
R, _ = cv2.Rodrigues(rvec)               # rvec (3 números) -> matriz de giro
camara = (-R.T @ tvec).ravel()           # posición de la cámara con el cartel como origen
cx, cy, cz = camara
 
distancia = np.sqrt(cx**2 + cz**2)                 # en el piso (sin la altura)
angulo = np.degrees(np.arctan2(cx, -cz))           # 0 = de frente; el signo dice de qué lado
 
# ---------- 6. Chequeo: error de reproyección ----------
reproyectados, _ = cv2.projectPoints(puntos_pared, rvec, tvec, K, distorsion)
error = np.linalg.norm(reproyectados.reshape(4, 2) - puntos_foto, axis=1).mean()
 
print(f"Distancia: {distancia:.2f} m")
print(f"Ángulo:    {angulo:+.1f}°")
print(f"Error de reproyección: {error:.1f} px  ({'OK' if error <= 3 else 'REVISAR clicks'})")
 