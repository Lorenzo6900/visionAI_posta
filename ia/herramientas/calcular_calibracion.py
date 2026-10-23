import cv2
import numpy as np
import glob

PATRON = (9, 6)
TAMANO_CUADRADO_MM = 20

# Grilla de posiciones reales, siempre la misma
objp = []
for fila in range(PATRON[1]):
    for columna in range(PATRON[0]):
        objp.append((columna * TAMANO_CUADRADO_MM, fila * TAMANO_CUADRADO_MM, 0))
objp = np.array(objp, dtype=np.float32)

archivos = glob.glob("capturas/*.jpg")
print(f"Encontradas {len(archivos)} imagenes")

puntos_3d = []
puntos_2d = []
tamano_imagen = None

for archivo in archivos:
    img = cv2.imread(archivo)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tamano_imagen = gris.shape[::-1]

    encontrado, esquinas = cv2.findChessboardCorners(gris, PATRON)

    if encontrado:
        puntos_3d.append(objp)
        puntos_2d.append(esquinas)
    else:
        print(f"Aviso: no se detecto el tablero en {archivo}")

print(f"Usando {len(puntos_3d)} imagenes validas de {len(archivos)}")

ok, matriz_intrinseca, coef_distorsion, rvecs, tvecs = cv2.calibrateCamera(
    puntos_3d, puntos_2d, tamano_imagen, None, None
)

print("\nMatriz intrinseca:")
print(matriz_intrinseca)
print("\nCoeficientes de distorsion:")
print(coef_distorsion)

error_total = 0
for i in range(len(puntos_3d)):
    puntos_reproyectados, _ = cv2.projectPoints(
        puntos_3d[i], rvecs[i], tvecs[i], matriz_intrinseca, coef_distorsion
    )
    diferencia = puntos_2d[i].reshape(-1, 2) - puntos_reproyectados.reshape(-1, 2)
    error = np.sqrt(np.mean(np.sum(diferencia**2, axis=1)))
    error_total += error

error_promedio = error_total / len(puntos_3d)
print(f"\nError de reproyeccion: {error_promedio:.4f} pixeles")

np.savez("calibracion.npz", matriz_intrinseca=matriz_intrinseca, coef_distorsion=coef_distorsion)
print("Guardado en calibracion.npz")