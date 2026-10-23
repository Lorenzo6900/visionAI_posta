import cv2, os, shutil
import numpy as np

carpeta = r"C:\VisionAI\dataset_real\frames"
dest = r"C:\VisionAI\dataset_real\descartados_borrosos"
os.makedirs(dest, exist_ok=True)

archivos = [f for f in os.listdir(carpeta) if f.endswith(".jpg")]
puntajes = {}
for f in archivos:
    img = cv2.imread(os.path.join(carpeta, f), cv2.IMREAD_GRAYSCALE)
    puntajes[f] = cv2.Laplacian(img, cv2.CV_64F).var()

corte = np.percentile(list(puntajes.values()), 8)
movidos = 0
for f, p in puntajes.items():
    if p < corte:
        shutil.move(os.path.join(carpeta, f), os.path.join(dest, f))
        movidos += 1

print(f"Corte: {corte:.1f} | movidos: {movidos} | quedan: {len(archivos) - movidos}")