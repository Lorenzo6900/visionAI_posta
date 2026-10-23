import cv2, numpy as np, glob, os

BASE = os.path.dirname(os.path.abspath(__file__))

def vector(ruta):
    img = cv2.imread(ruta)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()

def similitud(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

MODELOS = ["M01", "M03", "M05", "M06", "M08"]

galeria = []
for carpeta in MODELOS:
    for ruta in glob.glob(f"{BASE}/recortes/{carpeta}/*.jpg"):
        galeria.append((vector(ruta), carpeta))

print(f"Galería: {len(galeria)} vectores\n")

aciertos, total = 0, 0
for carpeta in MODELOS:
    for ruta in glob.glob(f"{BASE}/recortes/test/{carpeta}/*.jpg"):
        v = vector(ruta)
        sims = [(similitud(v, g), etiqueta) for g, etiqueta in galeria]
        mejor_sim, prediccion = max(sims)
        ok = prediccion == carpeta
        aciertos += ok
        total += 1
        print(f"{os.path.basename(ruta):28} real={carpeta}  pred={prediccion}  sim={mejor_sim:.3f}  {'OK' if ok else 'ERROR'}")

print(f"\nAccuracy: {aciertos}/{total} = {aciertos/total:.1%}\n")

for ruta in glob.glob(f"{BASE}/recortes/negativos/*.jpg"):
    v = vector(ruta)
    mejor_sim, prediccion = max([(similitud(v, g), e) for g, e in galeria])
    print(f"NEGATIVO {os.path.basename(ruta):20} sim={mejor_sim:.3f} (diría {prediccion})")