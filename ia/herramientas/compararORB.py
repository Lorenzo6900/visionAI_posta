import cv2, numpy as np, glob, os

BASE = os.path.dirname(os.path.abspath(__file__))

import torch, torchvision
from torchvision import transforms

modelo = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
modelo.classifier = torch.nn.Identity()   # le saco la capa que clasifica
modelo.eval()

preparar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

orb = cv2.ORB_create(nfeatures=500)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

def vector(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    kp, des = orb.detectAndCompute(img, None)
    return des

def similitud(a, b):
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0.0
    pares = bf.knnMatch(a, b, k=2)
    buenos = [m for m, n in pares if len(pares[0]) == 2 and m.distance < 0.75 * n.distance]
    return len(buenos) / max(len(a), 1)

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