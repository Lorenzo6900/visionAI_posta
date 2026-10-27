"""
Recorte enderezado y vector (embedding) de un cartel.

Lo usan herramientas/generar_galeria.py (en la PC, para armar data/galeria.npz) e identificar()
en main.py (en cada frame). Los dos TIENEN que hacer exactamente lo mismo; si no, los vectores
no se pueden comparar. Por eso está en un solo lugar.

TODO: en la raspi no hay PyTorch. MobileNet se va a tener que exportar (ONNX u otro, a decidir
con el equipo) y vector() cambiará por dentro, pero tiene que seguir dando los mismos números.
"""

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms

# Tamaño del recorte enderezado (el mismo que usaban herramientas/recortar_imagen.py y paraRecortar.py)
ANCHO_RECORTE, ALTO_RECORTE = 160, 190

_modelo = None
_preparar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _cargar_modelo():
    global _modelo
    if _modelo is None:
        _modelo = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
        _modelo.classifier = torch.nn.Identity()   # le saco la capa que clasifica
        _modelo.eval()
    return _modelo


def enderezar(frame: np.ndarray, esquinas: np.ndarray) -> np.ndarray:
    """Recorta el cartel y lo pone "de frente". esquinas: 4x2 en orden
    arriba-izq, arriba-der, abajo-der, abajo-izq (como las devuelve detectar())."""
    destino = np.float32([[0, 0], [ANCHO_RECORTE, 0], [ANCHO_RECORTE, ALTO_RECORTE], [0, ALTO_RECORTE]])
    M = cv2.getPerspectiveTransform(np.float32(esquinas), destino)
    return cv2.warpPerspective(frame, M, (ANCHO_RECORTE, ALTO_RECORTE))


def vector(recorte: np.ndarray) -> np.ndarray:
    """Vector de 1280 números del recorte (BGR, como lo da OpenCV), con largo 1.
    Con largo 1, el parecido entre dos vectores es directamente np.dot(a, b)."""
    rgb = cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        v = _cargar_modelo()(_preparar(rgb).unsqueeze(0)).squeeze().numpy()
    return (v / np.linalg.norm(v)).astype(np.float32)
