import cv2, numpy as np, glob, os

ENTRADA = "embeddings/test_galeria/M08"
SALIDA  = "embeddings/recortes/test/M08"
W, H = 160, 190

os.makedirs(SALIDA, exist_ok=True)
puntos = []

def click(evento, x, y, flags, param):
    if evento == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
        puntos.append((x, y))
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow("marcar", img)

for ruta in sorted(glob.glob(ENTRADA + "/*.jpg")):
    img = cv2.imread(ruta)
    puntos = []
    cv2.imshow("marcar", img)
    cv2.setMouseCallback("marcar", click)
    tecla = cv2.waitKey(0)

    if tecla == ord('s') or len(puntos) != 4:
        continue

    origen  = np.float32(puntos)
    destino = np.float32([[0,0], [W,0], [W,H], [0,H]])
    M = cv2.getPerspectiveTransform(origen, destino)
    recorte = cv2.warpPerspective(img, M, (W, H))
    cv2.imwrite(os.path.join(SALIDA, os.path.basename(ruta)), recorte)

cv2.destroyAllWindows()