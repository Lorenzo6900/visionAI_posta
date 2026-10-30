"""
Clase 11 + Clase 12 juntas: YOLO detecta el cartel, el modulo de esquinas
encuentra sus 4 esquinas y solvePnP (con calibracion.npz) da distancia y angulo.
 
Uso:  python pose_en_video.py
Cambia las rutas de abajo. Con la tecla q salis.
"""
 
import cv2
from ultralytics import YOLO
 
# El archivo con tu codigo de la Clase 11. Si lo guardaste con otro nombre,
# cambia "esquinas" por ese nombre (sin el .py).
from modelo_esquinas import esquinas_en_caja
from calcular_pose import calcular_pose, cargar_calibracion, TAMANO_CALIBRACION
 
RUTA_VIDEO = "clip_003_dia_2.mp4"     # o 0 para la camara
RUTA_MODELO = "best.pt"
 
# Medidas del cartel (metros, con PUNTO). Por ahora fijas: sirven si en el video
# aparece un solo tipo de cartel. Mas adelante hay que buscarlas en la tabla
# (marcador_id, ancho_cm, alto_cm) segun el cartel que reconozca el sistema.
ANCHO_CARTEL = 0.80
ALTO_CARTEL = 0.95
 
ERROR_MAX_PX = 3   # por encima de esto, la pose se muestra como "dudosa"
 
 
def main():
    K, distorsion = cargar_calibracion()   # calibracion.npz al lado de calcular_pose.py
    print(f"Calibracion cargada: fx={K[0,0]:.2f} (la de la Pi Camera tiene fx=950.57)")
    model = YOLO(RUTA_MODELO)
 
    cap = cv2.VideoCapture(RUTA_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir {RUTA_VIDEO}")
 
    # K solo sirve para la resolucion con la que se calibro
    ancho_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (ancho_v, alto_v) != TAMANO_CALIBRACION:
        raise SystemExit(f"El video es de {ancho_v}x{alto_v} y la calibracion es de "
                         f"{TAMANO_CALIBRACION[0]}x{TAMANO_CALIBRACION[1]}.")
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()
        res = model(frame, conf=0.25, verbose=False)[0]
 
        for caja in res.boxes:
            x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
            esq = esquinas_en_caja(frame, (x1, y1, x2, y2))
            if esq is None:
                cv2.putText(vis, "sin esquinas", (x1, max(y1 - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue
 
            for i, (x, y) in enumerate(esq, start=1):
                cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
                cv2.putText(vis, str(i), (int(x) + 8, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
 
            distancia, angulo, error = calcular_pose(esq, K, distorsion,
                                                     ANCHO_CARTEL, ALTO_CARTEL)
            if error <= ERROR_MAX_PX:
                texto = f"{distancia:.2f} m  {angulo:+.0f} grados"
                color = (0, 255, 0)
            else:
                texto = f"pose dudosa (error {error:.1f} px)"
                color = (0, 165, 255)
            cv2.putText(vis, texto, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
        cv2.imshow("esquinas + pose", vis)
        if cv2.waitKey(30) == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 