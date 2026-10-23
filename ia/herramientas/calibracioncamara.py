import cv2
import os
from picamera2 import Picamera2

PATRON = (9, 6)
MINIMO_NECESARIO = 20
CARPETA_CAPTURAS = "capturas"

os.makedirs(CARPETA_CAPTURAS, exist_ok=True)

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    raw={"size": (1640, 1232)},
)
picam2.configure(config)
picam2.start()

puntos_guardados = []

while True:                                  
    frame = picam2.capture_array()           

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    encontrado, esquinas = cv2.findChessboardCorners(gris, PATRON, flags=cv2.CALIB_CB_FAST_CHECK)

    limpia = frame.copy()       

    if encontrado:
        cv2.drawChessboardCorners(frame, PATRON, esquinas, encontrado)

    texto = f"Guardadas: {len(puntos_guardados)}/{MINIMO_NECESARIO}"
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("calibracion", frame)

    tecla = cv2.waitKey(1)

    if tecla == ord('s') and encontrado:
        puntos_guardados.append(esquinas)

        numero = len(puntos_guardados)
        nombre_archivo = f"{CARPETA_CAPTURAS}/captura_{numero:02d}.jpg"
        cv2.imwrite(nombre_archivo, limpia)  # <-- guarda la copia limpia

        print(f"Guardada #{numero} -> {nombre_archivo}")

    if tecla == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

print(f"Total de detecciones guardadas: {len(puntos_guardados)}")