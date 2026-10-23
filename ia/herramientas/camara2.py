import cv2
import time

cap = cv2.VideoCapture(0)
tiempo_antes = time.perf_counter()

tamano_ventana = 30
ventana = []
promedios = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    tiempo_actual = time.perf_counter()
    fps = 1 / (tiempo_actual - tiempo_antes)
    tiempo_antes = tiempo_actual

    ventana.append(fps)

    if len(ventana) > tamano_ventana:
        ventana.pop(0)

    if len(ventana) == tamano_ventana:
        promedio_actual = sum(ventana) / tamano_ventana
        promedios.append(promedio_actual)

        texto_fps = f"FPS Promedio: {int(promedio_actual)}"
    else:
        texto_fps = f"FPS: {int(fps)}"

    cv2.putText(
        frame,
        texto_fps,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("camara", frame)
    if cv2.waitKey(1) == ord('q'):
        break

print("Lista final de promedios:", promedios)

cap.release()
cv2.destroyAllWindows()

