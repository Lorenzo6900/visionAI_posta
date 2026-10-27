import cv2
import numpy as np


def interseccion(a, b, c, d):
    """Punto donde se cruzan la recta a-b y la recta c-d (None si son paralelas)."""
    r1 = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
    r2 = np.cross([c[0], c[1], 1.0], [d[0], d[1], 1.0])
    p = np.cross(r1, r2)
    if abs(p[2]) < 1e-6:
        return None
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float32)


def area_triangulo(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2


def esquinas_desde_poligono(pts, margen=0.3):
    """Polígono convexo de 4 o más vértices -> 4 esquinas en orden
    1 arriba-izq, 2 arriba-der, 3 abajo-der, 4 abajo-izq.
    Aprovecha que el triángulo blanco siempre corta la esquina de abajo a la derecha."""
    pts = np.array(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 4:
        return None
    original = pts.copy()

    # 1. Recorrer los vértices en sentido horario (como se ve en la imagen)
    n = len(pts)
    area = sum(pts[i, 0] * pts[(i + 1) % n, 1] - pts[(i + 1) % n, 0] * pts[i, 1]
               for i in range(n))
    if area < 0:
        pts = pts[::-1]

    # 2. Empezar por la esquina de arriba a la izquierda (x + y más chico)
    pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)

    # 3. Si hay más de 5 vértices, sacar los que menos aportan (ruido del contorno)
    while len(pts) > 5:
        n = len(pts)
        aporte = [area_triangulo(pts[i - 1], pts[i], pts[(i + 1) % n]) for i in range(n)]
        aporte[0] = np.inf                       # nunca sacar la esquina 1
        pts = np.delete(pts, int(np.argmin(aporte)), axis=0)

    # 4. Con 4 vértices ya están. Con 5, el triángulo blanco cortó la esquina 3:
    #    se prolongan el lado derecho (v1-v2) y el de abajo (v4-v3) hasta que se cruzan
    if len(pts) == 4:
        res = pts
    else:
        esquina3 = interseccion(pts[1], pts[2], pts[4], pts[3])
        if esquina3 is None:
            return None
        res = np.array([pts[0], pts[1], esquina3, pts[4]])

    # 5. Si alguna esquina cae lejísimo del polígono original, no la creemos
    xmin, ymin = original.min(axis=0)
    xmax, ymax = original.max(axis=0)
    mx, my = margen * (xmax - xmin), margen * (ymax - ymin)
    if not np.all((res[:, 0] >= xmin - mx) & (res[:, 0] <= xmax + mx) &
                  (res[:, 1] >= ymin - my) & (res[:, 1] <= ymax + my)):
        return None
    return res.astype(np.float32)


def mascara_por_centro(img, tol_h=18, factor_s=0.35, factor_v=0.5):
    """Lee el color del centro del recorte y arma la máscara de ese color."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    cy, cx = h // 2, w // 2
    dy, dx = int(h * 0.15), int(w * 0.15)
    parche = hsv[cy - dy:cy + dy, cx - dx:cx + dx].reshape(-1, 3)
    ok = parche[(parche[:, 1] > 60) & (parche[:, 2] > 40)]   # sin blancos ni oscuros
    if len(ok) < 20:
        return None
    h_med = float(np.median(ok[:, 0]))
    s_min = max(40, int(np.median(ok[:, 1]) * factor_s))
    v_min = max(30, int(np.median(ok[:, 2]) * factor_v))
    lo, hi = int(h_med - tol_h), int(h_med + tol_h)
    if lo < 0:                       # el rojo da la vuelta por 0
        m1 = cv2.inRange(hsv, (0, s_min, v_min), (hi, 255, 255))
        m2 = cv2.inRange(hsv, (lo + 180, s_min, v_min), (179, 255, 255))
        return cv2.bitwise_or(m1, m2)
    if hi > 179:
        m1 = cv2.inRange(hsv, (lo, s_min, v_min), (179, 255, 255))
        m2 = cv2.inRange(hsv, (0, s_min, v_min), (hi - 180, 255, 255))
        return cv2.bitwise_or(m1, m2)
    return cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))


def esquinas_en_recorte(img):
    mascara = mascara_por_centro(img)
    if mascara is None:
        return None
    # El filtro de limpieza se adapta al tamaño del recorte (carteles chicos = filtro chico)
    k = max(3, int(min(img.shape[:2]) * 0.03))
    k += (k % 2 == 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    casco = cv2.convexHull(max(contornos, key=cv2.contourArea))
    aprox = cv2.approxPolyDP(casco, 0.02 * cv2.arcLength(casco, True), True)
    return esquinas_desde_poligono(aprox.reshape(-1, 2))


def esquinas_en_caja(frame, caja, m=20):
    """Esquinas del cartel en coordenadas de la imagen COMPLETA (o None)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = caja
    ox, oy = max(x1 - m, 0), max(y1 - m, 0)        # dónde empieza el recorte
    ex, ey = min(x2 + m, W), min(y2 + m, H)
    esq = esquinas_en_recorte(frame[oy:ey, ox:ex])
    if esq is None:
        return None
    # Chequeo de sentido común: el cuadrilátero tiene que ocupar buena parte de la caja
    # (si ocupa muy poco o se sale de la caja, es una máscara mala) y ser convexo
    area = cv2.contourArea(esq.reshape(-1, 1, 2))
    area_caja = max((x2 - x1) * (y2 - y1), 1)
    if not (0.45 <= area / area_caja <= 1.2) or not cv2.isContourConvex(esq.reshape(-1, 1, 2)):
        return None
    return esq + np.array([ox, oy], dtype=np.float32)   # volver a la foto completa


if __name__ == "__main__":
    from ultralytics import YOLO

    RUTA_VIDEO = "clip_003_dia_2.mp4"      # <- tu video de dia 1 o 2 (o 0 para la cámara)

    model = YOLO("best.pt")
    cap = cv2.VideoCapture(RUTA_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir {RUTA_VIDEO}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = frame.copy()                          # dibujamos en una copia
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

        cv2.imshow("modulo esquinas", vis)
        if cv2.waitKey(30) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()