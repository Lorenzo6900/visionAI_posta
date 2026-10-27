"""
Arma data/galeria.npz: un vector por cada recorte de la galería, con el código del cartel (M01..M19).
Se corre en la PC. Son 3 pasos (desde la raíz del repo):

1. Recortes automáticos (YOLO + esquinas de Ivan, lo mismo que hace la raspi en cada frame):
       python -m ia.herramientas.generar_galeria recortar C:/VisionAI/embeddings/galeria
   Guarda los recortes enderezados en <galeria>_recortes/M01, M02, ... y anota en
   <galeria>_recortes/a_mano.txt las fotos donde no encontró las esquinas.

2. Revisar los recortes a ojo. Si alguno no es el cartel de su carpeta, borrarlo y agregar su foto
   a a_mano.txt. Después marcar esas fotos a mano:
       python -m ia.herramientas.marcar_a_mano C:/VisionAI/embeddings/galeria

3. Vectores, solo con los recortes que quedaron (ya revisados):
       python -m ia.herramientas.generar_galeria vectores C:/VisionAI/embeddings/galeria
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from ia.embeddings import enderezar, vector
from ia.main import GRAFO, RAIZ

MINIMO_POR_CARTEL = 5     # con menos vectores que esto, conviene sacar más fotos de ese cartel
EXTENSIONES = (".jpg", ".jpeg", ".png")

nombre_de = {c["codigo"]: c["nombre"] for c in GRAFO["carteles"].values()}


def carpetas_de_carteles(carpeta: Path) -> list[Path]:
    subs = sorted(p for p in carpeta.iterdir() if p.is_dir())
    for sub in subs:
        if sub.name not in nombre_de:
            print(f"Aviso: la carpeta {sub.name} no es ningún cartel de grafo.json, se saltea")
    return [s for s in subs if s.name in nombre_de]


def recortar(galeria: Path, recortes: Path) -> None:
    from ia.main import detectar     # carga YOLO solo en este paso

    a_mano = []
    print(f"{'cartel':6} {'nombre':22} {'fotos':>5} {'recortes':>8} {'sin cartel':>10} {'sin esquinas':>12} {'varios':>6}")
    for sub in carpetas_de_carteles(galeria):
        codigo = sub.name
        (recortes / codigo).mkdir(parents=True, exist_ok=True)
        fotos = sorted(p for p in sub.iterdir() if p.suffix.lower() in EXTENSIONES)
        hechos = sin_cartel = sin_esquinas = varios = 0
        for ruta in fotos:
            frame = cv2.imread(str(ruta))
            detecciones = detectar(frame)
            buenas = [d for d in detecciones if d.esquinas_refinadas]
            if not buenas:
                sin_cartel += not detecciones
                sin_esquinas += bool(detecciones)
                a_mano.append(f"{codigo}/{ruta.name}")
                continue
            # Si hay más de un cartel en la foto se usa el más grande, que puede NO ser el de la carpeta:
            # por eso hay que revisar los recortes (paso 2).
            varios += len(buenas) > 1
            mejor = max(buenas, key=lambda d: cv2.contourArea(d.esquinas.astype(np.float32)))
            cv2.imwrite(str(recortes / codigo / ruta.name), enderezar(frame, mejor.esquinas))
            hechos += 1
        print(f"{codigo:6} {nombre_de[codigo]:22} {len(fotos):>5} {hechos:>8} {sin_cartel:>10} {sin_esquinas:>12} {varios:>6}")

    (recortes / "a_mano.txt").write_text("\n".join(a_mano) + "\n", encoding="utf-8")
    print(f"\nRecortes en {recortes}")
    print(f"{len(a_mano)} fotos para marcar a mano, anotadas en {recortes / 'a_mano.txt'}")


def vectores(recortes: Path) -> None:
    lista, codigos, archivos = [], [], []
    print(f"{'cartel':6} {'nombre':22} {'vectores':>8}")
    for sub in carpetas_de_carteles(recortes):
        codigo = sub.name
        for ruta in sorted(p for p in sub.iterdir() if p.suffix.lower() in EXTENSIONES):
            lista.append(vector(cv2.imread(str(ruta))))
            codigos.append(codigo)
            archivos.append(f"{codigo}/{ruta.name}")
        cantidad = codigos.count(codigo)
        aviso = "  <- pocas" if cantidad < MINIMO_POR_CARTEL else ""
        print(f"{codigo:6} {nombre_de[codigo]:22} {cantidad:>8}{aviso}")

    faltan = sorted(set(nombre_de) - set(codigos))
    if faltan:
        print(f"\nCarteles sin ningún vector (no los va a poder reconocer): {', '.join(faltan)}")

    destino = RAIZ / "data" / "galeria.npz"
    np.savez(destino, vectores=np.array(lista), codigos=np.array(codigos), archivos=np.array(archivos))
    print(f"\n{len(lista)} vectores guardados en {destino}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("recortar", "vectores"):
        sys.exit("Uso: python -m ia.herramientas.generar_galeria recortar|vectores [carpeta de la galería]")
    galeria = Path(sys.argv[2] if len(sys.argv) > 2 else "C:/VisionAI/embeddings/galeria")
    recortes = galeria.parent / f"{galeria.name}_recortes"
    if sys.argv[1] == "recortar":
        recortar(galeria, recortes)
    else:
        vectores(recortes)
