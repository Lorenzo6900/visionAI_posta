# VisionAI

Sistema de navegación en interiores para personas con discapacidad visual. Una Raspberry Pi 3 (1 GB de RAM), con cámara y auriculares con micrófono, detecta carteles en el piso, calcula dónde está la persona y la guía por audio hasta un aula.

## Arquitectura
- Todo corre en la raspi, excepto la voz (STT y TTS), que la hace el backend de Tincho por HTTP (`voz/`).
- Pipeline por frame (~1 fps), en `ia/main.py`:
  YOLO11n (1 clase: `cartel`) -> esquinas -> embedding (qué cartel es) -> nodo -> solvePnP (posición) -> Dijkstra (ruta) -> motor de decisiones -> texto -> voz.
- El grafo vive en `data/grafo.json` y se lee al arrancar. En la raspi no hay base de datos. La fuente es la base Neon de Tincho (tablas `nodos`, `aristas` y `marcadores`): se exporta a `data/*.csv` y se regenera con `python ia/herramientas/generar_grafo.py`.
- Ids: se usan los ids de Neon (1 a 35). El N viejo de la planilla "tablas nodos" queda en `numero`. Los nodos van de a pares gemelos (uno por lado del pasillo, en el mismo punto), por ejemplo aula_201 y aula_202.
- Flujo de un cartel: YOLO lo detecta -> los embeddings dicen cuál es (su nombre, por ejemplo `impacto_social`; las carpetas de la galería se llaman por su código M01..M19) -> en `grafo.json` se busca su x, y, tamaño, orientación y nodo -> solvePnP da la posición de la persona -> Dijkstra -> instrucción.
- `grados` de un cartel = hacia dónde mira su frente, con este=0, norte=90, oeste=180, sur=270 (x crece al este, y al norte). Es lo opuesto a su `pared`.
- Motor de decisiones: habla si la instrucción cambió; si no cambió, repite como máximo cada 5 s; si la confianza es baja, dice que no está seguro.
- En la raspi **no se instala PyTorch**: los modelos (YOLO y el de embeddings) se exportan a ONNX y se corren con onnxruntime. Hace falta Raspberry Pi OS Lite de 64 bits: `dpkg --print-architecture` tiene que dar `arm64` (`armhf` = 32 bits, onnxruntime no se instala). No usar `uname -m`, que puede dar `aarch64` con sistema de 32 bits.

## Estructura
- `ia/main.py`: el pipeline. Hoy tiene una función por pieza con datos falsos.
- `detectar()`: por ahora corre YOLO con ultralytics y `models/best.pt` (lo de ONNX todavía no se decidió con el equipo). Las esquinas son las de la caja hasta que Ivan pase su código.
- `ia/herramientas/`: demos de clase y scripts de PC (calibración, recortes, comparación de embeddings, solvePnP, Dijkstra, filtrado de imágenes borrosas). Son la referencia para completar cada función de `main.py`.
- `voz/cliente.py`: stub del cliente HTTP de voz que corre en la raspi. Lo completa Tincho.
- `voz/backend/`: servidor Node de Tincho. Hace STT (`transcripcion.js`) y convierte lo que dijo la persona en un destino (`GET /destinos/buscar`, `busquedaService.js`). Las credenciales van en `voz/backend/.env` (ver `.env.example`). Arranca con `npm start`. La ruta NO se calcula acá: la calcula la raspi.
- `data/`: `grafo.json` (generado), las 3 tablas de Neon en CSV, `calibracion.npz`, `dataset.yaml`.
- `models/best.pt`: YOLO11n entrenado (mAP50 ≈ 0.92 en val d1+d2). Es el único peso que se sube a git.
- `docs/cvat/`: docker-compose de CVAT para etiquetar.

## Cómo correr
- PC: `pip install -r requirements-pc.txt`
- Raspi: `pip install -r requirements-raspi.txt` (picamera2 se instala con apt)
- Desde la raíz: `python -m ia.main`

## Pendientes conocidos
- **Recalibrar la cámara.** `data/calibracion.npz` parece hecho a unos 640x480 (centro óptico en 351, 282) y tiene k3 ≈ −48, que es sospechoso. La cámara ahora captura a 1280x720. `herramientas/solvePNP.py` usa una K nominal hasta que se recalibre.
- En Neon faltan 6 aristas de los gemelos lab_1 (19) y aula_214 (29). Mientras tanto las agrega `ARISTAS_FALTANTES` en `generar_grafo.py`; cuando estén en Neon, se vacía esa lista.
- La galería de embeddings tiene 5 de los 19 carteles (M01, M03, M05, M06, M08).
- Exportar `best.pt` y MobileNetV2 (embeddings) a ONNX.
- Falta el TTS (lo hace Tincho).

## Reglas
- No subir el dataset ni imágenes al repo. Los datos están en `C:\VisionAI` y en Roboflow.
- Código y comentarios en español, con el estilo simple de los scripts existentes.
