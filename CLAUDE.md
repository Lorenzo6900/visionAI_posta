# VisionAI

Sistema de navegación en interiores para personas con discapacidad visual. Una Raspberry Pi 3 (1 GB de RAM), con cámara, auriculares con micrófono y un botón, saca fotos del pasillo. Un servidor reconoce los carteles, calcula dónde está la persona y la guía por audio hasta un aula.

## Arquitectura (acordada el 2026-10-27)
```
raspi  ──frame / audio──▶  Node (Tincho)  ──frame / texto──▶  IA, FastAPI (ia/)
raspi  ◀──────audio──────  Node: STT y TTS ◀──────texto───────  IA
```
- **La IA no corre en la raspi**: corre en un servidor de ORT, junto con Node, en la misma máquina. Por eso en la raspi no hace falta PyTorch ni ONNX.
- **La raspi nunca le habla directo a la IA.** Node es la puerta de entrada, porque pasa el texto a audio antes de devolverlo.
- **Hay una sola raspi y un solo usuario a la vez.** La IA guarda un único estado global en memoria (destino, última instrucción, último cartel visto). No hay sesión ni autenticación.
- **Las indicaciones las arma la IA con el grafo**, no un chat de IA: el chat no conoce el mapa ni dónde está la persona.

### Contrato de la API
**Raspi → Node**
- `POST /frame`: `multipart/form-data`, campo `imagen` (JPEG, 1280x720). Responde con audio, o `204 No Content` si no hay nada que decir.
- `POST /audio`: el audio grabado mientras se apretaba el botón (wav o webm, lo decide Tincho). Responde con el audio de la frase de confirmación.

**Node → IA** (FastAPI en `http://localhost:8000`)
- `POST /frame`: el mismo JPEG, `multipart/form-data`, campo `imagen`. Devuelve:
  ```json
  {"texto": "Girá a la derecha y avanzá 3 metros.", "debug": {"nodo": 10, "x": 0.4, "y": 29.8, "rumbo": 92, "confianza": 1.0}}
  ```
  `"texto": null` si conviene callar. `debug` es para probar la posición sin escuchar el audio; Node lo ignora.
- `POST /destino`: `{"texto": "quiero ir al baño"}` (lo que transcribió Node). Devuelve:
  ```json
  {"frase": "Voy a llevarte a baño", "destino_id": 23}
  ```
  Si no entiende: `{"frase": "No te entendí. ¿A dónde querés ir?", "destino_id": null}`. Node pasa `frase` al TTS.
- Todo responde `200` con JSON. Si la IA falla o tarda, Node lo trata como `texto: null` y no corta el flujo.

### Pipeline de cada frame (`ia/main.py`)
`detectar()` (YOLO11n + esquinas de Ivan) -> `enderezar()` -> `identificar()` (vector + galería) -> cartel en `grafo.json` -> `posicion()` (solvePnP) -> `ruta()` (Dijkstra) -> `decidir()` -> texto.
- Solo se usan los carteles con esquinas reales (`esquinas_refinadas`). Sin ellas no se puede enderezar el recorte ni usar solvePnP.
- **Elección del cartel del frame** (cuando hay varios, por ejemplo en la pared sur): se descartan los cortados por el borde de la foto (caja a menos de `MARGEN_BORDE_PX = 5` del borde: sus esquinas caen sobre el borde y solvePnP da cualquier cosa) y los que `identificar()` no reconoce. De los que quedan se usa el más grande (área de las esquinas): el más cercano, con el que solvePnP es más preciso.
- `identificar()` responde solo si el cartel más parecido de la galería se parece al menos `PARECIDO_MINIMO = 0.75` y le gana al segundo por al menos `MARGEN_MINIMO = 0.04`. Si no, "no sé" y se espera al frame siguiente. En fotos del día 3, contesta en ~6 de cada 10 recortes y acierta el 97%. Se da por terminado así.

### Reglas del motor de decisiones (`decidir()`)
Todos los frames pasan por `decidir()`, también los que no tienen cartel. Gana el primer caso que se cumple:
1. **Callado:** después de llegar no dice nada hasta que la persona vuelva a hablar. Cualquier `/destino` lo saca, aunque no se le entienda.
2. **Sin destino:** "Decime a dónde querés ir." Gana aunque tampoco vea carteles.
3. **Sin posición válida** (no vio un cartel identificado, o solvePnP dio poca confianza: cuentan igual): se calla, y si pasan más de `SIN_CARTEL_S = 10` s desde la última posición válida dice "Mové la cabeza despacio." El reloj arranca de nuevo cuando llega un destino.Después lo repite como máximo cada REPETIR_MOVER_S = 15 s (las demás frases, cada 5 s).
4. **Llegada:** "Llegaste a … La puerta está a tu …", borra el destino y pasa a callado.
5. **Instrucción:** "Girá a la derecha y avanzá 3 metros", etc.

- Habla si la instrucción cambió. Si no cambió, la repite como máximo cada 5 s (`REPETIR_CADA_S`).
- **Destino nuevo:** un `/destino` nuevo reemplaza al anterior en cualquier momento, también a mitad de camino, y la próxima instrucción se dice enseguida.
- Ya no existe "No estoy seguro de dónde estás": la poca confianza cae en el caso 3.

### Texto -> destino (`/destino`)
En `ia/herramientas/destino.py` (`buscar_destino(texto, GRAFO["nodos"])`), portado de `voz/backend/busquedaService.js`. Se compara solo con el `nombre` (no con `numero`) y solo con los nodos `es_destino`. Siempre toma el mejor candidato y lo dice en voz alta ("Voy a llevarte a aula 203"); si ninguno pasa el puntaje mínimo, "No te entendí. ¿A dónde querés ir?". El nombre alcanza para diferenciar los gemelos (aula 201 y aula 202). No entiende "laboratorio" por "lab".

## Mapa y convenciones
- El grafo vive en `data/grafo.json` y se lee al arrancar. La fuente es la base Neon de Tincho (tablas `nodos`, `aristas` y `marcadores`): se exporta a `data/*.csv` y se regenera con `python ia/herramientas/generar_grafo.py`.
- Ids: se usan los ids de Neon (1 a 35). El N viejo de la planilla "tablas nodos" queda en `numero`. Los nodos van de a pares gemelos (uno por lado del pasillo, en el mismo punto).
- Los nombres de `grafo.json` se editaron a mano (sin guiones bajos). Si se corre `generar_grafo.py`, vuelven a tener guiones bajos. No rompe nada: `decidir()` y la búsqueda de destino los cambian por espacios.
- Ángulos: este=0, norte=90, oeste=180, sur=270 (x crece al este, y al norte). Positivo = antihorario, o sea que girar a la izquierda suma grados.
- `grados` de un cartel = hacia dónde mira su frente. Es lo opuesto a su `pared`.
- Cada cartel tiene un código M01..M19 (hoja "carteles_ivan" de la planilla). Las carpetas de la galería se llaman así.

## Estructura
- `ia/main.py`: el pipeline y el motor de decisiones. No se corre solo: `procesar_frame(frame, estado)` devuelve `(texto, debug)` y lo llama el servidor.
- `ia/servidor.py`: la app FastAPI con `POST /frame` y `POST /destino` y el único `Estado` en memoria. Se levanta desde la raíz con `uvicorn ia.servidor:app --host 127.0.0.1 --port 8000`; en `http://127.0.0.1:8000/docs` se prueban los endpoints a mano.
- `ia/embeddings.py`: `enderezar()` (recorte de 160x190) y `vector()` (MobileNetV2). Lo usan la galería y `identificar()`; los dos tienen que hacer exactamente lo mismo.
- `ia/herramientas/`:
  - Scripts de PC: `generar_grafo.py`, `generar_galeria.py` (recortar / vectores), `marcar_a_mano.py`, `probar_detectar.py`.
  - Usados por `main.py`: `modelo_esquinas.py` (esquinas de Ivan) y `ruteo.py` (Dijkstra). Si se cambian, cambia el pipeline.`destino.py` (texto -> destino) lo usa `servidor.py`.
  - Demos de clase: el resto.
- `voz/backend/`: servidor Node de Tincho. Las credenciales van en `voz/backend/.env`, que no se sube.
- `voz/cliente.py`: stub viejo. Con la arquitectura nueva, el cliente de la raspi lo hace Tincho.
- `data/`: `grafo.json`, `galeria.npz` (158 vectores de los 19 carteles), las 3 tablas de Neon en CSV, `calibracion.npz`, `dataset.yaml`.
- `models/best.pt`: YOLO11n entrenado (mAP50 ≈ 0.92 en val d1+d2). Es el único peso que se sube a git.
- `docs/cvat/`: docker-compose de CVAT.

## Galería de embeddings
Las fotos están en `C:\VisionAI\embeddings\galeria\M01..M19` (fuera del repo). Para regenerar `data/galeria.npz`, desde la raíz:
1. `python -m ia.herramientas.generar_galeria recortar C:/VisionAI/embeddings/galeria`
2. Revisar los recortes en `galeria_recortes/`: borrar los del cartel equivocado, agregar su foto a `a_mano.txt` y marcarlas con `python -m ia.herramientas.marcar_a_mano C:/VisionAI/embeddings/galeria` (clicks: arriba-izq, arriba-der, abajo-der, abajo-izq).
3. `python -m ia.herramientas.generar_galeria vectores C:/VisionAI/embeddings/galeria`

## Plan y reparto
**Primero (desbloquean al resto)**
1. Acordar entre los tres el contrato de la API (la propuesta está arriba).
2. Ivan: calibración real, con la misma cámara y a 1280x720, en `data/calibracion.npz`.
3. Subir `data/galeria.npz` al repo.

**IA (Lorenzo e Ivan)**
- Meter la calibración real en `main.py` (hoy usa la K nominal `K_CAMARA`) y verificar solvePnP con fotos tomadas desde posiciones conocidas. Es el control más importante: un solvePnP malo da instrucciones equivocadas.
- Hecho: elección del cartel del frame, texto -> destino, endpoints FastAPI y casos nuevos en `decidir()`.

**Backend (Tincho)**
- `POST /audio` y `POST /frame` como en el contrato, sin el chat de IA ni `/destinos/buscar`.
- TTS devuelto en la misma respuesta.
- Dirección fija del servidor de ORT. Probar que la raspi lo alcanza desde el WiFi de la demo, y que el servidor tiene salida a internet (STT y TTS usan servicios externos).
- Script que arranque juntos la IA (`uvicorn`) y Node (`npm start`).

**Raspi (Tincho, con Ivan en cámara y botón)**
- Cliente Python: manda frames con Picamera2 a 1280x720 desde que se enciende; graba mientras el botón está apretado y manda el audio; reproduce sin pisar un audio que todavía suena; arranca solo al encender (systemd).
- Raspberry Pi OS Lite de 64 bits: `dpkg --print-architecture` tiene que dar `arm64`.

**Integración**
1. Probar la tubería completa con una IA falsa que devuelva siempre el mismo texto (valida raspi, backend, red y TTS).
2. Conectar la IA real y probar en un tramo con carteles físicos.
3. Medir el error de posición en metros (con `debug`), las instrucciones correctas y la latencia total. No hay objetivo de tiempo: lo importante es que ande.

## Pendientes conocidos
- La calibración de Ivan (`data/calibracion.npz`) es de la Pi Camera en modo 1640x1232 -> 1280x720. Solo vale con ese modo y sin tocar el foco del lente; si se toca, hay que recalibrar. El cliente de la raspi tiene que sacar las fotos en ese modo.
- En Neon faltan 6 aristas de los gemelos lab_1 (19) y aula_214 (29). Mientras tanto las agrega `ARISTAS_FALTANTES` en `generar_grafo.py`.
- El código de esquinas de Ivan encuentra las esquinas en ~45% de las detecciones (falla con carteles blancos o cortados por el borde).
- La galería tiene pocas fotos de algunos carteles (M15 tiene 2) y pocas nítidas.
- `requirements-raspi.txt` es de la arquitectura vieja (onnxruntime en la raspi); hay que revisarlo para el cliente nuevo.
- Si Node manda un pedido mal armado (sin el campo `imagen` o sin `texto`), FastAPI responde `422`, no `200`.

## Reglas
- No subir el dataset ni imágenes al repo. Los datos están en `C:\VisionAI` y en Roboflow.
- Las credenciales (Neon, API key de ia-tic) van solo en `.env`, nunca en el código ni en `.env.example`.
- Código y comentarios en español, con el estilo simple de los scripts existentes.
