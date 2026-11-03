"""
Entrada para el hosting del colegio (HosTICer): ahí se corre `uvicorn main:app` desde la raíz del repo.
La app de verdad está en ia/servidor.py; en la PC se sigue usando levantar_servidor.bat.
"""

import os
import tempfile

# El disco del hosting es de solo lectura salvo /tmp y data/:
# ultralytics y matplotlib tienen que guardar su configuración en la carpeta temporal
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from ia.servidor import app  # noqa: E402,F401