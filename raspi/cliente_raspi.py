"""
Cliente de VisionAI en la Raspberry Pi: cámara, botón y audio.
- Saca una foto por segundo y la manda a Node (POST /frame). Si vuelve audio, lo reproduce.
- Mientras se aprieta el botón graba la voz; al soltarlo la manda a Node (POST /audio)
  y reproduce la respuesta.

Configuración por variables de entorno (si no están, se usan los valores de abajo):
  SERVIDOR    dirección de Node, por ejemplo http://10.8.5.126:3000
  MICROFONO   dispositivo de arecord ("default", o el de la placa USB, ej. plughw:CARD=Device)
  PARLANTE    dispositivo de salida para mpg123 (vacío = el de siempre)

Uso:  SERVIDOR=http://IP_DEL_SERVIDOR:3000 python3 cliente_raspi.py
"""

import os
import signal
import subprocess
import threading
import time

import cv2
import requests
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ====================================================================
# CONFIGURACIÓN
# ====================================================================
SERVIDOR_URL = os.environ.get("SERVIDOR", "http://192.168.1.50:3000")
MICROFONO = os.environ.get("MICROFONO", "default")
PARLANTE = os.environ.get("PARLANTE", "")
PIN_BOTON = 17                 # Pin GPIO donde está conectado el botón
GRABACION_MAXIMA_S = 8         # si el botón se queda apretado, corta igual

# Configuración del botón físico
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_BOTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Cámara: la MISMA configuración con la que Ivan calibró (ia/herramientas/calibracioncamara.py).
# Si se cambia, la calibración del servidor deja de valer.
camara = Picamera2()
camara.configure(camara.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    raw={"size": (1640, 1232)},
))
camara.start()
time.sleep(2)   # que ajuste la exposición

# Flags de control
procesando_audio = False
candado_parlante = threading.Lock()   # un audio a la vez: la guía y la respuesta no se pisan

# ====================================================================
# FUNCIONES AUXILIARES DE AUDIO Y CÁMARA
# ====================================================================

def reproducir_audio(datos_mp3, ruta_mp3):
    """Guarda el MP3 que mandó Node y lo reproduce (espera si ya hay otro sonando)."""
    with open(ruta_mp3, "wb") as f:
        f.write(datos_mp3)
    cmd = ["mpg123", "-q"] + (["-o", "alsa", "-a", PARLANTE] if PARLANTE else []) + [ruta_mp3]
    with candado_parlante:
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"❌ Error al reproducir audio: {e}")


def capturar_jpeg():
    """Saca una foto de 1280x720 y la devuelve como JPEG (bytes)."""
    frame = camara.capture_array()
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes() if ok else None


def grabar_mientras_apretado(ruta_wav):
    """Graba desde que se aprieta el botón hasta que se suelta (como máximo GRABACION_MAXIMA_S)."""
    proceso = subprocess.Popen(
        ["arecord", "-D", MICROFONO, "-f", "S16_LE", "-r", "16000", "-c", "1",
         "-d", str(GRABACION_MAXIMA_S), ruta_wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while GPIO.input(PIN_BOTON) == GPIO.LOW and proceso.poll() is None:
        time.sleep(0.05)
    if proceso.poll() is None:
        proceso.send_signal(signal.SIGINT)   # arecord cierra bien el .wav con Ctrl+C
    proceso.wait()

# ====================================================================
# HILO 1: TRANSMISIÓN DE FOTOGRAMAS (~1 foto por segundo)
# ====================================================================

def bucle_frames():
    print(f"📷 Transmisión de cámara iniciada hacia {SERVIDOR_URL}")

    while True:
        inicio = time.monotonic()
        # Si la persona está hablando, pausar la cámara para no saturar
        if not procesando_audio:
            try:
                jpeg = capturar_jpeg()
                if jpeg:
                    files = {"imagen": ("frame.jpg", jpeg, "image/jpeg")}
                    response = requests.post(f"{SERVIDOR_URL}/frame", files=files, timeout=8)

                    # 200 con contenido -> hay una indicación que reproducir; 204 -> nada que decir
                    if response.status_code == 200 and response.content:
                        print("🔊 Reproduciendo guía de navegación...")
                        reproducir_audio(response.content, "instruccion.mp3")
            except Exception as e:
                # Cualquier error (red, cámara) pierde solo esta foto: el bucle sigue
                print(f"⚠️ Frame perdido: {e}")

        time.sleep(max(0, 1 - (time.monotonic() - inicio)))   # una foto por segundo

# ====================================================================
# HILO 2: GRABACIÓN DE VOZ (Botón físico)
# ====================================================================

def atender_boton():
    global procesando_audio

    while True:
        # Detectar cuando el botón es presionado (LOW por PUD_UP)
        if GPIO.input(PIN_BOTON) == GPIO.LOW and not procesando_audio:
            procesando_audio = True
            print("\n🎤 Botón presionado: grabando hasta que lo sueltes...")
            try:
                audio_path = "consulta.wav"
                grabar_mientras_apretado(audio_path)

                print("📤 Enviando consulta al servidor Node...")
                with open(audio_path, "rb") as aud_file:
                    files = {"audio": ("consulta.wav", aud_file, "audio/wav")}
                    response = requests.post(f"{SERVIDOR_URL}/audio", files=files, timeout=15)

                if response.status_code == 200 and response.content:
                    print("🔊 Reproduciendo respuesta de destino...")
                    reproducir_audio(response.content, "respuesta.mp3")
                else:
                    print(f"⚠️ Servidor respondió con código: {response.status_code}")
            except Exception as e:
                print(f"❌ Error de comunicación en voz: {e}")
            finally:
                procesando_audio = False

        time.sleep(0.05)

# ====================================================================
# INICIO DEL PROGRAMA
# ====================================================================
if __name__ == "__main__":
    try:
        # Arrancar el hilo de la cámara en segundo plano
        thread_camara = threading.Thread(target=bucle_frames, daemon=True)
        thread_camara.start()

        # Correr la escucha del botón en el hilo principal
        atender_boton()

    except KeyboardInterrupt:
        print("\n👋 Deteniendo cliente de la Raspberry Pi...")
    finally:
        camara.stop()
        GPIO.cleanup()