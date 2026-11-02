import time
import subprocess
import threading
import requests
import RPi.GPIO as GPIO

# ====================================================================
# CONFIGURACIÓN
# ====================================================================
SERVIDOR_URL = "http://192.168.1.50:3000"  # Cambiar por la IP de tu servidor Node
PIN_BOTON = 17                              # Pin GPIO donde está conectado el botón

# Configuración del botón físico
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN_BOTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Flags de control
procesando_audio = False

# ====================================================================
# FUNCIONES AUXILIARES DE AUDIO Y CÁMARA
# ====================================================================

def reproducir_audio(ruta_mp3):
    """Reproduce el archivo MP3 recibido desde el servidor Node."""
    try:
        # Usa mpg123 en modo silencioso (-q)
        subprocess.run(["mpg123", "-q", ruta_mp3], check=True)
    except Exception as e:
        print(f"❌ Error al reproducir audio: {e}")

def capturar_foto(ruta_salida="frame.jpg"):
    """Toma una foto rápida de la cámara de la Pi (1280x720)."""
    try:
        # Usa libcamera-still (Raspberry Pi OS Bullseye/Bookworm)
        cmd = [
            "libcamera-still",
            "-o", ruta_salida,
            "--immediate", "-n",
            "--width", "1280",
            "--height", "720"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ruta_salida
    except Exception as e:
        print(f"❌ Error al capturar foto: {e}")
        return None

# ====================================================================
# HILO 1: TRANSMISIÓN DE FOTOGRAMAS (~1 foto por segundo)
# ====================================================================

def bucle_frames():
    global procesando_audio
    print("📷 Transmisión de cámara iniciada...")
    
    while True:
        # Si la persona está hablando/escuchando voz, pausar la cámara para no saturar
        if not procesando_audio:
            foto_path = capturar_foto("frame.jpg")
            
            if foto_path:
                try:
                    with open(foto_path, "rb") as img_file:
                        files = {"imagen": ("frame.jpg", img_file, "image/jpeg")}
                        response = requests.post(f"{SERVIDOR_URL}/frame", files=files, timeout=4)
                    
                    # Si el servidor responde 200 OK -> Hay una indicación de voz que reproducir
                    if response.status_code == 200 and response.content:
                        with open("instruccion.mp3", "wb") as f:
                            f.write(response.content)
                        
                        print("🔊 Reproduciendo guía de navegación...")
                        reproducir_audio("instruccion.mp3")

                except requests.exceptions.RequestException:
                    pass  # Ignorar timeouts o caídas breves de red sin cortar el bucle
                
        time.sleep(1)  # Esperar 1 segundo antes del próximo frame

# ====================================================================
# HILO 2: GRABACIÓN DE VOZ (Botón físico)
# ====================================================================

def atender_boton():
    global procesando_audio
    
    while True:
        # Detectar cuando el botón es presionado (LOW por PUD_UP)
        if GPIO.input(PIN_BOTON) == GPIO.LOW:
            if not procesando_audio:
                procesando_audio = True
                print("\n🎤 Botón presionado: Grabando consulta (4 segundos)...")
                
                audio_path = "consulta.wav"
                
                # Grabar 4 segundos con arecord (16kHz, 16bits, Mono)
                cmd_rec = ["arecord", "-D", "default", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", "4", audio_path]
                subprocess.run(cmd_rec, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                print(" Enviando consulta al servidor Node...")
                try:
                    with open(audio_path, "rb") as aud_file:
                        files = {"audio": ("consulta.wav", aud_file, "audio/wav")}
                        response = requests.post(f"{SERVIDOR_URL}/audio", files=files, timeout=8)
                    
                    if response.status_code == 200 and response.content:
                        with open("respuesta.mp3", "wb") as f:
                            f.write(response.content)
                        
                        print("🔊 Reproduciendo respuesta de destino...")
                        reproducir_audio("respuesta.mp3")
                    else:
                        print(f"⚠️ Servidor respondió con código: {response.status_code}")

                except Exception as e:
                    print(f"❌ Error de comunicación en voz: {e}")
                finally:
                    # Esperar a que el usuario suelte el botón para no duplicar grabaciones
                    while GPIO.input(PIN_BOTON) == GPIO.LOW:
                        time.sleep(0.1)
                    procesando_audio = False
                    
        time.sleep(0.1)

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
        GPIO.cleanup()