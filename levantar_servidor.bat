@echo off
rem Levanta la IA (uvicorn) y Node juntos, cada uno en su ventana.
cd /d "%~dp0"
start "VisionAI - IA" cmd /k "venv\Scripts\python.exe -m uvicorn ia.servidor:app --host 0.0.0.0 --port 8000"
start "VisionAI - Node" cmd /k "cd voz\backend && npm start"
echo.
echo IP de esta PC (la que tiene que usar la raspi):
ipconfig | findstr /i "IPv4"
pause