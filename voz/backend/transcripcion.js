import { openAsBlob } from 'node:fs';

async function transcribirAudio() {
  const url = "https://ia-tic.ort.edu.ar/api/v1/audio/transcriptions";
  const apiKey = process.env.IA_TIC_API_KEY; // va en voz/backend/.env
  const rutaArchivo = "./grabacion.m4a"; // Ruta a tu archivo local

  // 1. Cargar el archivo local como Blob
  const audioBlob = await openAsBlob(rutaArchivo);

  // 2. Crear el FormData
  const formData = new FormData();
  formData.append("file", audioBlob, "/grabacion.m4a");
  formData.append("model", "whisper-1"); // O el nombre de modelo que requiera la API

  // 3. Enviar la petición
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
    },
    body: formData,
  });

  const data = await response.json();
  console.log("Transcripción:", data.text);
}

transcribirAudio();
