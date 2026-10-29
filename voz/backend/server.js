import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'node:fs';
import gTTS from 'gtts';

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(express.json());

// Variables de entorno con valores por defecto
const PORT = process.env.PORT || 3000;
const IA_URL = process.env.IA_URL || "http://localhost:8000";
const ORT_API_KEY = process.env.ORT_API_KEY;   // va solo en el .env, nunca en el código
if (!ORT_API_KEY) console.warn("⚠️ Falta ORT_API_KEY en el .env: /audio no va a poder transcribir");
const ORT_TRANSCRIBE_URL = process.env.ORT_TRANSCRIBE_URL || "https://ia-tic.ort.edu.ar/api/v1/audio/transcriptions";
const ORT_MODEL_TRANSCRIPT = process.env.ORT_MODEL_TRANSCRIPT || "tic-transcript";

// Función auxiliar para convertir texto a MP3 mediante gTTS
function generarAudioMP3(texto, rutaSalida) {
  return new Promise((resolve, reject) => {
    const gtts = new gTTS(texto, 'es');
    gtts.save(rutaSalida, (err) => (err ? reject(err) : resolve(rutaSalida)));
  });
}

// Auxiliar para fetch con timeout
async function fetchConTimeout(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}

// ====================================================================
// 1. ENDPOINT: /frame (Recepción de fotos periódicas ~1/s)
// ====================================================================
app.post('/frame', upload.single('imagen'), async (req, res) => {
  let rutaImagenTemp = req.file?.path;
  let mp3SalidaTemp = null;

  try {
    if (!rutaImagenTemp) {
      return res.status(204).send();
    }

    // Preparar multipart/form-data para la IA de Lorenzo con el campo 'imagen'
    const imageBuffer = fs.readFileSync(rutaImagenTemp);
    const formData = new FormData();
    formData.append("imagen", new Blob([imageBuffer]), req.file.originalname || "frame.jpg");

    // Reenviar a la IA de Lorenzo con timeout de 5s
    const responseIA = await fetchConTimeout(`${IA_URL}/frame`, {
      method: "POST",
      body: formData
    }, 5000);

    if (!responseIA.ok) {
      console.warn(`⚠️ IA /frame devolvió estado ${responseIA.status}`);
      return res.status(204).send();
    }

    const dataIA = await responseIA.json();
    const textoAudio = dataIA.texto;

    // Si texto es null o está vacío -> 204 No Content
    if (!textoAudio || textoAudio.trim() === "") {
      return res.status(204).send();
    }

    console.log(`🔊 [Frame] Generando audio: "${textoAudio}"`);
    mp3SalidaTemp = `frame_${Date.now()}.mp3`;
    await generarAudioMP3(textoAudio, mp3SalidaTemp);

    const bufferMp3 = fs.readFileSync(mp3SalidaTemp);
    res.setHeader('Content-Type', 'audio/mpeg');
    return res.send(bufferMp3);

  } catch (error) {
    // Si la IA falla, expira el timeout o no está levantada, se trata como texto null (204)
    console.warn("⚠️ Error o timeout al procesar /frame en la IA:", error.message);
    return res.status(204).send();
  } finally {
    if (rutaImagenTemp && fs.existsSync(rutaImagenTemp)) fs.unlinkSync(rutaImagenTemp);
    if (mp3SalidaTemp && fs.existsSync(mp3SalidaTemp)) fs.unlinkSync(mp3SalidaTemp);
  }
});

// ====================================================================
// 2. ENDPOINT: /audio (Procesamiento de voz al presionar el botón)
// ====================================================================
app.post('/audio', upload.single('audio'), async (req, res) => {
  let rutaAudioTemp = req.file?.path;
  let mp3SalidaTemp = null;

  try {
    let textoTranscripto = "";

    if (rutaAudioTemp) {
      // Step 1: STT con API de ORT
      try {
        const audioBuffer = fs.readFileSync(rutaAudioTemp);
        const formDataSTT = new FormData();
        formDataSTT.append("file", new Blob([audioBuffer]), req.file.originalname || "audio.wav");
        formDataSTT.append("model", ORT_MODEL_TRANSCRIPT);

        const resSTT = await fetchConTimeout(ORT_TRANSCRIBE_URL, {
          method: "POST",
          headers: { "Authorization": `Bearer ${ORT_API_KEY}` },
          body: formDataSTT
        }, 8000);

        if (resSTT.ok) {
          const dataSTT = await resSTT.json();
          textoTranscripto = dataSTT.text ?? dataSTT.transcript ?? "";
          console.log(`💬 [Audio STT]: "${textoTranscripto}"`);
        } else {
          console.warn(`⚠️ Error en STT ORT (Estado ${resSTT.status})`);
        }
      } catch (errSTT) {
        console.warn("⚠️ Fallo o timeout en STT ORT:", errSTT.message);
      }
    }

    // Step 2: Enviar texto a la IA de Lorenzo en /destino
    let fraseRespuesta = "No te entendí. ¿A dónde querés ir?";

    try {
      const responseIA = await fetchConTimeout(`${IA_URL}/destino`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texto: textoTranscripto })
      }, 5000);

      if (responseIA.ok) {
        const dataIA = await responseIA.json();
        if (dataIA.frase) {
          fraseRespuesta = dataIA.frase;
        }
      } else {
        console.warn(`⚠️ IA /destino devolvió estado ${responseIA.status}`);
      }
    } catch (errIA) {
      console.warn("⚠️ Error o timeout al consultar IA /destino:", errIA.message);
    }

    // Step 3: Convertir fraseRespuesta a MP3 con gTTS
    console.log(`🔊 [Audio Respuesta]: "${fraseRespuesta}"`);
    mp3SalidaTemp = `destino_${Date.now()}.mp3`;
    await generarAudioMP3(fraseRespuesta, mp3SalidaTemp);

    const bufferMp3 = fs.readFileSync(mp3SalidaTemp);
    res.setHeader('Content-Type', 'audio/mpeg');
    return res.send(bufferMp3);

  } catch (error) {
    console.error("❌ Error grave en /audio:", error.message);
    // Fallback asegurado para no dejar al usuario sin respuesta hablada
    try {
      mp3SalidaTemp = `error_${Date.now()}.mp3`;
      await generarAudioMP3("No te entendí. ¿A dónde querés ir?", mp3SalidaTemp);
      const bufferMp3 = fs.readFileSync(mp3SalidaTemp);
      res.setHeader('Content-Type', 'audio/mpeg');
      return res.send(bufferMp3);
    } catch {
      return res.status(500).json({ error: "Error en síntesis de audio" });
    }
  } finally {
    if (rutaAudioTemp && fs.existsSync(rutaAudioTemp)) fs.unlinkSync(rutaAudioTemp);
    if (mp3SalidaTemp && fs.existsSync(mp3SalidaTemp)) fs.unlinkSync(mp3SalidaTemp);
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Servidor Node (Voz) listo en http://localhost:${PORT}`);
});