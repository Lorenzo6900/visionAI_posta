import fs from 'node:fs';
import { exec } from 'node:child_process';
import record from 'node-record-lpcm16';
import { Gpio } from 'onoff';

const CONFIG = {
  transcribeUrl: "https://ia-tic.ort.edu.ar/api/v1/audio/transcriptions",
  chatUrl: "https://ia-tic.ort.edu.ar/api/v1/chat/completions",
  apiKey: "sk-29b4ad5bea2de3a08666b543932c119c191d0dd8d91a9814",
  modelTranscript: "tic-transcript",
  modelChat: "tic-chat"
};

// Configurar el Botón en el GPIO 17 (resistencia de pull-up interna)
const boton = new Gpio(17, 'in', 'falling', { debounceTimeout: 300 });
let estaProcesando = false;

function ejecutarComando(cmd) {
  return new Promise((resolve, reject) => {
    exec(cmd, (err) => (err ? reject(err) : resolve()));
  });
}

// 1. Grabar audio desde el micrófono
function grabarAudio(duracionSegundos = 5) {
  return new Promise((resolve) => {
    const file = fs.createWriteStream('consulta.wav');
    console.log("🎤 Escuchando... Di a dónde quieres ir:");

    const recording = record.record({
      sampleRate: 16000,
      threshold: 0,
      recordProgram: 'rec'
    });

    recording.stream().pipe(file);

    setTimeout(() => {
      recording.stop();
      console.log(" Procesando audio...");
      resolve('consulta.wav');
    }, duracionSegundos * 1000);
  });
}

// 2. Transcribir audio con la API de ORT
async function transcribir(rutaArchivo) {
  const audioBlob = new Blob([fs.readFileSync(rutaArchivo)]);
  const formData = new FormData();
  formData.append("file", audioBlob, "consulta.wav");
  formData.append("model", CONFIG.modelTranscript);

  const response = await fetch(CONFIG.transcribeUrl, {
    method: "POST",
    headers: { "Authorization": `Bearer ${CONFIG.apiKey}` },
    body: formData
  });

  const data = await response.json();
  return data.text ?? data.transcript;
}

// 3. Consultar guía a la IA
async function consultarIA(textoConsulta) {
  console.log(`🤖 Usuario dijo: "${textoConsulta}"`);

  const response = await fetch(CONFIG.chatUrl, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${CONFIG.apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: CONFIG.modelChat,
      messages: [
        {
          role: "system",
          content: "Eres un asistente de voz de guía de espacio. Da instrucciones breves, sencillas y directas para que el usuario sepa cómo llegar a su destino. No uses viñetas ni caracteres especiales."
        },
        { role: "user", content: textoConsulta }
      ]
    })
  });

  const data = await response.json();
  return data.choices[0].message.content;
}

// 4. Reproducir respuesta por el altavoz
async function hablarEnAltavoz(texto) {
  console.log(`🔊 Respondiendo: "${texto}"`);
  const textoLimpio = texto.replace(/["'\n]/g, " ");
  const cmd = `gtts-cli "${textoLimpio}" --lang es --output respuesta.mp3 && mpg123 -q respuesta.mp3`;
  await ejecutarComando(cmd);
}

// --- FLUJO PRINCIPAL ---
async function iniciarAsistente() {
  if (estaProcesando) return; // Evitar llamadas superpuestas si presiona varias veces
  estaProcesando = true;

  try {
    const archivoAudio = await grabarAudio(5);
    const consultaTexto = await transcribir(archivoAudio);

    if (!consultaTexto) {
      await hablarEnAltavoz("No pude entender tu consulta, por favor intenta de nuevo.");
    } else {
      const respuestaIA = await consultarIA(consultaTexto);
      await hablarEnAltavoz(respuestaIA);
    }

  } catch (error) {
    console.error(" Error en el sistema:", error);
  } finally {
    estaProcesando = false;
    console.log("\n Presiona el botón físico para hacer otra consulta...");
  }
}

// Escuchar pulsación del botón GPIO
console.log(" Asistente activo. Presiona el botón en el GPIO 17 para hablar.");
boton.watch((err) => {
  if (err) {
    console.error('Error en GPIO:', err);
    return;
  }
  iniciarAsistente();
});

// Liberar recursos al cerrar el programa (Ctrl + C)
process.on('SIGINT', () => {
  boton.unexport();
  process.exit();
});