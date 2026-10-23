import { query } from "./db.js";

// Los nodos se leen una sola vez y quedan en memoria.
let nodosCache = null;
async function getNodos() {
  if (!nodosCache) {
    nodosCache = (await query("SELECT id, numero, nombre FROM nodos")).rows;
  }
  return nodosCache;
}

// Llamar si cambiás los nodos en la base mientras el servidor corre.
export function invalidarNodos() {
  nodosCache = null;
}

// ---- Números dichos en voz alta -> dígitos ("doscientos tres" -> "203") ----
const NUMEROS = {
  cero: 0, uno: 1, dos: 2, tres: 3, cuatro: 4, cinco: 5, seis: 6, siete: 7, ocho: 8, nueve: 9,
  diez: 10, once: 11, doce: 12, trece: 13, catorce: 14, quince: 15,
  dieciseis: 16, diecisiete: 17, dieciocho: 18, diecinueve: 19,
  veinte: 20, veintiuno: 21, veintidos: 22, veintitres: 23, veinticuatro: 24,
  veinticinco: 25, veintiseis: 26, veintisiete: 27, veintiocho: 28, veintinueve: 29,
  treinta: 30, cuarenta: 40, cincuenta: 50, sesenta: 60, setenta: 70, ochenta: 80, noventa: 90,
  cien: 100, ciento: 100, doscientos: 200, trescientos: 300, cuatrocientos: 400,
  quinientos: 500, seiscientos: 600, setecientos: 700, ochocientos: 800, novecientos: 900,
};

// Palabras que no ayudan a identificar el lugar
const RELLENO = new Set([
  "quiero", "ir", "a", "al", "la", "el", "los", "las", "de", "del", "llevame", "llevarme",
  "me", "lleves", "llevar", "necesito", "voy", "hacia", "por", "favor", "un", "una", "en",
  "donde", "queda", "esta", "hasta", "para", "vamos", "quisiera", "podes", "puedes", "y",
]);

const normalizar = (t) =>
  t.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "") // saca tildes y la ñ
    .replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();

// ¿Se puede sumar v al número que se viene armando? (200 + 3, 30 + 5, 200 + 15)
const puedeSumar = (total, v) =>
  v > 0 &&
  ((total >= 100 && total % 100 === 0 && v < 100) ||
    (total % 100 >= 20 && total % 10 === 0 && v < 10));

function palabrasANumeros(texto) {
  const tokens = texto.split(" ");
  const salida = [];
  let total = null;
  const cerrar = () => {
    if (total !== null) salida.push(String(total));
    total = null;
  };

  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    const v = NUMEROS[t];
    if (v !== undefined) {
      if (total !== null && puedeSumar(total, v)) total += v;
      else { cerrar(); total = v; }
    } else if (
      t === "y" && total !== null && total % 10 === 0 && total % 100 >= 20 &&
      NUMEROS[tokens[i + 1]] > 0 && NUMEROS[tokens[i + 1]] < 10
    ) {
      continue; // "treinta y cinco"
    } else {
      cerrar();
      salida.push(t);
    }
  }
  cerrar();
  return salida.join(" ");
}

const tokens = (texto) =>
  palabrasANumeros(normalizar(texto)).split(" ").filter((t) => t && !RELLENO.has(t));

// ---- Parecido entre palabras (tolera errores del reconocedor) ----
function distancia(a, b) {
  const fila = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let anterior = fila[0];
    fila[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const temp = fila[j];
      fila[j] = Math.min(fila[j] + 1, fila[j - 1] + 1, anterior + (a[i - 1] === b[j - 1] ? 0 : 1));
      anterior = temp;
    }
  }
  return fila[b.length];
}

function parecido(a, b) {
  if (a === b) return 1;
  const s = 1 - distancia(a, b) / Math.max(a.length, b.length);
  return s >= 0.75 ? s : 0;
}

const mejorCoincidencia = (t, lista) => Math.max(0, ...lista.map((x) => parecido(t, x)));
const promedio = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;

function puntaje(consulta, destino) {
  const tokensNombre = tokens(destino.nombre);
  const tokensMatch = [...tokensNombre, ...tokens(String(destino.numero))];
  const cobConsulta = promedio(consulta.map((t) => mejorCoincidencia(t, tokensMatch)));
  const cobDestino = tokensNombre.length
    ? promedio(tokensNombre.map((t) => mejorCoincidencia(t, consulta)))
    : 0;
  return 0.8 * cobConsulta + 0.2 * cobDestino;
}

/**
 * Recibe el texto reconocido por voz y devuelve el/los destinos más probables.
 * Devuelve null si no encuentra nada razonable.
 */
export async function buscarDestino(texto) {
  const consulta = tokens(texto);
  if (!consulta.length) return null;

  const nodos = await getNodos();
  const candidatos = nodos
    .filter((n) => n.nombre)
    .map((n) => ({ id: n.id, numero: n.numero, nombre: n.nombre, score: puntaje(consulta, n) }))
    .filter((c) => c.score >= 0.6)
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((c) => ({ ...c, score: Number(c.score.toFixed(2)) }));

  if (!candidatos.length) return null;

  const [mejor, segundo] = candidatos;
  const segura = mejor.score >= 0.85 && (!segundo || mejor.score - segundo.score >= 0.15);

  return {
    texto_normalizado: consulta.join(" "),
    confianza: segura ? "alta" : "baja",
    mejor,
    candidatos,
    // Frase lista para leer en voz alta y pedir confirmación
    pregunta: segura
      ? `¿Querés ir a ${mejor.nombre}?`
      : `¿Querés ir a ${candidatos.map((c) => c.nombre).join(" o a ")}?`,
  };
}