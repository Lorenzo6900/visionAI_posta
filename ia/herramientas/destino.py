"""
Texto -> destino: de lo que dijo la persona (ya transcripto por Node) al nodo del grafo.
Es el port a Python de voz/backend/busquedaService.js, con dos cambios:
  - Se compara solo con el nombre (no con numero) y solo con los nodos es_destino.
  - Si duda, no pregunta: toma el mejor candidato y lo dice ("Voy a llevarte a aula 203").
"""

import re
import unicodedata

PUNTAJE_MINIMO = 0.6       # por debajo de esto el candidato no cuenta (igual que en Node)
PARECIDO_PALABRA = 0.75    # dos palabras se parecen si comparten al menos esto (tolera errores del reconocedor)

NO_ENTENDI = "No te entendí. ¿A dónde querés ir?"

# Números dichos en voz alta -> dígitos ("doscientos tres" -> "203")
NUMEROS = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
    "veinte": 20, "veintiuno": 21, "veintidos": 22, "veintitres": 23, "veinticuatro": 24,
    "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
    "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
    "cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300, "cuatrocientos": 400,
    "quinientos": 500, "seiscientos": 600, "setecientos": 700, "ochocientos": 800, "novecientos": 900,
}

# Palabras que no ayudan a identificar el lugar
RELLENO = {
    "quiero", "ir", "a", "al", "la", "el", "los", "las", "de", "del", "llevame", "llevarme",
    "me", "lleves", "llevar", "necesito", "voy", "hacia", "por", "favor", "un", "una", "en",
    "donde", "queda", "esta", "hasta", "para", "vamos", "quisiera", "podes", "puedes", "y",
}


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes ni ñ, sin signos y con un solo espacio entre palabras."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")   # saca tildes y la ñ
    texto = re.sub(r"[^a-z0-9 ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def puede_sumar(total: int, valor: int) -> bool:
    """¿Se puede sumar valor al número que se viene armando? (200 + 3, 30 + 5, 200 + 15)"""
    return valor > 0 and ((total >= 100 and total % 100 == 0 and valor < 100)
                          or (total % 100 >= 20 and total % 10 == 0 and valor < 10))


def palabras_a_numeros(texto: str) -> str:
    palabras = texto.split(" ")
    salida = []
    total = None

    for i, palabra in enumerate(palabras):
        valor = NUMEROS.get(palabra)
        siguiente = NUMEROS.get(palabras[i + 1]) if i + 1 < len(palabras) else None
        if valor is not None:
            if total is not None and puede_sumar(total, valor):
                total += valor
            else:
                if total is not None:
                    salida.append(str(total))
                total = valor
        elif (palabra == "y" and total is not None and total % 10 == 0 and total % 100 >= 20
              and siguiente is not None and 0 < siguiente < 10):
            continue   # "treinta y cinco"
        else:
            if total is not None:
                salida.append(str(total))
            total = None
            salida.append(palabra)

    if total is not None:
        salida.append(str(total))
    return " ".join(salida)


def palabras_utiles(texto: str) -> list[str]:
    return [p for p in palabras_a_numeros(normalizar(texto)).split(" ") if p and p not in RELLENO]


# ---------- Parecido entre palabras (tolera errores del reconocedor) ----------

def distancia(a: str, b: str) -> int:
    """Cuántas letras hay que agregar, sacar o cambiar para pasar de a a b (Levenshtein)."""
    fila = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        anterior = fila[0]
        fila[0] = i
        for j in range(1, len(b) + 1):
            temp = fila[j]
            fila[j] = min(fila[j] + 1, fila[j - 1] + 1, anterior + (a[i - 1] != b[j - 1]))
            anterior = temp
    return fila[len(b)]


def parecido(a: str, b: str) -> float:
    if a == b:
        return 1.0
    s = 1 - distancia(a, b) / max(len(a), len(b))
    return s if s >= PARECIDO_PALABRA else 0.0


def mejor_coincidencia(palabra: str, lista: list[str]) -> float:
    return max([0.0] + [parecido(palabra, otra) for otra in lista])


def puntaje(consulta: list[str], nombre: str) -> float:
    """Qué tan bien coincide lo que dijo la persona con el nombre de un destino (0 a 1).
    Pesa más que se encuentren las palabras de la consulta en el nombre (80%) que al revés (20%)."""
    palabras_nombre = palabras_utiles(nombre)
    if not palabras_nombre:
        return 0.0
    cob_consulta = sum(mejor_coincidencia(p, palabras_nombre) for p in consulta) / len(consulta)
    cob_destino = sum(mejor_coincidencia(p, consulta) for p in palabras_nombre) / len(palabras_nombre)
    return 0.8 * cob_consulta + 0.2 * cob_destino


def buscar_destino(texto: str, nodos: dict) -> tuple[str, int | None]:
    """De lo que dijo la persona al destino. nodos: GRAFO["nodos"] (id -> nodo).
    Devuelve (frase para decir, id del destino). El id es None si no entendió."""
    consulta = palabras_utiles(texto)
    if not consulta:
        return NO_ENTENDI, None

    candidatos = [(puntaje(consulta, n["nombre"].replace("_", " ")), n)
                  for n in nodos.values() if n["es_destino"] and n["nombre"]]
    candidatos = [(p, n) for p, n in candidatos if p >= PUNTAJE_MINIMO]
    if not candidatos:
        return NO_ENTENDI, None

    # Aunque dude entre dos, toma el mejor: la frase dice a dónde lo va a llevar
    _, mejor = max(candidatos, key=lambda par: par[0])
    return f"Voy a llevarte a {mejor['nombre'].replace('_', ' ')}", mejor["id"]