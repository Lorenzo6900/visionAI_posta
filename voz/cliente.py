"""
Cliente de voz: habla con el backend de Tincho por HTTP.
Por ahora es un stub: imprime en pantalla en vez de mandar audio.
"""


def hablar(texto: str) -> None:
    # TODO (Tincho): POST del texto al backend y reproducir el audio que devuelve
    print(f"[VOZ] {texto}")


def escuchar() -> str:
    # TODO (Tincho): grabar del micrófono, mandarlo al backend y devolver el texto
    return "aula 5"
