import sys

# Windows: al redirigir la salida a un archivo, Python usa cp1252 y falla con δ, ⊕, á... (tarea #26).
# Todos los scripts importan este paquete, así que basta con forzar UTF-8 aquí.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
