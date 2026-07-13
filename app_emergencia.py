# -*- coding: utf-8 -*-
"""
Lanzador estable de PREDWEEM Lartigau con el subsistema temporal
validado en vK4.9.15.

El primer pico no se fija por fecha ni se desplaza mediante lag.
Resulta de la ANN y de los filtros hídrico, térmico, de latencia
y de umbral del primer pico definidos en app_emergencia_vK4_9_15.py.
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "app_emergencia_vK4_9_15.py"

if not SOURCE.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo validado: {SOURCE.name}"
    )

codigo = SOURCE.read_text(encoding="utf-8")
exec(
    compile(codigo, str(SOURCE), "exec"),
    {
        "__name__": "__main__",
        "__file__": str(SOURCE),
        "__package__": None,
    },
)
