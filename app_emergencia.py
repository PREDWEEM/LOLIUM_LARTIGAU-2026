# -*- coding: utf-8 -*-
"""
PREDWEEM Lolium — Lartigau.

La página muestra la alerta de grandes flujos de Lolium del paquete ``lolium-comun`` (probabilidad de gran flujo en
los próximos 7 días, qué hacer esta semana, ventana del graminicida de barbecho, residual a la siembra y flujos tardíos)
con el clima diario del repositorio (``meteo_daily.csv``).

El motor PREDWEEM anterior (red neuronal y balance hídrico, ``app_emergencia_core.py`` y sus modelos) ya no se
ejecuta: su inicio estimado de la emergencia erró 12–14 días en promedio y no mejoraba la alerta. Sus archivos quedan
en el repositorio solo como referencia.
"""
import importlib
from pathlib import Path

import pandas as pd
import streamlit as st

# Control de acceso (ingreso con Google y usuarios aprobados). Inactivo hasta configurar los secretos [acceso] y [auth].
from acceso_predweem import mostrar_sesion, requerir_acceso
from lolium_comun import pagina

BASE = Path(__file__).resolve().parent

requerir_acceso("Lartigau")
pagina.encabezado("🌾 PREDWEEM LOLIUM — LARTIGAU (BA, lat=-38.6166; lon=-61.7000)", "PREDWEEM Lartigau")
parametros = pagina.barra_lateral(BASE, "Lartigau")
meteo = pagina.cargar_meteo(BASE, None)

NOTA_LARTIGAU = (
    "**Validación en Lartigau.** El modelo se probó, sin reajustarlo, con los conteos de "
    "Lartigau 2026 (15 fechas entre el 01/02 y el 30/08; 14 intervalos de 5 a 21 días; 3 con emergencia ≥ 5 % de la "
    "campaña por semana). Emitida al comienzo de cada intervalo con el clima de esta app, la alerta separó bien los "
    "intervalos con gran flujo: AUC 0,88 solo con clima y 0,94 con la fecha de primeras plántulas (19/02). Los tres grandes flujos "
    "quedaron en *Alerta*; también quedaron en *Alerta* 4 de los 11 intervalos sin gran flujo.\n\n"
    "**Límites en Lartigau.** Una sola campaña. El clima es el de MeteoBahía para Coronel Falcón (pronósticos archivados, "
    "no observaciones de estación). En 2026 la emergencia fue más temprana que el calendario de referencia regional (65 % al 20/03 "
    "contra 39 % esperado). La alerta con conteos está calibrada con conteos semanales: con recorridas cada 2–3 semanas, "
    "conviene usar sobre todo la fecha de primeras plántulas."
)


_resultados_alerta = None
if meteo is not None and not meteo.empty:
    try:
        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        _resultados_alerta = _vis_alerta.mostrar_alerta_lolmu(
            meteo,
            **parametros,
    tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
    curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
    curva_tiempo_termico=True,  # con primeras plántulas, curva acumulada en tiempo térmico desde esa fecha
    localidad="Lartigau",
    nota_sitio=NOTA_LARTIGAU,
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta: {_error_alerta}")

pagina.descarga_excel(_resultados_alerta, "Lartigau", "descarga_excel_resultados_final_lartigau")
mostrar_sesion()
