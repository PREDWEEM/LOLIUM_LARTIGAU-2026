# -*- coding: utf-8 -*-
"""
Punto de entrada de PREDWEEM Lartigau.

La aplicación científica original se conserva en ``app_emergencia_core.py``. Antes de ejecutarla:

* ``parchear_visualizacion_intensidad_relativa`` incorpora al motor el decaimiento desde el 15 de abril
  (y ajustes de formato de su gráfico);
* se retiran de la página los elementos que reemplaza la alerta de flushes del final: la sección
  "Datos del lote" (se usa el clima MeteoBahía del repositorio y la cobertura de rastrojo por defecto),
  las métricas superiores, los mensajes de inicio, las pestañas (Monitor de decisión, Agua y suelo,
  Análisis estratégico, Bio-calibración y Exportar), la leyenda de versión, la imagen de fondo
  (reemplazada por un fondo liso claro) y, en la barra lateral, los parámetros del motor (quedan en sus
  valores por defecto) y sus avisos; la barra lateral conserva los parámetros de la ventana de control.

El motor sigue calculando como antes: su primer pico se usa como inicio estimado del reloj térmico hasta
que se informa la fecha de las primeras plántulas. La alerta (``alerta_lolmu.py``) es la calibrada con
17 campañas de Bordenave y validada con los conteos de Lartigau 2026.
"""
from pathlib import Path

from visualizacion_horizonte_pronostico import mostrar_horizonte_pronostico
from visualizacion_intensidad_relativa import parchear_visualizacion_intensidad_relativa

# Control de acceso (ingreso con Google y usuarios aprobados). Inactivo hasta configurar los secretos [acceso] y [auth].
import streamlit as st
from acceso_predweem import mostrar_sesion, requerir_acceso

requerir_acceso("Lartigau")

_CORE_APP = Path(__file__).with_name("app_emergencia_core.py")
_core_source = _CORE_APP.read_text(encoding="utf-8")
_core_source = parchear_visualizacion_intensidad_relativa(_core_source)


def _reemplazar(origen: str, viejo: str, nuevo: str) -> str:
    if viejo not in origen:
        raise RuntimeError(f"No se encontró el bloque a reemplazar en el motor: {viejo[:70]!r}")
    return origen.replace(viejo, nuevo, 1)


def _cortar(origen: str, desde: str, hasta: str | None, nuevo: str) -> str:
    """Reemplaza el tramo [desde, hasta) del motor (hasta el final si ``hasta`` es None)."""
    ini = origen.find(desde)
    fin = len(origen) if hasta is None else origen.find(hasta, ini)
    if ini < 0 or fin < ini:
        raise RuntimeError(f"No se encontró el tramo a retirar del motor: {desde[:70]!r}")
    return origen[:ini] + nuevo + origen[fin:]


_FONDO_CSS = """<style>
.stApp { background: #f4f6f3; }
[data-testid="stAppViewContainer"] > .main, [data-testid="stMain"] { background: transparent; }
[data-testid="stSidebar"] { background-color: #e8f3ea; border-right: 1px solid #cfe3d3; }
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #14532d !important; }
[data-testid="stCaptionContainer"], .stCaption { color: #4b5563 !important; }
h1, h2, h3, h4 { color: #1f2937; }
</style>"""

# fondo liso claro en lugar de la imagen de plantas (mejor contraste)
_core_source = _reemplazar(_core_source, 'set_background("fondo_predweem_v3.png")',
                           "st.markdown(_FONDO_CSS, unsafe_allow_html=True)")
# la regla "header {visibility: hidden;}" ocultaba también el encabezado de los calendarios desplegables
_core_source = _reemplazar(_core_source, "        header {visibility: hidden;}",
                           '        header[data-testid="stHeader"] {visibility: hidden;}')
# leyenda de versión del motor
_core_source = _cortar(_core_source, "st.caption(\n    \"vK4.9.15 Adaptada", "with st.expander(\"📂 1. Datos del lote\"", "")

# "Datos del lote" y barra lateral del motor: clima del repositorio, sin conteos del formato anterior,
# cobertura y parámetros del motor por defecto; en la barra lateral, solo la ventana de control
_core_source = _cortar(_core_source, 'with st.expander("📂 1. Datos del lote", expanded=True):', "if ann_model is None:", '''coverage_percent = COBERTURA_PREDETERMINADA
ke_value, thermal_modulator = surface_parameters(coverage_percent)
weather_raw = load_data(None, "meteo_daily")
field_raw = None

logo_path = BASE / "logo.png"
if logo_path.is_file():
    st.sidebar.image(str(logo_path), width="stretch")
else:
    st.sidebar.markdown("## 🌾 PREDWEEM")
    st.sidebar.caption("Motor predictivo Lolium — Lartigau")

st.sidebar.markdown("## ⚙️ Ventana de control")
st.sidebar.caption("Tiempo térmico sobre la temperatura media del aire y objetivos de la ventana post-emergente.")
_col_t1, _col_t2 = st.sidebar.columns(2)
with _col_t1:
    t_base = st.number_input("T Base", value=T_BASE_PREDETERMINADA, step=0.5)
with _col_t2:
    t_optimum = st.number_input("T Óptima Max", value=T_OPTIMA_PREDETERMINADA, step=1.0)
t_critical = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, float(T_CRITICA_PREDETERMINADA))
st.sidebar.markdown("**Objetivos (°Cd)**")
tt_control = st.sidebar.number_input("TT Control Post-emergente (°Cd)", min_value=0, value=TT_CONTROL_PREDETERMINADO, step=10)
tt_limit = st.sidebar.number_input("Límite Ventana (°Cd)", min_value=0, value=TT_LIMITE_PREDETERMINADO, step=10)

# parámetros del motor PREDWEEM en sus valores por defecto (controles retirados de la página)
alert_threshold = 0.0001
residual_days = 0
w_max_value = WMAX_PREDETERMINADO
kr_exponent = EXPONENTE_KR_PREDETERMINADO
thermoinhibition_threshold = UMBRAL_TERMINHIBICION
hydric_shock_threshold = UMBRAL_CHOQUE_HIDRICO_MM

''')

# a comienzos de la campaña (1 de enero) el motor no tiene aún 30 días de clima: aviso en lugar del error
_core_source = _reemplazar(_core_source, "except Exception as exc:\n    st.exception(exc)\n    st.stop()", '''except Exception as exc:
    if "30 días" in str(exc):
        st.info("Campaña recién iniciada: la alerta y el reloj térmico se muestran cuando el clima de la campaña "
                "reúne al menos 30 días (desde fines de enero).")
    else:
        st.exception(exc)
    st.stop()''')

# métricas superiores, mensajes de inicio, pestañas, exportación y leyenda final del motor
_core_source = _cortar(_core_source, "top_metrics = st.columns(5)", None, "")

exec(
    compile(_core_source, str(_CORE_APP), "exec"),
    globals(),
)

if "simulation" in globals() and isinstance(simulation, pd.DataFrame) and not simulation.empty:
    st.divider()
    mostrar_horizonte_pronostico(simulation, "Lartigau")

NOTA_LARTIGAU = (
    "**Validación en Lartigau.** El modelo se calibró en Bordenave y se probó, sin reajustarlo, con los conteos de "
    "Lartigau 2026 (15 fechas entre el 01/02 y el 30/08; 14 intervalos de 5 a 21 días; 3 con emergencia ≥ 5 % de la "
    "campaña por semana). Emitida al comienzo de cada intervalo con el clima de esta app, la alerta separó bien los "
    "intervalos con gran flujo: AUC 0,88 solo con clima y 0,94 con la fecha de primeras plántulas (19/02). Los tres grandes flujos "
    "quedaron en *Alerta*; también quedaron en *Alerta* 4 de los 11 intervalos sin gran flujo.\n\n"
    "**Límites en Lartigau.** Una sola campaña. El clima es el de MeteoBahía para Coronel Falcón (pronósticos archivados, "
    "no observaciones de estación). En 2026 la emergencia fue más temprana que el calendario de Bordenave (65 % al 20/03 "
    "contra 39 % esperado). La alerta con conteos está calibrada con conteos semanales: con recorridas cada 2–3 semanas, "
    "conviene usar sobre todo la fecha de primeras plántulas."
)

# Alerta de flushes (independiente del motor científico; no modifica sus resultados)
if "weather_raw" in globals() and isinstance(weather_raw, pd.DataFrame) and not weather_raw.empty:
    st.divider()
    try:
        import importlib

        import alerta_lolmu as _alerta_lolmu
        import visualizacion_alerta_lolmu as _vis_alerta

        # Streamlit conserva en memoria los módulos ya importados entre ejecuciones: tras una actualización del
        # repositorio, recargarlos evita usar una versión vieja
        importlib.reload(_alerta_lolmu)
        importlib.reload(_vis_alerta)
        mostrar_alerta_lolmu = _vis_alerta.mostrar_alerta_lolmu

        _resultados_alerta = mostrar_alerta_lolmu(
            weather_raw,
            inicio_predicho=globals().get("first_peak_date"),
            t_base=float(globals().get("t_base", 2.0)),
            t_opt=float(globals().get("t_optimum", 20.0)),
            t_crit=float(globals().get("t_critical", 30.0)),
            tt_control=float(globals().get("tt_control", 600.0)),
            tt_limite=float(globals().get("tt_limit", 800.0)),
            tope_hasta_ultimo_cero=True,  # semanas ya cerradas por un conteo posterior en cero: probabilidad 0
            curva_desde_ultimo_cero=True,  # la curva acumulada arranca a mitad de camino entre el último cero y el primer conteo con plántulas
            curva_tiempo_termico=True,  # con primeras plántulas, curva acumulada en tiempo térmico desde esa fecha
            localidad="Lartigau",
            nota_sitio=NOTA_LARTIGAU,
        )
    except Exception as _error_alerta:  # noqa: BLE001
        st.warning(f"No se pudo calcular la alerta: {_error_alerta}")


def _escribir_hoja(writer, dataframe, nombre):
    """Escribe una hoja y aplica un formato básico de lectura."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return
    dataframe.to_excel(writer, sheet_name=nombre, index=False)
    hoja = writer.sheets[nombre]
    ultima_columna = max(0, len(dataframe.columns) - 1)
    hoja.freeze_panes(1, 0)
    hoja.autofilter(0, 0, len(dataframe), ultima_columna)
    hoja.set_column(0, ultima_columna, 18)


# Descarga final: los resultados que muestra la app (alerta, emergencia acumulada y ventana de control)
_resultados_alerta = globals().get("_resultados_alerta")
if isinstance(_resultados_alerta, dict) and _resultados_alerta:
    reporte_excel_final = io.BytesIO()
    with pd.ExcelWriter(reporte_excel_final, engine="xlsxwriter", datetime_format="dd/mm/yyyy", date_format="dd/mm/yyyy") as writer:
        for _hoja, _tabla in _resultados_alerta.items():
            _escribir_hoja(writer, _tabla, _hoja)
    reporte_excel_final.seek(0)
    _emision = _resultados_alerta["Resumen"].set_index("Indicador")["Valor"].get("Fecha de emisión", "")

    st.divider()
    st.subheader("📥 Descarga de resultados")
    st.caption(
        "Excel con los resultados que muestra la app para la fecha de emisión elegida: resumen (nivel y probabilidad de "
        "gran flujo, inicio y ventana de control 600–800 °Cd, emergencia acumulada esperada), alerta semanal, emergencia "
        "acumulada con su rango, tiempo térmico diario desde el inicio, conteos usados, clima, parámetros y validación."
    )
    st.download_button(
        label="📊 Descargar resultados en Excel",
        data=reporte_excel_final.getvalue(),
        file_name=f"PREDWEEM_Lolium_Lartigau_alerta_{str(_emision).replace('/', '-') or 'resultados'}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        key="descarga_excel_resultados_final_lartigau",
    )

mostrar_sesion()
