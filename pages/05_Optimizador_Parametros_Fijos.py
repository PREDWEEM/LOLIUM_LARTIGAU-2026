# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import io
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from predweem_optimizer_fijos import (
    DEFAULT_OPTIMIZED_PARAMETERS,
    FIXED_PARAMETERS,
    PARAMETER_SPACE,
    load_ann_model,
    optimize_parameters_temporal_cv,
    params_to_json,
    prepare_field,
    prepare_weather,
    surface_parameters,
)

APP_VERSION = "LARTIGAU_HIDRICO_FIJOS_v1"

st.set_page_config(
    page_title="Optimizador Lartigau — parámetros fijos",
    page_icon="🧬",
    layout="wide",
)


def read_table(source):
    if source is None:
        return None
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    return pd.read_csv(source)


def default_file(candidates):
    for name in candidates:
        path = BASE / name
        if path.exists():
            return path
    return None


st.title("🧬 Optimizador hídrico — Lartigau")
st.caption(f"PREDWEEM 2026 · Parámetros temporales fijos · {APP_VERSION}")
st.success(
    "El primer pico no se fija por fecha y no se desplaza con lag. "
    "Debe resultar de la ANN y de los filtros validados en vK4.9.15."
)

fixed_df = pd.DataFrame({
    "Parámetro fijo": list(FIXED_PARAMETERS),
    "Valor": list(FIXED_PARAMETERS.values()),
    "Estado": "EXCLUIDO DE LA OPTIMIZACIÓN",
})
with st.expander("🔒 Parámetros fijos y auditables", expanded=True):
    st.dataframe(fixed_df, width="stretch", hide_index=True)
    st.caption(
        "Latencia, termoinhibición, choque hídrico, umbral del primer pico "
        "y lag no forman parte del espacio de búsqueda."
    )

try:
    ann_model = load_ann_model(BASE)
except Exception as exc:
    st.error(f"No se pudo cargar la red neuronal: {exc}")
    st.stop()

weather_default = default_file(["meteo_daily.csv", "meteo_daily.xlsx"])
field_default = default_file([
    "LARTIGAU_campo.xlsx",
    "LARTIGAU_campo.csv",
    "VALIDA.xlsx",
    "VALIDACION.xlsx",
])

with st.expander("1. Datos", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        weather_upload = st.file_uploader(
            "Meteorología diaria", type=["csv", "xlsx", "xls"]
        )
        st.caption(
            f"Automático: {weather_default.name if weather_default else 'no disponible'}"
        )
    with c2:
        field_upload = st.file_uploader(
            "Observaciones de campo", type=["csv", "xlsx", "xls"]
        )
        field_mode = st.selectbox(
            "Formato observado",
            ["interval", "cumulative"],
            format_func=lambda x: (
                "Conteo por intervalo" if x == "interval" else "Conteo acumulado"
            ),
        )
        st.caption(
            f"Automático: {field_default.name if field_default else 'no disponible'}"
        )

weather_source = weather_upload or weather_default
field_source = field_upload or field_default

with st.expander("2. Cobertura y espacio de búsqueda", expanded=True):
    coverage = st.slider(
        "Cobertura de rastrojo manual (%)",
        min_value=0,
        max_value=100,
        value=75,
        step=5,
        help="No se optimiza. Ke y modulador térmico se derivan con las curvas originales.",
    )
    ke, thermal = surface_parameters(coverage)
    m1, m2, m3 = st.columns(3)
    m1.metric("Cobertura manual", f"{coverage}%")
    m2.metric("Ke derivado", f"{ke:.3f}")
    m3.metric("Modulador térmico", f"{thermal:.3f}")

    optimized = st.multiselect(
        "Parámetros libres a optimizar",
        list(PARAMETER_SPACE),
        default=DEFAULT_OPTIMIZED_PARAMETERS,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    ranges = pd.DataFrame([
        {
            "Parámetro libre": name,
            "Mínimo": spec.low,
            "Máximo": spec.high,
            "Inicial": spec.default,
        }
        for name, spec in PARAMETER_SPACE.items()
    ])
    st.dataframe(ranges, width="stretch", hide_index=True)

    a, b, c, d = st.columns(4)
    n_global = a.number_input("Iteraciones globales", 50, 5000, 400, 50)
    n_local = b.number_input("Refinamiento local", 0, 3000, 200, 50)
    seed = c.number_input("Semilla", 0, 999999, 42, 1)
    robustness = d.slider("Penalización por inestabilidad", 0.0, 0.5, 0.15, 0.01)
    e, f = st.columns(2)
    folds = e.slider("Bloques temporales", 2, 5, 3, 1)
    latitude = f.number_input("Latitud Hargreaves", value=-38.6166, format="%.4f")

run = st.button("🚀 Optimizar parámetros libres", type="primary", width="stretch")

if run:
    if weather_source is None or field_source is None:
        st.error("Se requieren meteorología y observaciones de campo.")
        st.stop()
    if not optimized:
        st.error("Seleccione al menos un parámetro libre.")
        st.stop()
    try:
        weather = prepare_weather(read_table(weather_source))
        field = prepare_field(read_table(field_source), value_mode=field_mode)
        progress = st.progress(10, text="Construyendo CV temporal...")
        result = optimize_parameters_temporal_cv(
            weather,
            field,
            ann_model,
            optimized_parameters=optimized,
            cobertura_pct=coverage,
            n_global=int(n_global),
            n_local=int(n_local),
            seed=int(seed),
            latitude=float(latitude),
            robustness_penalty=float(robustness),
            n_folds=int(folds),
            min_intervals_per_fold=2,
        )
        progress.progress(100, text="Optimización finalizada.")
        st.session_state["optimizer_fixed_result"] = result
        st.session_state["optimizer_fixed_selected"] = optimized
    except Exception as exc:
        st.exception(exc)
        st.stop()

if "optimizer_fixed_result" in st.session_state:
    result = st.session_state["optimizer_fixed_result"]
    optimized = st.session_state["optimizer_fixed_selected"]
    summary = result["best_summary"]

    st.success("Óptimo seleccionado sin modificar los parámetros temporales fijos.")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Score CV", f"{summary['Score_CV']:.3f}")
    c2.metric("KGE", f"{summary['KGE_Flujos_Media']:.3f}")
    c3.metric("NSE", f"{summary['NSE_Flujos_Media']:.3f}")
    c4.metric("CCC", f"{summary['CCC_Acumulado_Media']:.3f}")
    c5.metric("RMSE", f"{summary['RMSE_Acumulado_Media']:.3f}")
    c6.metric("Desfase inicio", f"{summary['Desfase_Inicio_Dias']:+d} días")

    tabs = st.tabs([
        "Parámetros",
        "CV temporal",
        "Curvas",
        "Candidatos",
        "Exportar",
    ])

    with tabs[0]:
        best = result["best_params"]
        free_df = pd.DataFrame({
            "Parámetro libre": list(best),
            "Valor óptimo": list(best.values()),
            "Optimizado": [name in optimized for name in best],
        })
        st.dataframe(free_df, width="stretch", hide_index=True)
        st.markdown("#### Parámetros fijos conservados")
        st.dataframe(fixed_df, width="stretch", hide_index=True)
        fecha_pico = summary["Fecha_Primer_Pico_Simulado"]
        fecha_pico_txt = (
            pd.Timestamp(fecha_pico).strftime("%d-%m-%Y")
            if pd.notna(fecha_pico)
            else "N/D"
        )
        st.info(f"Fecha del primer pico resultante: {fecha_pico_txt}")

    with tabs[1]:
        st.dataframe(result["cv_by_fold"], width="stretch", hide_index=True)
        st.markdown("#### Intervalos asignados a cada bloque")
        st.dataframe(result["fold_intervals"], width="stretch", hide_index=True)

    with tabs[2]:
        sim = result["simulation"]
        sync = result["full_sync"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sim["Fecha"], y=sim["EMERREL"],
            mode="lines", name="Simulación diaria",
        ))
        fig.add_trace(go.Scatter(
            x=sync["Fecha"], y=sync["Campo_Relativo"],
            mode="markers+lines", name="Campo por intervalo",
        ))
        fig.update_layout(
            title="Emergencia simulada y observaciones",
            xaxis_title="Fecha",
            yaxis_title="Emergencia relativa",
            hovermode="x unified",
            height=480,
        )
        st.plotly_chart(fig, width="stretch")

        fig_ac = go.Figure()
        fig_ac.add_trace(go.Scatter(
            x=sync["Fecha"], y=sync["Campo_Acumulado"],
            mode="markers+lines", name="Campo acumulado",
        ))
        fig_ac.add_trace(go.Scatter(
            x=sync["Fecha"], y=sync["Sim_Acumulado"],
            mode="lines", line=dict(dash="dash"), name="Modelo acumulado",
        ))
        fig_ac.update_layout(
            title="Curvas acumuladas",
            xaxis_title="Fecha",
            yaxis_title="Proporción acumulada",
            height=430,
        )
        st.plotly_chart(fig_ac, width="stretch")

    with tabs[3]:
        columns = [
            "Score_CV", "Score_CV_Medio", "Score_CV_SD",
            "Score_CV_Peor_Bloque", "Desfase_Inicio_Dias",
            "Fecha_Primer_Pico_Simulado", "Etapa",
        ] + list(PARAMETER_SPACE)
        st.dataframe(
            result["results"][columns].head(100),
            width="stretch",
            hide_index=True,
        )

    with tabs[4]:
        ke, thermal = surface_parameters(result["cobertura_pct"])
        export_params = {
            **result["best_params"],
            "cobertura_manual_pct": result["cobertura_pct"],
            "ke_derivado": ke,
            "modulador_termico_derivado": thermal,
        }
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            pd.DataFrame([export_params]).to_excel(
                writer, sheet_name="Parametros_Libres", index=False
            )
            pd.DataFrame([result["fixed_parameters"]]).to_excel(
                writer, sheet_name="Parametros_Fijos", index=False
            )
            pd.DataFrame([result["best_summary"]]).to_excel(
                writer, sheet_name="Resumen_CV", index=False
            )
            result["cv_by_fold"].to_excel(
                writer, sheet_name="Metricas_Bloques", index=False
            )
            result["results"].to_excel(
                writer, sheet_name="Candidatos", index=False
            )
            result["full_sync"].to_excel(
                writer, sheet_name="Ajuste_Completo", index=False
            )
            result["simulation"].to_excel(
                writer, sheet_name="Simulacion_Diaria", index=False
            )
        st.download_button(
            "📥 Descargar Excel",
            output.getvalue(),
            "PREDWEEM_Lartigau_optimizacion_parametros_libres.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        st.download_button(
            "📄 Descargar JSON",
            params_to_json(result),
            "PREDWEEM_Lartigau_parametros_libres.json",
            mime="application/json",
            width="stretch",
        )
