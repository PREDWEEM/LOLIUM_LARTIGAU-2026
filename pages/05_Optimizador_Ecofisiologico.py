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

from predweem_optimizer import (
    DEFAULT_OPTIMIZED_PARAMETERS,
    PARAMETER_SPACE,
    load_ann_model,
    optimize_parameters_temporal_cv,
    params_to_json,
    prepare_field,
    prepare_weather,
    surface_parameters,
)

APP_VERSION = "LARTIGAU_CV_TEMPORAL_v1_2026-07-13"

st.set_page_config(
    page_title="Optimizador ecofisiológico Lartigau",
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


def parameter_importance(results: pd.DataFrame, optimized: list[str]) -> pd.DataFrame:
    rows = []
    for name in optimized:
        if name in results and results[name].nunique() > 1:
            rho = results[[name, "Score_CV"]].corr(method="spearman").iloc[0, 1]
            if pd.notna(rho):
                rows.append({
                    "Parametro": name,
                    "Rho_Spearman": float(rho),
                    "Importancia_Abs": abs(float(rho)),
                })
    if not rows:
        return pd.DataFrame(columns=["Parametro", "Rho_Spearman", "Importancia_Abs"])
    return pd.DataFrame(rows).sort_values("Importancia_Abs", ascending=False).reset_index(drop=True)


def metric_cards(title, summary, score_key):
    st.markdown(f"### {title}")
    cols = st.columns(6)
    values = [
        ("Score", summary.get(score_key, 0.0), ".3f"),
        ("KGE", summary.get("KGE_Flujos_Media", 0.0), ".3f"),
        ("NSE", summary.get("NSE_Flujos_Media", 0.0), ".3f"),
        ("CCC", summary.get("CCC_Acumulado_Media", 0.0), ".3f"),
        ("RMSE", summary.get("RMSE_Acumulado_Media", 0.0), ".3f"),
        ("T50", summary.get("Desfase_T50_Media", 0.0), "+.0f"),
    ]
    for col, (label, value, fmt) in zip(cols, values):
        try:
            col.metric(label, format(float(value), fmt), "días" if label == "T50" else None)
        except Exception:
            col.metric(label, "N/D")


st.title("🧬 Optimizador ecofisiológico — Lartigau")
st.caption(f"PREDWEEM Lartigau 2026 · CV temporal interna · {APP_VERSION}")
st.warning(
    "La página utiliza un único conjunto de campo y lo divide en bloques "
    "cronológicos contiguos. El resultado es validación interna para selección "
    "provisional de parámetros; no reemplaza una validación externa en otra campaña."
)
st.info(
    "La cobertura de rastrojo se optimiza como porcentaje. Ke y el modulador "
    "térmico se derivan automáticamente con las mismas curvas de app_emergencia.py."
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
    "validacion.xlsx",
])

with st.expander("1. Datos disponibles", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        weather_upload = st.file_uploader("Meteorología diaria", type=["csv", "xlsx", "xls"])
        st.caption(
            "Archivo automático: "
            f"{weather_default.name if weather_default else 'no disponible; cárguelo manualmente'}"
        )
    with c2:
        field_upload = st.file_uploader("Observaciones de campo", type=["csv", "xlsx", "xls"])
        field_mode = st.selectbox(
            "Formato de la variable observada",
            ["interval", "cumulative"],
            format_func=lambda value: (
                "Flujo/conteo por intervalo" if value == "interval" else "Conteo acumulado"
            ),
        )
        st.caption(
            "Archivo automático: "
            f"{field_default.name if field_default else 'no disponible; cárguelo manualmente'}"
        )

weather_source = weather_upload or weather_default
field_source = field_upload or field_default

with st.expander("2. Espacio de búsqueda", expanded=True):
    optimized = st.multiselect(
        "Variables a optimizar",
        list(PARAMETER_SPACE),
        default=DEFAULT_OPTIMIZED_PARAMETERS,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    st.caption(
        "Los parámetros no seleccionados permanecen en los valores actuales de Lartigau. "
        "Lag = 0 reproduce la ausencia de desplazamiento temporal del modelo operativo."
    )
    a, b, c, d = st.columns(4)
    n_global = a.number_input("Iteraciones globales", 50, 5000, 400, 50)
    n_local = b.number_input("Refinamiento local", 0, 3000, 200, 50)
    seed = c.number_input("Semilla", 0, 999999, 42, 1)
    robustness = d.slider("Penalización por inestabilidad", 0.0, 0.5, 0.15, 0.01)
    e, f = st.columns(2)
    folds = e.slider(
        "Bloques temporales solicitados",
        min_value=2,
        max_value=5,
        value=3,
        step=1,
        help="Cada bloque debe contener al menos dos intervalos.",
    )
    latitude = f.number_input(
        "Latitud para ET0 Hargreaves",
        value=-38.6166,
        format="%.4f",
    )

run = st.button("🚀 Optimizar Lartigau", type="primary", width="stretch")

if run:
    if weather_source is None or field_source is None:
        st.error("Se requieren meteorología diaria y observaciones de campo.")
        st.stop()
    if not optimized:
        st.error("Seleccione al menos una variable para optimizar.")
        st.stop()
    try:
        weather_data = prepare_weather(read_table(weather_source))
        field_data = prepare_field(read_table(field_source), value_mode=field_mode)
        progress = st.progress(10, text="Construyendo bloques temporales...")
        result = optimize_parameters_temporal_cv(
            weather_data,
            field_data,
            ann_model,
            optimized_parameters=optimized,
            n_global=int(n_global),
            n_local=int(n_local),
            seed=int(seed),
            latitude=float(latitude),
            robustness_penalty=float(robustness),
            n_folds=int(folds),
            min_intervals_per_fold=2,
        )
        progress.progress(100, text="Optimización y CV temporal finalizadas.")
    except Exception as exc:
        st.exception(exc)
        st.stop()
    st.session_state["lartigau_cv_result"] = result
    st.session_state["lartigau_cv_optimized"] = optimized

if "lartigau_cv_result" in st.session_state:
    result = st.session_state["lartigau_cv_result"]
    optimized = st.session_state["lartigau_cv_optimized"]
    st.success("Parámetros seleccionados por desempeño medio y estabilidad temporal.")
    metric_cards("Validación cruzada temporal interna", result["best_summary"], "Score_CV")

    apparent = result["apparent_summary"]
    apparent_cards = {
        "Score_Aparente": apparent.get("Score_Calibracion", 0.0),
        "KGE_Flujos_Media": apparent.get("KGE_Flujos_Media", 0.0),
        "NSE_Flujos_Media": apparent.get("NSE_Flujos_Media", 0.0),
        "CCC_Acumulado_Media": apparent.get("CCC_Acumulado_Media", 0.0),
        "RMSE_Acumulado_Media": apparent.get("RMSE_Acumulado_Media", 0.0),
        "Desfase_T50_Media": apparent.get("Desfase_T50_Media", 0.0),
    }
    metric_cards("Evaluación descriptiva sobre toda la serie", apparent_cards, "Score_Aparente")

    tabs = st.tabs([
        "Parámetros óptimos",
        "Bloques temporales",
        "Candidatos",
        "Gráficos",
        "Sensibilidad",
        "Descargas",
    ])

    with tabs[0]:
        best = result["best_params"]
        ke, thermal = surface_parameters(best["cobertura_pct"])
        params_df = pd.DataFrame({
            "Parametro": list(best),
            "Valor_optimo": list(best.values()),
            "Optimizado": [name in optimized for name in best],
        })
        st.dataframe(params_df, width="stretch", hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cobertura óptima", f"{int(best['cobertura_pct'])}%")
        c2.metric("Ke derivado", f"{ke:.3f}")
        c3.metric("Modulador térmico derivado", f"{thermal:.3f}")
        if int(best.get("lag_dias", 0)) != 0:
            st.warning(
                "El lag óptimo es distinto de cero. app_emergencia.py deberá incorporar "
                "un desplazamiento de la señal para reproducir exactamente este resultado."
            )

    with tabs[1]:
        st.dataframe(result["cv_by_fold"], width="stretch", hide_index=True)
        st.markdown("#### Definición de los intervalos")
        st.dataframe(result["fold_intervals"], width="stretch", hide_index=True)

    with tabs[2]:
        columns = [
            "Score_CV",
            "Score_CV_Medio",
            "Score_CV_SD",
            "Score_CV_Peor_Bloque",
            "Etapa",
        ] + list(optimized)
        st.dataframe(result["results"][columns].head(50), width="stretch", hide_index=True)

    with tabs[3]:
        sync = result["apparent_sync"]
        if sync.empty:
            st.warning("No se pudieron construir intervalos.")
        else:
            fig = go.Figure()
            for group, part in sync.groupby("Grupo"):
                fig.add_trace(go.Scatter(
                    x=part["Fecha"], y=part["Campo_Acumulado"],
                    mode="markers+lines", name=f"Campo {group}",
                ))
                fig.add_trace(go.Scatter(
                    x=part["Fecha"], y=part["Sim_Acumulado"],
                    mode="lines", line=dict(dash="dash"), name=f"Modelo {group}",
                ))
            fig.update_layout(
                title="Evaluación completa del mismo set (no es validación independiente)",
                yaxis_title="Proporción acumulada",
                xaxis_title="Fecha",
                hovermode="x unified",
                height=480,
            )
            st.plotly_chart(fig, width="stretch")

            cv_sync = result["cv_sync"]
            fig11 = go.Figure()
            fig11.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="1:1", line=dict(dash="dash")
            ))
            fig11.add_trace(go.Scatter(
                x=cv_sync["Campo_Relativo"], y=cv_sync["Sim_Relativo"],
                mode="markers", text=cv_sync["Fold"], name="Intervalos retenidos",
            ))
            fig11.update_layout(
                title="CV temporal: observado vs. simulado por intervalo",
                xaxis_title="Observado relativo",
                yaxis_title="Simulado relativo",
                height=430,
            )
            st.plotly_chart(fig11, width="stretch")

    with tabs[4]:
        importance = parameter_importance(result["results"], optimized)
        st.caption(
            "Correlación de Spearman con el score de CV. Es una sensibilidad "
            "aproximada y no demuestra causalidad."
        )
        st.dataframe(importance, width="stretch", hide_index=True)
        if not importance.empty:
            fig_imp = go.Figure(go.Bar(
                x=importance["Importancia_Abs"], y=importance["Parametro"], orientation="h"
            ))
            fig_imp.update_layout(
                title="Sensibilidad global aproximada",
                xaxis_title="|rho de Spearman|",
                yaxis_title="",
            )
            st.plotly_chart(fig_imp, width="stretch")

    with tabs[5]:
        best_export = dict(result["best_params"])
        ke, thermal = surface_parameters(best_export["cobertura_pct"])
        best_export["ke_suelo_derivado"] = ke
        best_export["mod_termico_derivado"] = thermal
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            pd.DataFrame([best_export]).to_excel(
                writer, sheet_name="Parametros_Optimos", index=False
            )
            pd.DataFrame([result["best_summary"]]).to_excel(
                writer, sheet_name="Resumen_CV_Temporal", index=False
            )
            pd.DataFrame([result["apparent_summary"]]).to_excel(
                writer, sheet_name="Ajuste_Aparente", index=False
            )
            result["cv_by_fold"].to_excel(
                writer, sheet_name="Metricas_Bloques", index=False
            )
            result["fold_intervals"].to_excel(
                writer, sheet_name="Definicion_Bloques", index=False
            )
            result["results"].to_excel(
                writer, sheet_name="Candidatos", index=False
            )
            result["cv_sync"].to_excel(
                writer, sheet_name="Intervalos_CV", index=False
            )
            result["apparent_sync"].to_excel(
                writer, sheet_name="Intervalos_Completos", index=False
            )
        st.download_button(
            "📥 Descargar informe Excel",
            output.getvalue(),
            "PREDWEEM_Lartigau_optimizacion_CV_temporal.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        st.download_button(
            "📄 Descargar parámetros JSON",
            params_to_json(result["best_params"]),
            "PREDWEEM_Lartigau_parametros_optimos.json",
            mime="application/json",
            width="stretch",
        )
