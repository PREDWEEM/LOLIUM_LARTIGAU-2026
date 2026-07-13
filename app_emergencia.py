# -*- coding: utf-8 -*-
'PREDWEEM Lartigau vK4.9.21 — sincronía ajustable del primer pico.'
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SOURCE_WRAPPER = BASE / "app_emergencia_vK4_9_20.py"

if not SOURCE_WRAPPER.exists():
    raise FileNotFoundError(
        f"Falta la versión optimizada previa: {SOURCE_WRAPPER.name}"
    )

wrapper = SOURCE_WRAPPER.read_text(encoding="utf-8")
wrapper = wrapper.replace(
    "'PREDWEEM Lartigau vK4.9.20: parámetros óptimos de CV temporal interna.'",
    "'PREDWEEM Lartigau vK4.9.21: parámetros óptimos y sincronía manual del primer pico.'",
    1,
)
wrapper = wrapper.replace(
    "# 🌾 PREDWEEM INTEGRAL vK4.9.20 — LOLIUM LARTIGAU 2026",
    "# 🌾 PREDWEEM INTEGRAL vK4.9.21 — LOLIUM LARTIGAU 2026",
)
wrapper = wrapper.replace(
    "# - LAG TEMPORAL: desplazamiento optimizado de +6 días.",
    "# - SINCRONÍA DEL PICO: lag inicial +6 días, ajustable entre 0 y 60 días.",
)

panel_assignment = '    panel = """# --- PARÁMETROS OPTIMIZADOS Y SINCRONÍA TEMPORAL ---\nst.sidebar.divider()\nst.sidebar.markdown("## ⏱️ 4. Sincronía del Primer Pico")\nlag_emergencia_dias = st.sidebar.slider(\n    "Desfase temporal de emergencia (días)",\n    min_value=0,\n    max_value=60,\n    value=LAG_EMERGENCIA_DIAS,\n    step=1,\n    help=(\n        "Aumente este valor si el pico simulado ocurre antes del intervalo "\n        "observado de mayor emergencia. El óptimo inicial fue +6 días."\n    ),\n)\n\nwith st.sidebar.expander("🧬 Parámetros óptimos Lartigau", expanded=False):\n    st.caption(\n        "Seleccionados con CV temporal interna. "\n        "La cobertura y el desfase temporal permanecen ajustables."\n    )\n    st.dataframe(\n        pd.DataFrame({\n            "Parámetro": list(PARAMETROS_OPTIMOS_LARTIGAU_20260713),\n            "Valor óptimo": list(PARAMETROS_OPTIMOS_LARTIGAU_20260713.values()),\n        }),\n        width="stretch",\n        hide_index=True,\n    )\n    st.write({\n        "Cobertura actual (%)": int(cobertura_pct),\n        "Ke derivado": float(ke_val),\n        "Modulador térmico derivado": float(mod_termico),\n        "Lag aplicado (días)": int(lag_emergencia_dias),\n    })\n\n# ---------------------------------------------------------\n# 6. MOTOR DE CÁLCULO\n"""\n'
wrapper, count = re.subn(
    r"    panel = '''# --- PARÁMETROS OPTIMIZADOS ---.*?^'''",
    panel_assignment.rstrip(),
    wrapper,
    count=1,
    flags=re.S | re.M,
)
if count != 1:
    raise RuntimeError(
        f"No se pudo reemplazar el panel temporal: coincidencias={count}"
    )

old_lag_call = 'df, lag_dias=LAG_EMERGENCIA_DIAS, col="EMERREL"'
new_lag_call = 'df, lag_dias=lag_emergencia_dias, col="EMERREL"'
if wrapper.count(old_lag_call) != 1:
    raise RuntimeError(
        f"No se pudo reemplazar el lag rígido: coincidencias={wrapper.count(old_lag_call)}"
    )
wrapper = wrapper.replace(old_lag_call, new_lag_call, 1)

diagnostic_injection = '\n    sync_vars_peak = """    lag_inicio_dias = None\n    fecha_primer_flujo_obs = None\n    fecha_pico_obs_inicio = None\n    fecha_pico_obs_fin = None\n    fecha_pico_sim = None\n    desfase_pico_dias = None\n    lag_recomendado = int(lag_emergencia_dias)\n"""\n    text = replace_once(\n        text,\n        r"    lag_inicio_dias = None\\n    fecha_primer_flujo_obs = None\\n",\n        sync_vars_peak,\n        "variables de sincronía del pico",\n    )\n\n    peak_calc = """        if not muestreos_con_plantas.empty:\n            fecha_primer_flujo_obs = muestreos_con_plantas.iloc[0][col_fecha]\n            if fecha_inicio_ventana is not None:\n                lag_inicio_dias = (fecha_inicio_ventana - fecha_primer_flujo_obs).days\n\n        # El conteo observado representa un intervalo, no un día puntual.\n        campo_pico = (\n            df_campo.dropna(subset=[col_fecha, col_plm2])\n            .sort_values(col_fecha)\n            .reset_index(drop=True)\n            .copy()\n        )\n        if not campo_pico.empty and campo_pico[col_plm2].max() > 0:\n            posicion_pico = int(\n                np.argmax(campo_pico[col_plm2].to_numpy(float))\n            )\n            fecha_pico_obs_fin = pd.Timestamp(\n                campo_pico.iloc[posicion_pico][col_fecha]\n            )\n            fecha_pico_obs_inicio = (\n                pd.Timestamp(df["Fecha"].min()) - pd.Timedelta(days=1)\n                if posicion_pico == 0\n                else pd.Timestamp(\n                    campo_pico.iloc[posicion_pico - 1][col_fecha]\n                )\n            )\n            sim_hasta_campo = df[\n                df["Fecha"] <= campo_pico[col_fecha].max()\n            ].copy()\n            if not sim_hasta_campo.empty and sim_hasta_campo["EMERREL"].max() > 0:\n                fecha_pico_sim = pd.Timestamp(\n                    sim_hasta_campo.loc[\n                        sim_hasta_campo["EMERREL"].idxmax(), "Fecha"\n                    ]\n                )\n                if fecha_pico_sim <= fecha_pico_obs_inicio:\n                    desfase_pico_dias = int(\n                        (fecha_pico_sim - fecha_pico_obs_inicio).days\n                    )\n                elif fecha_pico_sim > fecha_pico_obs_fin:\n                    desfase_pico_dias = int(\n                        (fecha_pico_sim - fecha_pico_obs_fin).days\n                    )\n                else:\n                    desfase_pico_dias = 0\n                lag_recomendado = int(np.clip(\n                    lag_emergencia_dias - desfase_pico_dias,\n                    0,\n                    60,\n                ))\n\n        df_sincronizado"""\n    text = replace_once(\n        text,\n        r"        if not muestreos_con_plantas\\.empty:.*?\\n\\n        df_sincronizado",\n        peak_calc,\n        "cálculo del desfase del pico",\n        re.S,\n    )\n\n    peak_panel = """            st.markdown(\n                "<p class=\'metric-header\' style=\'margin-top:15px;\'>"\n                "📍 SINCRONÍA DEL PRIMER PICO</p>",\n                unsafe_allow_html=True,\n            )\n            p1, p2, p3, p4 = st.columns(4)\n            p1.metric(\n                "Pico simulado",\n                fecha_pico_sim.strftime("%d-%m-%Y")\n                if fecha_pico_sim is not None else "N/A",\n            )\n            intervalo_pico_obs = (\n                f"{fecha_pico_obs_inicio.strftime(\'%d-%m\')} a "\n                f"{fecha_pico_obs_fin.strftime(\'%d-%m-%Y\')}"\n                if fecha_pico_obs_inicio is not None\n                and fecha_pico_obs_fin is not None\n                else "N/A"\n            )\n            p2.metric("Intervalo pico observado", intervalo_pico_obs)\n            p3.metric(\n                "Desfase fuera del intervalo",\n                f"{desfase_pico_dias:+d} días"\n                if desfase_pico_dias is not None else "N/A",\n                "Negativo = modelo anticipa",\n                delta_color="inverse",\n            )\n            p4.metric(\n                "Lag sugerido",\n                f"{lag_recomendado} días"\n                if desfase_pico_dias is not None else "N/A",\n                f"Actual: {lag_emergencia_dias} días",\n            )\n            if desfase_pico_dias is not None and desfase_pico_dias < 0:\n                st.warning(\n                    f"El pico simulado ocurre {abs(desfase_pico_dias)} días "\n                    f"antes del intervalo observado. "\n                    f"Pruebe un lag de {lag_recomendado} días."\n                )\n            elif desfase_pico_dias is not None and desfase_pico_dias > 0:\n                st.info(\n                    f"El pico simulado ocurre {desfase_pico_dias} días "\n                    f"después del intervalo observado. "\n                    f"Pruebe un lag de {lag_recomendado} días."\n                )\n            elif desfase_pico_dias is not None:\n                st.success(\n                    "El máximo simulado cae dentro del intervalo observado "\n                    "de mayor emergencia."\n                )\n\n            # --- TABLA HTML: MATRIZ DE CONFUSIÓN ---"""\n    text = replace_once(\n        text,\n        r"            # --- TABLA HTML: MATRIZ DE CONFUSIÓN ---",\n        peak_panel,\n        "panel de sincronía del pico",\n    )\n'
marker = "    return text\n\n\nif not SOURCE.exists():"
if wrapper.count(marker) != 1:
    raise RuntimeError(
        f"No se pudo insertar el diagnóstico: coincidencias={wrapper.count(marker)}"
    )
wrapper = wrapper.replace(
    marker,
    diagnostic_injection + "\n    return text\n\n\nif not SOURCE.exists():",
    1,
)

exec(
    compile(wrapper, str(SOURCE_WRAPPER), "exec"),
    {
        "__name__": "__main__",
        "__file__": str(SOURCE_WRAPPER),
        "__package__": None,
    },
)
