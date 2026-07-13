# -*- coding: utf-8 -*-
'PREDWEEM Lartigau vK4.9.20: parámetros óptimos de CV temporal interna.'
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "app_emergencia_vK4_9_15.py"

PARAMETROS_OPTIMOS_LARTIGAU_20260713 = {
    "w_max": 18.816194,
    "cobertura_pct": 57,
    "humedad_p50": 0.416332,
    "pendiente_hidrica": 10.0,
    "humedad_corte": 0.330487,
    "recarga_relativa": 0.590681,
    "latencia_jd": 20,
    "ventana_termica": 19,
    "umbral_termoinhibicion": 26.398075,
    "ventana_lluvia": 3,
    "umbral_choque_hidrico": 53.041243,
    "fin_choque_jd": 110,
    "techo_choque": 1.0,
    "umbral_primer_pico": 0.680112,
    "persistencia_primer_pico": 1,
    "lag_dias": 22,
}


def replace_once(text, pattern, replacement, label, flags=0):
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(
            f"No se pudo aplicar el ajuste '{label}': coincidencias={count}"
        )
    return updated


def patch(text):
    text = text.replace(
        "# 🌾 PREDWEEM INTEGRAL vK4.9.15 — LOLIUM LARTIGAU 2026",
        "# 🌾 PREDWEEM INTEGRAL vK4.9.20 — LOLIUM LARTIGAU 2026",
        1,
    )
    text = text.replace(
        "# - LATENCIA INICIAL: Bloqueo estricto de emergencia los primeros 45 días del año.",
        "# - LATENCIA INICIAL: bloqueo optimizado hasta JD 20.",
        1,
    )
    text = text.replace(
        "# - CHOQUE HÍDRICO CONSERVADO: Umbral de lluvia acumulada de 3 días = 45 mm.",
        "# - CHOQUE HÍDRICO: umbral optimizado de 53.041243 mm acumulados en 3 días.",
        1,
    )
    text = text.replace(
        "# - ESCUDO TERMOFISIOLÓGICO INTACTO: Media móvil de 5 días y umbral térmico sin cambios.",
        "# - ESCUDO TERMOFISIOLÓGICO: ventana 19 días y umbral inicial 26.398075 °C.",
        1,
    )
    text = text.replace(
        "# - PRIMER PICO VÁLIDO: La campaña se habilita únicamente cuando EMERREL > 0.70.",
        "# - PRIMER PICO VÁLIDO: campaña habilitada cuando EMERREL > 0.680112.",
        1,
    )
    text = text.replace(
        "# - UX DINÁMICA: Sombreados de fondo basados en las fechas reales de muestreo.",
        "# - COBERTURA MANUAL: valor inicial óptimo 57%; Ke y modulador térmico derivados.\n"
        "# - LAG TEMPORAL: desplazamiento optimizado de +6 días.\n"
        "# - UX DINÁMICA: Sombreados de fondo basados en las fechas reales de muestreo.",
        1,
    )

    params = '''PARAMETROS_OPTIMOS_LARTIGAU_20260713 = {
    "w_max": 18.816194,
    "cobertura_pct": 57,
    "humedad_p50": 0.416332,
    "pendiente_hidrica": 10.0,
    "humedad_corte": 0.330487,
    "recarga_relativa": 0.590681,
    "latencia_jd": 20,
    "ventana_termica": 19,
    "umbral_termoinhibicion": 26.398075,
    "ventana_lluvia": 3,
    "umbral_choque_hidrico": 53.041243,
    "fin_choque_jd": 110,
    "techo_choque": 1.0,
    "umbral_primer_pico": 0.680112,
    "persistencia_primer_pico": 1,
    "lag_dias": 22,
}
UMBRAL_PRIMER_PICO = PARAMETROS_OPTIMOS_LARTIGAU_20260713["umbral_primer_pico"]
LAG_EMERGENCIA_DIAS = int(PARAMETROS_OPTIMOS_LARTIGAU_20260713["lag_dias"])
'''
    text = replace_once(
        text,
        r"UMBRAL_PRIMER_PICO\s*=\s*0\.70\s*\n",
        params,
        "bloque de parámetros",
    )

    balance = '''def balance_hidrico_superficial(prec, et0, w_max=20.0, ke_suelo=0.4):
    prec = np.asarray(prec, dtype=float)
    et0 = np.asarray(et0, dtype=float)
    n = len(prec)
    w = np.zeros(n, dtype=float)
    if n == 0:
        return w
    w[0] = np.clip(w_max / 2.0 + prec[0] - et0[0] * ke_suelo, 0.0, w_max)
    for i in range(1, n):
        w[i] = np.clip(w[i - 1] + prec[i] - et0[i] * ke_suelo, 0.0, w_max)
    return w
'''
    text = replace_once(
        text,
        r"def balance_hidrico_superficial\(prec, et0, w_max=20\.0, ke_suelo=0\.4\):.*?\n    return w\n",
        balance,
        "balance hídrico",
        re.S,
    )

    lag_function = '''    return df, idx_primer_pico

def aplicar_lag_emergencia(df, lag_dias=LAG_EMERGENCIA_DIAS, col="EMERREL"):
    # Desplaza la señal diaria; lag positivo retrasa la emergencia.
    df = df.copy()
    df[f"{col}_SIN_LAG"] = df[col].copy()
    valores = df[col].to_numpy(float)
    desplazado = np.zeros_like(valores)
    lag_dias = int(lag_dias)
    if lag_dias == 0:
        desplazado = valores.copy()
    elif lag_dias > 0 and lag_dias < len(valores):
        desplazado[lag_dias:] = valores[:-lag_dias]
    elif lag_dias < 0 and abs(lag_dias) < len(valores):
        k = abs(lag_dias)
        desplazado[:-k] = valores[k:]
    df[col] = desplazado
    df["Lag_Emergencia_Dias"] = lag_dias
    return df

class PracticalANNModel:
'''
    text = replace_once(
        text,
        r"    return df, idx_primer_pico\n\nclass PracticalANNModel:\n",
        lag_function,
        "función de lag",
    )

    sync_function = '''def sincronizar_intervalos_variables(df_sim, df_campo, col_fecha, col_plm2):
    # Sincroniza todos los conteos, incluido el primer intervalo observado.
    campo = df_campo.sort_values(col_fecha).copy()
    campo[col_fecha] = pd.to_datetime(campo[col_fecha], errors="coerce")
    campo[col_plm2] = pd.to_numeric(campo[col_plm2], errors="coerce")
    campo = campo.dropna(subset=[col_fecha, col_plm2]).copy()
    campo[col_plm2] = campo[col_plm2].clip(lower=0.0)
    if campo.empty:
        return pd.DataFrame()

    campo["Campo_Acum_Abs"] = campo[col_plm2].cumsum()
    inicio_sim = pd.Timestamp(df_sim["Fecha"].min()) - pd.Timedelta(days=1)
    registros = []

    for i, row in campo.reset_index(drop=True).iterrows():
        f_inicio = inicio_sim if i == 0 else campo.iloc[i - 1][col_fecha]
        f_fin = row[col_fecha]
        flujo_obs = float(row[col_plm2])
        mask_sim = (df_sim["Fecha"] > f_inicio) & (df_sim["Fecha"] <= f_fin)
        flujo_sim = float(df_sim.loc[mask_sim, "EMERREL"].sum())
        acum_sim_fin = float(
            df_sim.loc[df_sim["Fecha"] <= f_fin, "EMERREL"].sum()
        )
        registros.append({
            "Fecha": f_fin,
            "Dias_Intervalo": int((f_fin - f_inicio).days),
            "Flujo_Obs_Abs": flujo_obs,
            "Flujo_Sim_Abs": flujo_sim,
            "Acum_Obs_Abs": float(row["Campo_Acum_Abs"]),
            "Acum_Sim_Abs": acum_sim_fin,
        })

    df_res = pd.DataFrame(registros)
    total_obs = float(df_res["Flujo_Obs_Abs"].sum())
    ultima_fecha = campo[col_fecha].max()
    total_sim = float(
        df_sim.loc[df_sim["Fecha"] <= ultima_fecha, "EMERREL"].sum()
    )
    df_res["Campo_Relativo"] = (
        df_res["Flujo_Obs_Abs"] / total_obs if total_obs > 0 else 0.0
    )
    df_res["Sim_Relativo"] = (
        df_res["Flujo_Sim_Abs"] / total_sim if total_sim > 0 else 0.0
    )
    df_res["Campo_Acumulado"] = (
        df_res["Acum_Obs_Abs"] / total_obs if total_obs > 0 else 0.0
    )
    df_res["Sim_Acumulado"] = (
        df_res["Acum_Sim_Abs"] / total_sim if total_sim > 0 else 0.0
    )
    return df_res

def calcular_metricas_validacion_integral'''
    text = replace_once(
        text,
        r"def sincronizar_intervalos_variables\(df_sim, df_campo, col_fecha, col_plm2\):.*?\ndef calcular_metricas_validacion_integral",
        sync_function,
        "sincronización de intervalos",
        re.S,
    )

    text = replace_once(
        text,
        r'min_value=0, max_value=100, value=75, step=5,\n\s*help="[^"]*"',
        'min_value=0, max_value=100, value=57, step=1,\n'
        '                help="Control manual; óptimo CV temporal = 57%. Ke y modulador térmico se derivan con las curvas originales."',
        "cobertura óptima manual",
    )
    text = replace_once(
        text,
        r'umbral_termoinhibicion = st\.sidebar\.number_input\([^\n]+\)',
        'umbral_termoinhibicion = st.sidebar.number_input('
        '"Umbral Termoinhibición (°C)", 15.0, 35.0, 26.398075, 0.1, format="%.3f")',
        "umbral de termoinhibición",
    )
    text = replace_once(
        text,
        r'umbral_choque_hidrico = st\.sidebar\.slider\(.*?\n\)',
        'umbral_choque_hidrico = st.sidebar.slider(\n'
        '    "Choque Hídrico 3 días (mm)",\n'
        '    min_value=20.0,\n'
        '    max_value=100.0,\n'
        '    value=53.041243,\n'
        '    step=0.1,\n'
        '    format="%.1f mm",\n'
        ')',
        "umbral de choque hídrico",
        re.S,
    )
    text = replace_once(
        text,
        r'w_max_val = st\.sidebar\.number_input\("Cap\. de Campo Superficial \(mm\)", value=20\.0, step=1\.0\)',
        'w_max_val = st.sidebar.number_input('
        '"Cap. de Campo Superficial (mm)", value=18.816194, step=0.1, format="%.3f")',
        "Wmax",
    )

    panel = '''# --- PARÁMETROS OPTIMIZADOS ---
with st.sidebar.expander("🧬 Parámetros óptimos Lartigau", expanded=False):
    st.caption(
        "Seleccionados con CV temporal interna sobre el único set disponible. "
        "La cobertura permanece manual y parte de 57%."
    )
    st.dataframe(
        pd.DataFrame({
            "Parámetro": list(PARAMETROS_OPTIMOS_LARTIGAU_20260713),
            "Valor óptimo": list(PARAMETROS_OPTIMOS_LARTIGAU_20260713.values()),
        }),
        width="stretch",
        hide_index=True,
    )
    st.write({
        "Cobertura actual (%)": int(cobertura_pct),
        "Ke derivado": float(ke_val),
        "Modulador térmico derivado": float(mod_termico),
    })
    st.info(f"Lag temporal aplicado: +{LAG_EMERGENCIA_DIAS} días.")

# ---------------------------------------------------------
# 6. MOTOR DE CÁLCULO
'''
    text = replace_once(
        text,
        r"# --- MODO DESARROLLADOR: OPTIMIZADOR 2D ---.*?# ---------------------------------------------------------\n# 6\. MOTOR DE CÁLCULO\n",
        panel,
        "panel de parámetros",
        re.S,
    )

    motor = '''    # 2. Latencia optimizada.
    latencia_jd = int(PARAMETROS_OPTIMOS_LARTIGAU_20260713["latencia_jd"])
    df.loc[df["Julian_days"] <= latencia_jd, "EMERREL"] = 0.0

    # 3. Choque hídrico de ruptura temprana.
    ventana_lluvia = int(PARAMETROS_OPTIMOS_LARTIGAU_20260713["ventana_lluvia"])
    col_prec_acum = f"Prec_{ventana_lluvia}d"
    df[col_prec_acum] = df["Prec"].rolling(
        window=ventana_lluvia, min_periods=1
    ).sum()
    mask_ruptura = (
        (df["Julian_days"] > latencia_jd)
        & (
            df["Julian_days"]
            <= int(PARAMETROS_OPTIMOS_LARTIGAU_20260713["fin_choque_jd"])
        )
        & (df[col_prec_acum] >= umbral_choque_hidrico)
    )
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(
        df.loc[mask_ruptura, "EMERREL"],
        float(PARAMETROS_OPTIMOS_LARTIGAU_20260713["techo_choque"]),
    )

    # 4. Balance hídrico optimizado. Ke deriva de la cobertura manual.
    df["ET0"] = calcular_et0_hargreaves(
        df["Julian_days"].values,
        df["TMAX"].values,
        df["TMIN"].values,
        latitud=-38.6166,
    )
    df["W_superficial"] = balance_hidrico_superficial(
        df["Prec"].values,
        df["ET0"].values,
        w_max=w_max_val,
        ke_suelo=ke_val,
    )
    humedad_relativa = df["W_superficial"] / max(w_max_val, 1e-12)
    pendiente = float(
        PARAMETROS_OPTIMOS_LARTIGAU_20260713["pendiente_hidrica"]
    )
    p50 = float(PARAMETROS_OPTIMOS_LARTIGAU_20260713["humedad_p50"])
    exponente = np.clip(
        -pendiente * (humedad_relativa - p50), -60.0, 60.0
    )
    df["Hydric_Factor"] = 1.0 / (1.0 + np.exp(exponente))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]
    df.loc[
        humedad_relativa
        < float(PARAMETROS_OPTIMOS_LARTIGAU_20260713["humedad_corte"]),
        "EMERREL",
    ] = 0.0
    df["Recarga_Habilitada"] = pd.Series(
        humedad_relativa
        >= float(PARAMETROS_OPTIMOS_LARTIGAU_20260713["recarga_relativa"]),
        index=df.index,
    ).cummax()
    df.loc[~df["Recarga_Habilitada"], "EMERREL"] = 0.0

    # 5. Escudo termofisiológico optimizado.
    ventana_termica = int(
        PARAMETROS_OPTIMOS_LARTIGAU_20260713["ventana_termica"]
    )
    df["Tmedia"] = df["Tmedia_aire"]
    col_tmedia_movil = f"Tmedia_{ventana_termica}d"
    df[col_tmedia_movil] = df["Tmedia"].rolling(
        window=ventana_termica, min_periods=1
    ).mean()
    df.loc[
        df[col_tmedia_movil] >= umbral_termoinhibicion,
        "EMERREL",
    ] = 0.0
    df["Termoinhibida"] = (
        df[col_tmedia_movil] >= umbral_termoinhibicion
    )
    df["EMERREL"] = np.clip(df["EMERREL"], 0.0, 1.0)

    # 6. Primer pico y lag temporal optimizados.
    df, idx_primer_pico_original = aplicar_filtro_primer_pico(
        df, umbral=UMBRAL_PRIMER_PICO
    )
    df = aplicar_lag_emergencia(
        df, lag_dias=LAG_EMERGENCIA_DIAS, col="EMERREL"
    )
    df, idx_primer_pico = aplicar_filtro_primer_pico(
        df, umbral=UMBRAL_PRIMER_PICO
    )
    # ----------------------------------------------------

    df["DG"]'''
    text = replace_once(
        text,
        r"    # 2\. Choque Hídrico de Ruptura Temprana.*?\n    df\[\"DG\"\]",
        motor,
        "motor ecofisiológico",
        re.S,
    )
    return text


if not SOURCE.exists():
    raise FileNotFoundError(
        f"Falta el modelo base preservado: {SOURCE.name}"
    )

source = SOURCE.read_text(encoding="utf-8")
optimized = patch(source)
exec(
    compile(optimized, str(SOURCE), "exec"),
    {"__name__": "__main__", "__file__": str(SOURCE), "__package__": None},
)
