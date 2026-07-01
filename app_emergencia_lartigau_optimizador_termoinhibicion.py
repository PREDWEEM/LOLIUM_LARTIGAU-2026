# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.9.18 — LOLIUM LARTIGAU 2026
# Actualización y Rigor Científico:
# - ADAPTACIÓN LARTIGAU: Coordenadas fijas en -38.6166 para ET0 y balances.
# - IDENTIDAD: PREDWEEM by GUILLERMO R. CHANTRE.
# - LATENCIA INICIAL: Bloqueo estricto de emergencia los primeros 25 días del año.
# - TERMOINHIBICIÓN CONTINUA: distribución normal acumulada complementaria,
#   con media, desvío, factor mínimo y persistencia ajustables.
# - OPTIMIZADOR EXCLUSIVO DE TERMOINHIBICIÓN: calibra T50, sigma,
#   factor mínimo de habilitación y persistencia, manteniendo fijos los
#   parámetros hídricos, la ANN y la lógica de lluvia.
# - VALIDACIÓN DE FRECUENCIA VARIABLE: Reemplazo de remuestreo sintético por
#   Integración Dinámica de Intervalo Real (Event-to-Event), apto para frecuencias de 7 a 21 días.
# - OPTIMIZADOR 2D BIO-FÍSICO: Barrido de alta eficiencia sobre W_Max y Ke usando fechas de campo puras.
# - COINCIDENCIA OPERATIVA: Métricas F1-Score, Exactitud Global y Matriz de Confusión interactiva.
# - SINCRONÍA DE INICIO: Evaluación del desfase temporal del primer flujo (Gatillo de DGA).
# - UX DINÁMICA: Sombreados de fondo basados en las fechas reales de muestreo.
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import time
from datetime import timedelta
from pathlib import Path
import base64
import math

# ---------------------------------------------------------
# 1. PANTALLA DE CARGA
# ---------------------------------------------------------
if 'arranque_fase' not in st.session_state:
    st.set_page_config(page_title="PREDWEEM LARTIGAU INTEGRAL", layout="wide", page_icon="🌾")
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.info("🚜 **Iniciando Servidor PREDWEEM Lartigau...** Cargando motores dinámicos por evento.")
    st.progress(20)
    
    st.session_state.arranque_fase = 1
    time.sleep(0.1)
    st.rerun()

if 'arranque_fase' in st.session_state and st.session_state.arranque_fase == 1:
    st.session_state.arranque_fase = 2 

# ---------------------------------------------------------
# 2. CONFIGURACIÓN DE ESTILOS GLOBALES
# ---------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] {
        background-color: #dcfce7;
        border-right: 1px solid #bbf7d0;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: #166534 !important;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-header { color: #1e293b; font-weight: bold; margin-bottom: -10px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    div[data-testid="stVerticalBlockBorderWrapper"], 
    div[data-testid="stContainerBorder"],
    div[data-testid="stContainer"] > div > div[style*="border"],
    div[data-testid="stVerticalBlock"] > div[style*="border-radius"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        padding: 15px !important;
        border: 1px solid #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

def set_bg_hack(main_bg_file):
    try:
        with open(main_bg_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""<style>.stApp {{ background-image: url(data:image/png;base64,{encoded_string}); background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed; }}</style>""",
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

set_bg_hack("fondo_predweem_v3.png") 

# ---------------------------------------------------------
# 3. ROBUSTEZ Y ARCHIVOS (MOCKS)
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))

    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        p1 = np.exp(-((jd - 100) ** 2) / 600)
        p2 = np.exp(-((jd - 160) ** 2) / 900) + 0.3 * np.exp(-((jd - 260) ** 2) / 1200)
        p3 = np.exp(-((jd - 230) ** 2) / 1500)
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump({"JD_common": jd, "curves_interp": [p2, p1, p3], "medoids_k3": [0, 1, 2]}, f)

create_mock_files_if_missing()

# ---------------------------------------------------------
# 4. LÓGICA TÉCNICA E INTEGRACIÓN POR INTERVALOS VARIABLES
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na + 1, nb + 1), np.inf)
    dp[0, 0] = 0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base: return 0.0
    elif t <= t_opt: return t - t_base
    elif t < t_crit: return (t - t_base) * ((t_crit - t) / (t_crit - t_opt))
    else: return 0.0

def calcular_et0_hargreaves(jday, tmax, tmin, latitud=-38.6166):
    lat_rad = np.radians(latitud)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    dec = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    ws = np.arccos(-np.tan(lat_rad) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws))
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0)

def balance_hidrico_superficial(prec, et0, w_max=20.0, ke_suelo=0.4):
    n = len(prec)
    w = np.zeros(n)
    w[0] = w_max / 2.0 
    for i in range(1, n):
        evaporacion_real = et0[i] * ke_suelo
        w[i] = max(0.0, min(w_max, w[i-1] + prec[i] - evaporacion_real))
    return w


def factor_termoinhibicion_normal(temperatura, media=24.0, desvio=2.0):
    """Factor continuo de aptitud térmica basado en una distribución normal.

    Se utiliza la función de supervivencia:
        F = 1 - Phi((T - media) / desvio)

    F vale 0,50 cuando T = media. Valores bajos de temperatura producen
    factores próximos a 1 y valores altos producen factores próximos a 0.
    """
    if desvio <= 0:
        raise ValueError("El desvío de termoinhibición debe ser mayor que cero.")

    temperatura = np.asarray(temperatura, dtype=float)
    z = (temperatura - float(media)) / (float(desvio) * np.sqrt(2.0))
    erf_vectorizado = np.vectorize(math.erf, otypes=[float])
    cdf_normal = 0.5 * (1.0 + erf_vectorizado(z))
    return np.clip(1.0 - cdf_normal, 0.0, 1.0)


def habilitacion_termica(
    factor_termico,
    factor_minimo=0.35,
    persistencia_dias=2,
):
    """Habilita el inicio cuando la aptitud térmica es suficiente.

    La distribución normal nunca alcanza exactamente cero. Por eso, sin una
    habilitación explícita, valores térmicos muy pequeños pueden superar un
    umbral de alerta bajo y mantener invariable la fecha de inicio.

    Se exige:
    - factor térmico >= factor_minimo;
    - persistencia durante varios días consecutivos.
    """
    if not 0.0 <= factor_minimo <= 1.0:
        raise ValueError("El factor térmico mínimo debe estar entre 0 y 1.")
    if int(persistencia_dias) < 1:
        raise ValueError("La persistencia térmica debe ser de al menos un día.")

    serie = pd.Series(np.asarray(factor_termico, dtype=float))
    cumple = serie.ge(float(factor_minimo))

    if int(persistencia_dias) > 1:
        cumple = (
            cumple.astype(int)
            .rolling(
                window=int(persistencia_dias),
                min_periods=int(persistencia_dias),
            )
            .sum()
            .ge(int(persistencia_dias))
        )

    return cumple.fillna(False).to_numpy(dtype=bool)

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])
    def normalize(self, X): return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        a1 = np.tanh(Xn @ self.IW + self.bIW)
        emerrel = (np.tanh((a1 @ self.LW.T).flatten() + self.bLW) + 1) / 2
        return emerrel, np.cumsum(emerrel)

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE / "IW.npy"), np.load(BASE / "bias_IW.npy"), np.load(BASE / "LW.npy"), np.load(BASE / "bias_out.npy"))
        with open(BASE / "modelo_clusters_k3.pkl", "rb") as f: k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

def load_data(file_uploader, default_name):
    if file_uploader:
        return pd.read_excel(file_uploader) if file_uploader.name.endswith((".xlsx", ".xls")) else pd.read_csv(file_uploader)
    elif (BASE / f"{default_name}.csv").exists():
        return pd.read_csv(BASE / f"{default_name}.csv")
    elif (BASE / f"{default_name}.xlsx").exists():
        return pd.read_excel(BASE / f"{default_name}.xlsx")
    
    github_url = f"https://raw.githubusercontent.com/PREDWEEM/LOLIUM_LARTIGAU-2026/main/{default_name}.csv"
    try:
        return pd.read_csv(github_url)
    except:
        return None

def sincronizar_intervalos_variables(df_sim, df_campo, col_fecha, col_plm2):
    df_campo = df_campo.sort_values(col_fecha).copy()
    df_campo['Campo_Acum_Abs'] = df_campo[col_plm2].cumsum()
    
    fechas_reales = df_campo[col_fecha].tolist()
    registros = []
    
    for i in range(1, len(fechas_reales)):
        f_inicio = fechas_reales[i-1]
        f_fin = fechas_reales[i]
        dias_intervalo = (f_fin - f_inicio).days
        
        obs_inicio = df_campo.loc[df_campo[col_fecha] == f_inicio, 'Campo_Acum_Abs'].values[0]
        obs_fin = df_campo.loc[df_campo[col_fecha] == f_fin, 'Campo_Acum_Abs'].values[0]
        flujo_obs = max(0.0, obs_fin - obs_inicio)
        
        mask_sim = (df_sim['Fecha'] > f_inicio) & (df_sim['Fecha'] <= f_fin)
        flujo_sim = df_sim.loc[mask_sim, 'EMERREL'].sum()
        
        acum_sim_fin = df_sim.loc[df_sim['Fecha'] <= f_fin, 'EMERREL'].sum()
        
        registros.append({
            'Fecha': f_fin,
            'Dias_Intervalo': dias_intervalo,
            'Flujo_Obs_Abs': flujo_obs,
            'Flujo_Sim_Abs': flujo_sim,
            'Acum_Obs_Abs': obs_fin,
            'Acum_Sim_Abs': acum_sim_fin
        })
        
    df_res = pd.DataFrame(registros)
    if df_res.empty:
        return pd.DataFrame()
        
    total_obs = df_res['Flujo_Obs_Abs'].sum()
    total_sim = df_sim.loc[df_sim['Fecha'] <= fechas_reales[-1], 'EMERREL'].sum()
    
    df_res['Campo_Relativo'] = df_res['Flujo_Obs_Abs'] / total_obs if total_obs > 0 else 0.0
    df_res['Sim_Relativo'] = df_res['Flujo_Sim_Abs'] / total_sim if total_sim > 0 else 0.0
    
    df_res['Campo_Acumulado'] = df_res['Acum_Obs_Abs'] / df_campo['Campo_Acum_Abs'].max() if df_campo['Campo_Acum_Abs'].max() > 0 else 0.0
    df_res['Sim_Acumulado'] = df_res['Acum_Sim_Abs'] / df_sim['EMERREL'].sum() if df_sim['EMERREL'].sum() > 0 else 0.0
    
    return df_res

def calcular_metricas_validacion_integral(df_sync, umbral_deteccion=0.05):
    if df_sync.empty or len(df_sync) < 2:
        return {"Pearson_Flujos": 0.0, "NSE_Flujos": 0.0, "KGE_Flujos": 0.0, 
                "RMSE_Acumulado": 0.0, "CCC_Acumulado": 0.0, "R2_Acumulado": 0.0,
                "Exactitud_Global": 0.0, "F1_Score_Coincidencia": 0.0, 
                "Hits": 0, "Misses": 0, "Falsos_Positivos": 0, "Correctos_Negativos": 0}

    mask_activos = (df_sync['Campo_Relativo'] > 0) | (df_sync['Sim_Relativo'] > 0)
    df_activos = df_sync[mask_activos].copy()
    
    if len(df_activos) < 2:
        pearson_r, nse_flujos, kge_flujos = 0.0, 0.0, 0.0
    else:
        obs = df_activos['Campo_Relativo'].values
        sim = df_activos['Sim_Relativo'].values
        
        std_obs, std_sim = np.std(obs), np.std(sim)
        pearson_r = np.corrcoef(obs, sim)[0, 1] if std_obs > 0 and std_sim > 0 else 0.0
        
        var_obs_sum = np.sum((obs - np.mean(obs))**2)
        nse_flujos = 1 - (np.sum((sim - obs)**2) / var_obs_sum) if var_obs_sum > 0 else 0.0
        
        if np.mean(obs) > 0 and std_obs > 0:
            r = pearson_r
            alpha = std_sim / std_obs
            beta = np.mean(sim) / np.mean(obs)
            kge_flujos = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            kge_flujos = 0.0

    obs_acum, sim_acum = df_sync['Campo_Acumulado'].values, df_sync['Sim_Acumulado'].values
    rmse_acumulado = np.sqrt(np.mean((obs_acum - sim_acum)**2))
    
    mean_obs_ac, mean_sim_ac = np.mean(obs_acum), np.mean(sim_acum)
    var_obs_ac, var_sim_ac = np.var(obs_acum), np.var(sim_acum)
    covar_ac = np.mean((obs_acum - mean_obs_ac) * (sim_acum - mean_sim_ac))
    
    denominador_ccc = var_obs_ac + var_sim_ac + (mean_obs_ac - mean_sim_ac)**2
    ccc_acumulado = (2 * covar_ac) / denominador_ccc if denominador_ccc > 0 else 0.0
    
    ss_res_ac = np.sum((obs_acum - sim_acum)**2)
    ss_tot_ac = np.sum((obs_acum - mean_obs_ac)**2)
    r2_acumulado = 1 - (ss_res_ac / ss_tot_ac) if ss_tot_ac > 0 else 0.0
    
    # --- MÉTRICAS DE COINCIDENCIA POR INTERVALO (AGREEMENT) ---
    obs_eventos = df_sync['Campo_Relativo'] > umbral_deteccion
    sim_eventos = df_sync['Sim_Relativo'] > umbral_deteccion

    hits = np.sum(obs_eventos & sim_eventos)                 
    misses = np.sum(obs_eventos & ~sim_eventos)              
    false_alarms = np.sum(~obs_eventos & sim_eventos)        
    correct_negatives = np.sum(~obs_eventos & ~sim_eventos)  

    total_intervalos = len(df_sync)

    exactitud = (hits + correct_negatives) / total_intervalos if total_intervalos > 0 else 0.0
    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    recall = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "Pearson_Flujos": pearson_r, 
        "NSE_Flujos": nse_flujos,
        "KGE_Flujos": kge_flujos,
        "RMSE_Acumulado": rmse_acumulado, 
        "CCC_Acumulado": ccc_acumulado,
        "R2_Acumulado": r2_acumulado,
        "Exactitud_Global": exactitud,
        "F1_Score_Coincidencia": f1_score,
        "Hits": int(hits),
        "Misses": int(misses),
        "Falsos_Positivos": int(false_alarms),
        "Correctos_Negativos": int(correct_negatives)
    }

# ---------------------------------------------------------
# 4.5 OPTIMIZADOR EXCLUSIVO DE TERMOINHIBICIÓN
# ---------------------------------------------------------
def _escalar_metrica_menos1_a_1(valor):
    """Convierte una métrica acotada conceptualmente entre -1 y 1 a 0-1."""
    if valor is None or not np.isfinite(valor):
        return 0.0
    return float((np.clip(valor, -1.0, 1.0) + 1.0) / 2.0)


def _score_termoinhibicion(metricas, lag_inicio=None, lag_t50=None):
    """Índice compuesto para ordenar combinaciones térmicas.

    La ponderación prioriza la coincidencia de eventos y el ajuste de los
    flujos, pero también penaliza desfases del inicio y del T50.
    """
    f1 = float(np.clip(metricas["F1_Score_Coincidencia"], 0.0, 1.0))
    exactitud = float(np.clip(metricas["Exactitud_Global"], 0.0, 1.0))
    nse = _escalar_metrica_menos1_a_1(metricas["NSE_Flujos"])
    kge = _escalar_metrica_menos1_a_1(metricas["KGE_Flujos"])
    ccc = _escalar_metrica_menos1_a_1(metricas["CCC_Acumulado"])
    r2 = _escalar_metrica_menos1_a_1(metricas["R2_Acumulado"])

    rmse = metricas["RMSE_Acumulado"]
    score_rmse = float(np.exp(-4.0 * rmse)) if np.isfinite(rmse) else 0.0

    score_inicio = (
        float(np.exp(-abs(lag_inicio) / 14.0))
        if lag_inicio is not None and np.isfinite(lag_inicio)
        else 0.0
    )
    score_t50 = (
        float(np.exp(-abs(lag_t50) / 14.0))
        if lag_t50 is not None and np.isfinite(lag_t50)
        else 0.0
    )

    return (
        0.25 * f1
        + 0.10 * exactitud
        + 0.15 * nse
        + 0.15 * kge
        + 0.10 * ccc
        + 0.08 * r2
        + 0.07 * score_rmse
        + 0.05 * score_inicio
        + 0.05 * score_t50
    )


def _grilla_termoinhibicion(modo, valores_actuales):
    """Construye la grilla sin variar parámetros hídricos ni de la ANN."""
    media_actual, desvio_actual, factor_actual, persistencia_actual = valores_actuales

    if modo == "Rápido":
        medias = np.arange(20.0, 29.1, 1.0)
        desvios = np.arange(0.5, 4.1, 0.5)
        factores = np.arange(0.25, 0.76, 0.10)
        persistencias = np.arange(1, 5, 1, dtype=int)

    elif modo == "Fino alrededor de valores actuales":
        medias = np.arange(
            max(15.0, media_actual - 2.0),
            min(35.0, media_actual + 2.0) + 0.001,
            0.25,
        )
        desvios = np.arange(
            max(0.25, desvio_actual - 1.0),
            min(10.0, desvio_actual + 1.0) + 0.001,
            0.25,
        )
        factores = np.arange(
            max(0.05, factor_actual - 0.20),
            min(0.95, factor_actual + 0.20) + 0.001,
            0.05,
        )
        persistencias = np.arange(
            max(1, int(persistencia_actual) - 2),
            min(10, int(persistencia_actual) + 2) + 1,
            1,
            dtype=int,
        )

    else:  # Estándar
        medias = np.arange(19.0, 30.1, 0.5)
        desvios = np.arange(0.5, 5.1, 0.5)
        factores = np.arange(0.20, 0.81, 0.10)
        persistencias = np.arange(1, 6, 1, dtype=int)

    return (
        np.round(medias, 4),
        np.round(desvios, 4),
        np.round(factores, 4),
        persistencias,
    )


def optimizar_parametros_termoinhibicion(
    df_meteo,
    df_campo,
    modelo_ann,
    mod_termico,
    w_max_fijo,
    ke_fijo,
    umbral_choque_hidrico_fijo,
    umbral_alerta,
    modo_busqueda="Estándar",
    criterio_orden="Score compuesto",
    valores_actuales=(24.0, 2.0, 0.35, 2),
    latitud_lartigau=-38.6166,
    progreso_callback=None,
):
    """Optimiza solo T50, sigma, factor mínimo y persistencia térmica.

    Permanecen constantes:
    - pesos y sesgos de la ANN;
    - capacidad hídrica superficial;
    - coeficiente Ke;
    - cobertura/modulador térmico;
    - umbral de choque hídrico;
    - latencia estricta de los primeros 25 días.
    """
    df = df_meteo.copy()
    df.columns = [str(c).upper().strip() for c in df.columns]
    df = df.rename(
        columns={
            "FECHA": "Fecha",
            "DATE": "Fecha",
            "TMAX": "TMAX",
            "TMIN": "TMIN",
            "PREC": "Prec",
            "LLUVIA": "Prec",
        }
    )

    requeridas = {"Fecha", "TMAX", "TMIN", "Prec"}
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(
            "Faltan columnas meteorológicas: " + ", ".join(sorted(faltantes))
        )

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    for columna in ["TMAX", "TMIN", "Prec"]:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df = (
        df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"])
        .sort_values("Fecha")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No quedaron datos meteorológicos válidos.")

    campo = df_campo.copy()
    col_fecha = "FECHA" if "FECHA" in campo.columns else campo.columns[0]
    col_plm2 = "PLM2" if "PLM2" in campo.columns else campo.columns[1]
    campo[col_fecha] = pd.to_datetime(campo[col_fecha], errors="coerce")
    campo[col_plm2] = pd.to_numeric(campo[col_plm2], errors="coerce")
    campo = (
        campo.dropna(subset=[col_fecha, col_plm2])
        .sort_values(col_fecha)
        .reset_index(drop=True)
    )
    if len(campo) < 2:
        raise ValueError("Se requieren al menos dos fechas válidas de campo.")

    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2.0
    amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2.0
    df["TMAX_suelo"] = df["Tmedia_aire"] + amplitud_termica * float(mod_termico)
    df["TMIN_suelo"] = df["Tmedia_aire"] - amplitud_termica * float(mod_termico)

    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    emerrel_base = np.maximum(emerrel_raw, 0.0)

    # Misma lógica no térmica del motor principal.
    prec_3d = df["Prec"].rolling(window=3, min_periods=1).sum().to_numpy()
    mask_ruptura = (
        (df["Julian_days"].to_numpy() > 25)
        & (df["Julian_days"].to_numpy() <= 110)
        & (prec_3d >= float(umbral_choque_hidrico_fijo))
    )
    emerrel_base[mask_ruptura] = np.maximum(emerrel_base[mask_ruptura], 1.0)

    et0 = calcular_et0_hargreaves(
        df["Julian_days"].to_numpy(),
        df["TMAX"].to_numpy(),
        df["TMIN"].to_numpy(),
        latitud=latitud_lartigau,
    )
    agua = balance_hidrico_superficial(
        df["Prec"].to_numpy(),
        et0,
        w_max=float(w_max_fijo),
        ke_suelo=float(ke_fijo),
    )
    humedad_relativa = agua / float(w_max_fijo)
    factor_hidrico = 1.0 / (1.0 + np.exp(-10.0 * (humedad_relativa - 0.3)))

    emerrel_base *= factor_hidrico
    emerrel_base[humedad_relativa < 0.20] = 0.0

    lluvia_recarga = pd.Series(
        df["Prec"].to_numpy() >= float(w_max_fijo)
    ).cummax().to_numpy()
    emerrel_base[~lluvia_recarga] = 0.0
    emerrel_base[df["Julian_days"].to_numpy() <= 25] = 0.0

    tmedia_10d = (
        df["Tmedia_aire"]
        .rolling(window=10, min_periods=1)
        .mean()
        .to_numpy()
    )

    medias, desvios, factores, persistencias = _grilla_termoinhibicion(
        modo_busqueda,
        valores_actuales,
    )
    total = len(medias) * len(desvios) * len(factores) * len(persistencias)

    fecha_obs_inicio = None
    positivos = campo[campo[col_plm2] > 0]
    if not positivos.empty:
        fecha_obs_inicio = positivos.iloc[0][col_fecha]

    total_obs = campo[col_plm2].sum()
    fecha_t50_obs = None
    if total_obs > 0:
        acum_obs = campo[col_plm2].cumsum() / total_obs
        candidatos = campo.loc[acum_obs >= 0.5, col_fecha]
        if not candidatos.empty:
            fecha_t50_obs = candidatos.iloc[0]

    resultados = []
    realizado = 0
    fechas = df["Fecha"]

    for media in medias:
        for desvio in desvios:
            factor_termico = factor_termoinhibicion_normal(
                tmedia_10d,
                media=float(media),
                desvio=float(desvio),
            )

            for factor_min in factores:
                for persistencia in persistencias:
                    habilitada = habilitacion_termica(
                        factor_termico,
                        factor_minimo=float(factor_min),
                        persistencia_dias=int(persistencia),
                    )

                    emerrel = emerrel_base * factor_termico
                    emerrel[~habilitada] = 0.0

                    df_sim = pd.DataFrame(
                        {
                            "Fecha": fechas,
                            "EMERREL": emerrel,
                        }
                    )
                    df_sync = sincronizar_intervalos_variables(
                        df_sim,
                        campo,
                        col_fecha,
                        col_plm2,
                    )
                    metricas = calcular_metricas_validacion_integral(df_sync)

                    fechas_inicio_sim = fechas[emerrel >= float(umbral_alerta)]
                    fecha_inicio_sim = (
                        fechas_inicio_sim.iloc[0]
                        if len(fechas_inicio_sim) > 0
                        else None
                    )
                    lag_inicio = (
                        (fecha_inicio_sim - fecha_obs_inicio).days
                        if fecha_inicio_sim is not None
                        and fecha_obs_inicio is not None
                        else None
                    )

                    lag_t50 = None
                    mask_periodo_campo = fechas <= campo[col_fecha].max()
                    emer_periodo = emerrel[mask_periodo_campo.to_numpy()]
                    fechas_periodo = fechas[mask_periodo_campo]
                    suma_sim = emer_periodo.sum()
                    if suma_sim > 0 and fecha_t50_obs is not None:
                        acum_sim = np.cumsum(emer_periodo) / suma_sim
                        indices_t50 = np.flatnonzero(acum_sim >= 0.5)
                        if len(indices_t50) > 0:
                            fecha_t50_sim = fechas_periodo.iloc[indices_t50[0]]
                            lag_t50 = (fecha_t50_sim - fecha_t50_obs).days

                    score = _score_termoinhibicion(
                        metricas,
                        lag_inicio=lag_inicio,
                        lag_t50=lag_t50,
                    )

                    resultados.append(
                        {
                            "Media T50 (°C)": float(media),
                            "Desvío σ (°C)": float(desvio),
                            "Factor mínimo": float(factor_min),
                            "Persistencia (días)": int(persistencia),
                            "Score compuesto": score,
                            "F1-Score": metricas["F1_Score_Coincidencia"],
                            "Exactitud": metricas["Exactitud_Global"],
                            "NSE (Flujos)": metricas["NSE_Flujos"],
                            "KGE": metricas["KGE_Flujos"],
                            "CCC (Acumulado)": metricas["CCC_Acumulado"],
                            "R2 (Acumulado)": metricas["R2_Acumulado"],
                            "RMSE (Acumulado)": metricas["RMSE_Acumulado"],
                            "Lag inicio (días)": lag_inicio,
                            "|Lag inicio|": (
                                abs(lag_inicio) if lag_inicio is not None else np.inf
                            ),
                            "Lag T50 (días)": lag_t50,
                            "|Lag T50|": (
                                abs(lag_t50) if lag_t50 is not None else np.inf
                            ),
                        }
                    )

                    realizado += 1
                    if progreso_callback and (
                        realizado == total or realizado % max(1, total // 100) == 0
                    ):
                        progreso_callback(realizado, total)

    resultados = pd.DataFrame(resultados)
    if resultados.empty:
        return resultados

    ordenes = {
        "Score compuesto": ("Score compuesto", False),
        "F1-Score": ("F1-Score", False),
        "NSE (Flujos)": ("NSE (Flujos)", False),
        "KGE": ("KGE", False),
        "CCC (Acumulado)": ("CCC (Acumulado)", False),
        "RMSE (Acumulado)": ("RMSE (Acumulado)", True),
        "|Lag inicio|": ("|Lag inicio|", True),
        "|Lag T50|": ("|Lag T50|", True),
    }
    columna, ascendente = ordenes.get(
        criterio_orden,
        ("Score compuesto", False),
    )

    return resultados.sort_values(
        by=[columna, "F1-Score", "NSE (Flujos)"],
        ascending=[ascendente, False, False],
        na_position="last",
    ).reset_index(drop=True)

# ---------------------------------------------------------
# 5. INTERFAZ PRINCIPAL Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()

st.title("🌾 PREDWEEM LOLIUM — LARTIGAU (BA) lat=-38.6166 lon=-61.7000")

with st.expander("📂 1. Datos del Lote", expanded=True):
    col_upload, col_rastrojo = st.columns(2)
    
    with col_upload:
        archivo_meteo = st.file_uploader("1. Clima (Lartigau)", type=["xlsx", "csv"])
        archivo_campo = st.file_uploader("2. Campo (Validación Real Variable)", type=["xlsx", "csv"])
        
    with col_rastrojo:
        with st.container(border=True):
            st.markdown("#### 🌾 Manejo de Superficie")
            cobertura_pct = st.slider(
                "Cobertura de Rastrojo en Suelo (%)",
                min_value=0, max_value=100, value=70, step=5,
                help="0% = Suelo desnudo. 100% = Cobertura total (Lartigau Calibración Óptima = 70%)."
            )

            x_cobertura = [0, 30, 70, 100]
            ke_val = float(np.interp(cobertura_pct, x_cobertura, [0.95, 0.50, 0.25, 0.10]))
            mod_termico = float(np.interp(cobertura_pct, x_cobertura, [1.00, 0.95, 0.90, 0.80]))

            html_card = f"""
            <div style="background-color: #ffffff; padding: 15px 20px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border: 1px solid #e2e8f0; margin-top: 15px;">
                <h5 style="color: #1e293b; margin-top: 0; margin-bottom: 12px; font-size: 0.95rem;">Parámetros Dinámicos Aplicados</h5>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #475569; font-size: 0.9rem;">Coeficiente Hídrico Suelo (Ke):</span>
                    <span style="color: #0284c7; font-weight: bold; font-size: 1.05rem;">{ke_val:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #475569; font-size: 0.9rem;">Modulador Térmico Suelo:</span>
                    <span style="color: #b91c1c; font-weight: bold; font-size: 1.05rem;">{mod_termico:.2f}</span>
                </div>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)

df_meteo_raw = load_data(archivo_meteo, "meteo_daily")
df_campo_raw = load_data(archivo_campo, "LARTIGAU_campo")

# --- SIDEBAR ---
st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/LOLIUM_LARTIGAU-2026/main/logo.png", width="stretch")

st.sidebar.markdown("## ⚙️ 2. Fisiología y Logística")
umbral_er = st.sidebar.slider("Umbral Alerta Temprana", 0.001, 0.80, 0.005)

st.sidebar.markdown("**Termoinhibición — distribución normal**")
col_tm, col_ts = st.sidebar.columns(2)
with col_tm:
    media_termoinhibicion = st.number_input(
        "Media T50 (°C)",
        min_value=15.0,
        max_value=35.0,
        value=23.5,
        step=0.5,
        help="Temperatura media móvil donde el factor térmico vale 0,50.",
    )
with col_ts:
    desvio_termoinhibicion = st.number_input(
        "Desvío σ (°C)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Controla cuán gradual es la transición entre aptitud e inhibición.",
    )


factor_min_inicio_termico = st.sidebar.slider(
    "Factor térmico mínimo para iniciar",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help=(
        "La emergencia permanece en cero mientras el factor de aptitud "
        "térmica sea inferior a este valor."
    ),
)

persistencia_termica_dias = st.sidebar.number_input(
    "Persistencia térmica para iniciar (días)",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
    help=(
        "Días consecutivos con aptitud térmica suficiente antes de habilitar "
        "el inicio de la emergencia."
    ),
)

st.sidebar.markdown("**Ruptura de Dormición (Otoño)**")
umbral_choque_hidrico = st.sidebar.slider("Choque Hídrico 3 días (mm)", 20.0, 100.0, 30.0)

residualidad = st.sidebar.number_input("Residualidad Herbicida (días)", 0, 60, 0)

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", value=2.0, step=0.5)
with col_t2: t_opt_max = st.number_input("T Óptima Max", value=20.0, step=1.0)
t_critica = st.sidebar.slider("T Crítica (Stop)", 26.0, 42.0, 30.0)

st.sidebar.markdown("**Objetivos (°Cd)**")
dga_optimo = st.sidebar.number_input("TT Control Post-emergente", value=600, step=10)
dga_critico = st.sidebar.number_input("Límite Ventana", value=800, step=10)

st.sidebar.divider()
st.sidebar.markdown("## 💧 3. Balance Hídrico (Suelo)")
w_max_val = st.sidebar.number_input("Cap. de Campo Superficial (mm)", value=20.0, step=1.0)

st.sidebar.divider()
st.sidebar.markdown("## 📊 4. Estado de Validación")
st.sidebar.info("🔬 **Modo Event-to-Event Activado**: Las ventanas de validación se auto-ajustan dinámicamente según el calendario real de tus datos de campo (7-21 días).")

# --- OPTIMIZADOR EXCLUSIVO DE TERMOINHIBICIÓN ---
with st.sidebar.expander(
    "🌡️ Optimizador de termoinhibición",
    expanded=False,
):
    st.caption(
        "Optimiza solamente T50, σ, factor mínimo y persistencia. "
        "W_Max, Ke, cobertura, ANN y choque hídrico permanecen fijos."
    )

    modo_opt_termico = st.selectbox(
        "Resolución de búsqueda",
        [
            "Rápido",
            "Estándar",
            "Fino alrededor de valores actuales",
        ],
        index=1,
        key="modo_opt_termico",
    )

    criterio_opt_termico = st.selectbox(
        "Criterio principal",
        [
            "Score compuesto",
            "F1-Score",
            "NSE (Flujos)",
            "KGE",
            "CCC (Acumulado)",
            "RMSE (Acumulado)",
            "|Lag inicio|",
            "|Lag T50|",
        ],
        index=0,
        key="criterio_opt_termico",
    )

    if st.button(
        "Ejecutar optimización térmica",
        key="boton_opt_termico",
        width="stretch",
    ):
        if (
            df_meteo_raw is not None
            and df_campo_raw is not None
            and modelo_ann is not None
        ):
            barra_opt = st.progress(0)
            estado_opt = st.empty()

            def actualizar_progreso_termico(realizado, total):
                fraccion = realizado / total if total else 1.0
                barra_opt.progress(min(fraccion, 1.0))
                estado_opt.caption(
                    f"Evaluando combinación {realizado:,} de {total:,}"
                )

            try:
                with st.spinner(
                    "Calibrando exclusivamente la termoinhibición..."
                ):
                    tabla_optima_termica = optimizar_parametros_termoinhibicion(
                        df_meteo=df_meteo_raw,
                        df_campo=df_campo_raw,
                        modelo_ann=modelo_ann,
                        mod_termico=mod_termico,
                        w_max_fijo=w_max_val,
                        ke_fijo=ke_val,
                        umbral_choque_hidrico_fijo=umbral_choque_hidrico,
                        umbral_alerta=umbral_er,
                        modo_busqueda=modo_opt_termico,
                        criterio_orden=criterio_opt_termico,
                        valores_actuales=(
                            media_termoinhibicion,
                            desvio_termoinhibicion,
                            factor_min_inicio_termico,
                            persistencia_termica_dias,
                        ),
                        latitud_lartigau=-38.6166,
                        progreso_callback=actualizar_progreso_termico,
                    )

                st.session_state["resultado_opt_termico"] = (
                    tabla_optima_termica
                )
                barra_opt.progress(1.0)
                estado_opt.caption("Optimización finalizada.")

            except Exception as exc:
                st.error(f"No se pudo ejecutar la optimización: {exc}")

        else:
            st.error(
                "Se requieren datos meteorológicos, datos de campo "
                "y el modelo ANN."
            )

    if "resultado_opt_termico" in st.session_state:
        tabla_optima_termica = st.session_state["resultado_opt_termico"]

        if not tabla_optima_termica.empty:
            mejor = tabla_optima_termica.iloc[0]

            st.success(
                "Mejor combinación: "
                f"T50={mejor['Media T50 (°C)']:.2f} °C; "
                f"σ={mejor['Desvío σ (°C)']:.2f} °C; "
                f"factor={mejor['Factor mínimo']:.2f}; "
                f"persistencia={int(mejor['Persistencia (días)'])} días."
            )

            st.metric(
                "Score compuesto",
                f"{mejor['Score compuesto']:.3f}",
            )
            st.caption(
                f"F1={mejor['F1-Score']:.3f} | "
                f"NSE={mejor['NSE (Flujos)']:.3f} | "
                f"KGE={mejor['KGE']:.3f} | "
                f"CCC={mejor['CCC (Acumulado)']:.3f}"
            )

            columnas_mostrar = [
                "Media T50 (°C)",
                "Desvío σ (°C)",
                "Factor mínimo",
                "Persistencia (días)",
                "Score compuesto",
                "F1-Score",
                "NSE (Flujos)",
                "KGE",
                "CCC (Acumulado)",
                "RMSE (Acumulado)",
                "Lag inicio (días)",
                "Lag T50 (días)",
            ]
            st.dataframe(
                tabla_optima_termica[columnas_mostrar].head(20),
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Descargar resultados del optimizador",
                data=tabla_optima_termica.to_csv(index=False).encode("utf-8"),
                file_name="optimizacion_termoinhibicion_Lartigau.csv",
                mime="text/csv",
                width="stretch",
            )

# ---------------------------------------------------------
# 6. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if df_meteo_raw is not None and modelo_ann is not None:

    df = df_meteo_raw.copy()
    df.columns = [c.upper().strip() for c in df.columns]
    df = df.rename(columns={'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'})
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "Prec"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear

    # Simulación Térmica
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2
    amplitud_termica = (df["TMAX"] - df["TMIN"]) / 2
    df["TMAX_suelo"] = df["Tmedia_aire"] + (amplitud_termica * mod_termico)
    df["TMIN_suelo"] = df["Tmedia_aire"] - (amplitud_termica * mod_termico)

    df_campo, col_fecha, col_plm2 = None, None, None
    if df_campo_raw is not None:
        df_campo = df_campo_raw.copy()
        col_fecha = 'FECHA' if 'FECHA' in df_campo.columns else df_campo.columns[0]
        col_plm2 = 'PLM2' if 'PLM2' in df_campo.columns else df_campo.columns[1]
        df_campo[col_fecha] = pd.to_datetime(df_campo[col_fecha])
        df_campo = df_campo.sort_values(col_fecha).reset_index(drop=True)
        max_plm2 = df_campo[col_plm2].max()
        df_campo['Campo_Normalizado'] = df_campo[col_plm2] / max_plm2 if max_plm2 > 0 else 0

    # ----------------------------------------------------
    # CORRECCIÓN: Lógica Fisiológica Ordenada
    # ----------------------------------------------------
    # 1. Predicción Neural Base
    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)

    # 2. Bypass Ruptura Temprana (Lartigau = 1.0) - ESTRICTAMENTE LUEGO DEL DÍA 25
    df["Prec_3d"] = df["Prec"].rolling(window=3, min_periods=1).sum()
    mask_ruptura = (df["Julian_days"] > 25) & (df["Julian_days"] <= 110) & (df["Prec_3d"] >= umbral_choque_hidrico)
    df.loc[mask_ruptura, "EMERREL"] = np.maximum(df.loc[mask_ruptura, "EMERREL"], 1.0)

    # 3. Balance Hídrico Superficial (Lartigau)
    df["ET0"] = calcular_et0_hargreaves(df["Julian_days"].values, df["TMAX"].values, df["TMIN"].values, latitud=-38.6166)
    df["W_superficial"] = balance_hidrico_superficial(df["Prec"].values, df["ET0"].values, w_max=w_max_val, ke_suelo=ke_val)
    humedad_relativa = df["W_superficial"] / w_max_val
    df["Hydric_Factor"] = 1 / (1 + np.exp(-10 * (humedad_relativa - 0.3)))
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]

    df.loc[humedad_relativa < 0.20, "EMERREL"] = 0.0
    df['Lluvia_Recarga'] = (df['Prec'] >= w_max_val).cummax()
    df.loc[~df['Lluvia_Recarga'], "EMERREL"] = 0.0

    # 4. Termoinhibición continua mediante distribución normal acumulada
    df["Tmedia"] = df["Tmedia_aire"]
    df["Tmedia_10d"] = (
        df["Tmedia"]
        .rolling(window=10, min_periods=1)
        .mean()
    )
    df["Factor_Termoinhibicion"] = factor_termoinhibicion_normal(
        df["Tmedia_10d"].values,
        media=media_termoinhibicion,
        desvio=desvio_termoinhibicion,
    )
    df["EMERREL"] *= df["Factor_Termoinhibicion"]

    df["Habilitacion_Termica"] = habilitacion_termica(
        df["Factor_Termoinhibicion"].values,
        factor_minimo=factor_min_inicio_termico,
        persistencia_dias=persistencia_termica_dias,
    )
    df.loc[~df["Habilitacion_Termica"], "EMERREL"] = 0.0

    # 5. BLOQUEO FINAL ESTRICTO: Latencia Temprana (Primeros 25 días del año)
    df.loc[df["Julian_days"] <= 25, "EMERREL"] = 0.0
    # ----------------------------------------------------

    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    fecha_hoy = pd.Timestamp.now().normalize()
    if fecha_hoy not in df['Fecha'].values: fecha_hoy = df['Fecha'].max()
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()

    # --- LOGÍSTICA DE RECUENTO TÉRMICO ---
    dga_hoy, dga_7dias = 0.0, 0.0
    fecha_inicio_ventana, fecha_control, fecha_limite = None, None, None
    msg_estado = "Esperando pulso de emergencia..."

    if indices_pulso:
        fecha_inicio_ventana = df.loc[indices_pulso[0], "Fecha"]
        df_desde_pico = df[df["Fecha"] >= fecha_inicio_ventana].copy()
        df_desde_pico["DGA_cum"] = df_desde_pico["DG"].cumsum()
        
        df_control = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_optimo]
        if not df_control.empty: fecha_control = df_control.iloc[0]["Fecha"]
        
        df_limite = df_desde_pico[df_desde_pico["DGA_cum"] >= dga_critico]
        if not df_limite.empty: fecha_limite = df_limite.iloc[0]["Fecha"]
        
        dga_hoy = df.loc[(df["Fecha"] >= fecha_inicio_ventana) & (df["Fecha"] <= fecha_hoy), "DG"].sum()
        idx_hoy = df[df["Fecha"] == fecha_hoy].index[0]
        
        dga_7dias = dga_hoy + df.iloc[idx_hoy + 1: idx_hoy + 8]["DG"].sum() if idx_hoy + 8 <= len(df) else dga_hoy
        msg_estado = f"Pico detectado el {fecha_inicio_ventana.strftime('%d/%m')}"

    # Sincronización Event-to-Event
    pearson_r, nse_flujos, kge_flujos, rmse_acum, ccc_acum, r2_acum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    exactitud_global, f1_score_coincidencia = 0.0, 0.0
    hits_val, misses_val, falsos_pos_val, correctos_neg_val = 0, 0, 0, 0
    pec, peak_lag, lead_time, desfase_t50 = 0.0, 0, 0, 0
    df_sincronizado = pd.DataFrame()
    
    # --- CÁLCULO DE SINCRONÍA DE INICIO ---
    lag_inicio_dias = None
    fecha_primer_flujo_obs = None

    if df_campo is not None:
        muestreos_con_plantas = df_campo[df_campo[col_plm2] > 0]
        if not muestreos_con_plantas.empty:
            fecha_primer_flujo_obs = muestreos_con_plantas.iloc[0][col_fecha]
            if fecha_inicio_ventana is not None:
                lag_inicio_dias = (fecha_inicio_ventana - fecha_primer_flujo_obs).days

        df_sincronizado = sincronizar_intervalos_variables(df, df_campo, col_fecha, col_plm2)
        if not df_sincronizado.empty:
            metricas_robustas = calcular_metricas_validacion_integral(df_sincronizado, umbral_deteccion=0.05)
            
            pearson_r = metricas_robustas["Pearson_Flujos"]
            nse_flujos = metricas_robustas["NSE_Flujos"]
            kge_flujos = metricas_robustas["KGE_Flujos"]
            rmse_acum = metricas_robustas["RMSE_Acumulado"]
            ccc_acum = metricas_robustas["CCC_Acumulado"]
            r2_acum = metricas_robustas["R2_Acumulado"]
            exactitud_global = metricas_robustas["Exactitud_Global"]
            f1_score_coincidencia = metricas_robustas["F1_Score_Coincidencia"]
            
            hits_val = metricas_robustas["Hits"]
            misses_val = metricas_robustas["Misses"]
            falsos_pos_val = metricas_robustas["Falsos_Positivos"]
            correctos_neg_val = metricas_robustas["Correctos_Negativos"]

            tot_plm2 = df_campo[col_plm2].sum()
            if tot_plm2 > 0:
                df_campo['cum_plm2_norm'] = df_campo[col_plm2].cumsum() / tot_plm2
                t50_obs_date = df_campo[df_campo['cum_plm2_norm'] >= 0.5].iloc[0][col_fecha]
                df_sim_trunc = df[df['Fecha'] <= df_campo[col_fecha].max()].copy()
                tot_emer = df_sim_trunc['EMERREL'].sum()
                
                if tot_emer > 0:
                    df_sim_trunc['cum_emer_norm'] = df_sim_trunc['EMERREL'].cumsum() / tot_emer
                    t50_sim_date = df_sim_trunc[df_sim_trunc['cum_emer_norm'] >= 0.5].iloc[0]['Fecha']
                    desfase_t50 = (t50_sim_date - t50_obs_date).days

            if fecha_control:
                malezas_totales_campo = df_campo[col_plm2].sum()
                pec = ((df_campo.loc[df_campo[col_fecha] <= fecha_control, col_plm2].sum() / malezas_totales_campo) * 100 if malezas_totales_campo > 0 else 0)
                peak_lag = (fecha_control - df_campo.loc[df_campo[col_plm2].idxmax(), col_fecha]).days
                df_alertas = df[df['EMERREL'] >= umbral_er]
                lead_time = (fecha_control - (df_alertas['Fecha'].iloc[0] if not df_alertas.empty else fecha_inicio_ventana)).days

    # Transformación Logarítmica Analítica
    c_log = 0.01
    df["EMERREL_LOG"] = np.log10(df["EMERREL"] + c_log)
    umbral_er_log = np.log10(umbral_er + c_log)
    if df_campo is not None:
        df_campo['Campo_Normalizado_LOG'] = np.log10(df_campo['Campo_Normalizado'] + c_log)

    # FRONT-END VISUAL
    colorscale_hard = [[0.0, "green"], [0.01, "green"], [0.02, "red"], [1.0, "red"]]
    st.plotly_chart(go.Figure(data=go.Heatmap(z=[df["EMERREL"].values], x=df["Fecha"], y=["Emergencia"], colorscale=colorscale_hard, zmin=0, zmax=1, showscale=False)).update_layout(height=120, margin=dict(t=30, b=0, l=10, r=10), title="Mapa de Riesgo Temporal (Tasa Diaria)"), width="stretch")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR DE DECISIÓN", "💧 PRECIPITACIONES Y SUELO", "📈 ANÁLISIS ESTRATÉGICO", "🧪 BIO-CALIBRACIÓN"])

    with tab1:
        if df_campo is not None:
            st.markdown("<p class='metric-header'>🚜 FIDELIDAD CIENTÍFICA (Evaluación sobre Intervalos Reales Variable)</p>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            
            c1.metric("Eficiencia (KGE)", f"{kge_flujos:.3f}", "Ajuste por Evento")
            c2.metric("Predictivo (NSE)", f"{nse_flujos:.3f}", "Flujos Puros")
            c3.metric("Trayectoria (CCC)", f"{ccc_acum:.3f}", "Curva Acum.")
            c4.metric("Error (RMSE)", f"{rmse_acum:.3f}", "Desvío Acumulado", delta_color="inverse")
            c5.metric("Desfase (T50)", f"{desfase_t50:+d} días", "Sincronía Operativa", delta_color="inverse" if desfase_t50 > 0 else "normal" if desfase_t50 < 0 else "off")

            st.markdown("<p class='metric-header' style='margin-top:15px;'>🎯 COINCIDENCIA POR INTERVALO DE MUESTREO</p>", unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            d1.metric("F1-Score (Coincidencia)", f"{f1_score_coincidencia:.3f}", "Fidelidad en ventanas activas")
            d2.metric("Exactitud Global", f"{exactitud_global * 100:.1f}%", "Acuerdo total (Invierno + Verano)")
            d3.metric("Falsos Positivos", f"{falsos_pos_val}", "Intervalos simulados sin contraparte real", delta_color="inverse")

            # --- SECCIÓN: SINCRONÍA DE INICIO ---
            st.markdown("<p class='metric-header' style='margin-top:15px;'>⏰ SINCRONÍA DE INICIO (Gatillo de Tiempo Térmico)</p>", unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            s1.metric("Inicio Simulado", fecha_inicio_ventana.strftime('%d-%m-%Y') if fecha_inicio_ventana else "N/A")
            s2.metric("Inicio Observado (Campo)", fecha_primer_flujo_obs.strftime('%d-%m-%Y') if fecha_primer_flujo_obs else "N/A")
            
            str_lag = "N/A"
            if lag_inicio_dias is not None:
                str_lag = f"{lag_inicio_dias:+} días"
            s3.metric("Desfase de Gatillo", str_lag, "Negativo = Modelo Anticipa", delta_color="inverse" if (lag_inicio_dias and lag_inicio_dias > 0) else "normal" if (lag_inicio_dias and lag_inicio_dias < 0) else "off")
            
            # --- TABLA HTML: MATRIZ DE CONFUSIÓN ---
            html_cm = f"""
            <div style="background-color:#ffffff; padding:15px; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.1); border:1px solid #e2e8f0; margin-top:15px;">
                <p style="color:#1e293b; font-weight:bold; margin-top:0; margin-bottom:10px;">🧩 Matriz de Confusión (Intervalos de Monitoreo)</p>
                <table style="width:100%; text-align:center; border-collapse: collapse; font-family:sans-serif;">
                    <tr>
                        <th style="border-bottom:2px solid #e2e8f0; padding:10px; color:#475569; width:34%;">Realidad ⬇ / Simulación ➡</th>
                        <th style="border-bottom:2px solid #e2e8f0; padding:10px; background-color:#eff6ff; color:#1e3a8a; width:33%;">🚨 Modelo Predice FLUJO</th>
                        <th style="border-bottom:2px solid #e2e8f0; padding:10px; background-color:#f8fafc; color:#475569; width:33%;">💤 Modelo Predice INACTIVO</th>
                    </tr>
                    <tr>
                        <td style="border-bottom:1px solid #e2e8f0; padding:10px; font-weight:bold; color:#166534; background-color:#f0fdf4;">🌱 Campo: HUBO Flujo</td>
                        <td style="border-bottom:1px solid #e2e8f0; padding:10px; background-color:#dcfce7; font-size:1.1em; font-weight:bold; color:#166534;">{hits_val} <span style="font-size:0.8em; font-weight:normal;">(Hits / Coincidencia)</span></td>
                        <td style="border-bottom:1px solid #e2e8f0; padding:10px; background-color:#fee2e2; font-size:1.1em; font-weight:bold; color:#991b1b;">{misses_val} <span style="font-size:0.8em; font-weight:normal;">(Omisiones)</span></td>
                    </tr>
                    <tr>
                        <td style="padding:10px; font-weight:bold; color:#475569; background-color:#f8fafc;">🛑 Campo: SIN Flujo</td>
                        <td style="padding:10px; background-color:#fee2e2; font-size:1.1em; font-weight:bold; color:#991b1b;">{falsos_pos_val} <span style="font-size:0.8em; font-weight:normal;">(Falsas Alarmas)</span></td>
                        <td style="padding:10px; background-color:#dcfce7; font-size:1.1em; font-weight:bold; color:#166534;">{correctos_neg_val} <span style="font-size:0.8em; font-weight:normal;">(Correctos Negativos)</span></td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(html_cm, unsafe_allow_html=True)

            if fecha_control:
                st.markdown("<p class='metric-header' style='margin-top:15px;'>⚙️ LOGÍSTICA DE CONTROL EN LOTE</p>", unsafe_allow_html=True)
                l1, l2, l3 = st.columns(3)
                l1.metric("Control Efectivo (PEC)", f"{pec:.1f}%", "A la fecha de aplicación")
                l2.metric("Lag (Desfase)", f"{peak_lag} días", "Vs Pico Real de Campo")
                l3.metric("Lead Time", f"{lead_time} días", "Ventana de Alerta")
            st.markdown("---")

        col_main, col_gauge = st.columns([2, 1])

        with col_main:
            fig_emer = go.Figure()
            
            # --- SOMBREADO DINÁMICO SEGÚN FECHAS REALES DE MONITOREO ---
            if df_campo is not None:
                fechas_lote = df_campo[col_fecha].sort_values().tolist()
                for i in range(1, len(fechas_lote), 2):
                    fig_emer.add_vrect(
                        x0=fechas_lote[i-1], x1=fechas_lote[i], 
                        fillcolor="rgba(148, 163, 184, 0.12)", 
                        layer="below", line_width=0
                    )
            
            fig_emer.add_trace(go.Scatter(x=df["Fecha"], y=df["EMERREL_LOG"], mode='lines', name='Tasa Diaria Sim. (Log)', line=dict(color='#166534', width=2.5), fill='tozeroy', fillcolor='rgba(22, 101, 52, 0.1)'))
            fig_emer.add_hline(y=umbral_er_log, line_dash="dash", line_color="orange", annotation_text=f"Umbral Alerta ({umbral_er})")

            if df_campo is not None:
                fig_emer.add_trace(go.Scatter(x=df_campo[col_fecha], y=df_campo['Campo_Normalizado_LOG'], mode='markers+lines', name='Recuentos de Campo Real (Log)', marker=dict(color='#dc2626', size=10, symbol='diamond'), line=dict(color='rgba(220, 38, 38, 0.4)', dash='dot')))

            if fecha_control:
                fig_emer.add_vline(x=fecha_control.timestamp() * 1000, line_dash="dot", line_color="red", line_width=3, annotation_text=f"Control ({dga_optimo}°Cd)", annotation_position="top left", annotation_font=dict(color="red", size=12))
                fig_emer.add_vrect(x0=fecha_control.timestamp() * 1000, x1=(fecha_control + timedelta(days=residualidad)).timestamp() * 1000, fillcolor="blue", opacity=0.1, layer="below", line_width=0, annotation_text=f"Protección ({residualidad}d)", annotation_position="top left")

                if fecha_limite:
                    fig_emer.add_vline(x=fecha_limite.timestamp() * 1000, line_dash="dot", line_color="orange", line_width=3, annotation_text=f"Límite ({dga_critico}°Cd)", annotation_position="top right", annotation_font=dict(color="orange", size=12))
                    fig_emer.add_vrect(
                        x0=fecha_control.timestamp() * 1000, x1=fecha_limite.timestamp() * 1000, 
                        fillcolor="rgba(255, 165, 0, 0.18)", layer="below", line_width=0,
                        annotation_text="Ventana de Aplicación Eficiente", annotation_position="top left"
                    )

            fig_emer.update_layout(title="Dinámica Fisiológica de Emergencia (Bloques según Intervalos Reales)", yaxis_title="Log10(Emergencia + 0.01)", height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_emer, width="stretch")

            if fecha_inicio_ventana:
                st.success(f"📅 **Inicio de Conteo Térmico:** {fecha_inicio_ventana.strftime('%d-%m-%Y')} (Detección biológica inicial)")
                if fecha_control: st.error(f"🎯 **MOMENTO ÓPTIMO DE TRATAMIENTO:** {fecha_control.strftime('%d-%m-%Y')}. Acumulación térmica de **{dga_optimo} °Cd** post-emergencia.")
            else:
                st.warning(f"⏳ Fuera de ventana activa o esperando tasa diaria >= {umbral_er}.")

        with col_gauge:
            max_axis = dga_critico * 1.2
            st.plotly_chart(go.Figure().add_trace(go.Indicator(mode="gauge+number", value=dga_hoy, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "<b>TT POST-EMERGENCIA (°Cd)</b>", 'font': {'size': 18}}, gauge={'axis': {'range': [None, max_axis]}, 'bar': {'color': "#1e293b", 'thickness': 0.3}, 'steps': [{'range': [0, dga_optimo], 'color': "#4ade80"}, {'range': [dga_optimo, dga_critico], 'color': "#facc15"}, {'range': [dga_critico, max_axis], 'color': "#f87171"}], 'threshold': {'line': {'color': "#2563eb", 'width': 6}, 'thickness': 0.8, 'value': dga_7dias}})).add_annotation(x=0.5, y=-0.1, text=f"{msg_estado}<br>Pronóstico +7d: <b>{dga_7dias:.1f} °Cd</b>", showarrow=False, font=dict(size=14, color="#1e3a8a"), align="center").update_layout(height=350, margin=dict(t=80, b=50, l=30, r=30)), width="stretch")

        if df_campo is not None and not df_sincronizado.empty:
            st.markdown("---")
            st.markdown("<p class='metric-header' style='margin-top:20px;'>📈 DISPERSIÓN DE AJUSTE Y RECTAS 1:1</p>", unsafe_allow_html=True)
            col_curva, col_disp = st.columns([2, 1])
            
            with col_curva:
                fig_acum = go.Figure()
                fig_acum.add_trace(go.Scatter(x=df_sincronizado['Fecha'], y=df_sincronizado['Campo_Acumulado'] * 100, mode='markers+lines', name='Real de Campo (%)', marker=dict(color='#dc2626', size=8, symbol='diamond'), line=dict(color='#dc2626', width=2)))
                fig_acum.add_trace(go.Scatter(x=df_sincronizado['Fecha'], y=df_sincronizado['Sim_Acumulado'] * 100, mode='lines', name='Simulado PREDWEEM (%)', line=dict(color='#166534', width=3, dash='dash')))
                st.plotly_chart(fig_acum.update_layout(title="Llenado Cinético (Curvas Acumuladas Puras)", xaxis_title="Calendario", yaxis_title="Emergencia Acumulada (%)", height=430, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)), width="stretch")

            with col_disp:
                tab_flujos, tab_acum = st.tabs(["1:1 Flujos", "1:1 Acumulado"])
                
                with tab_flujos:
                    fig_1to1 = go.Figure()
                    fig_1to1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='1:1', line=dict(color='gray', dash='dash')))
                    fig_1to1.add_trace(go.Scatter(
                        x=df_sincronizado['Campo_Relativo'], y=df_sincronizado['Sim_Relativo'], 
                        mode='markers', name='Eventos',
                        marker=dict(color='#2563eb', size=12, line=dict(width=1, color='DarkBlue')),
                        text=df_sincronizado['Fecha'].dt.strftime('%d-%m-%Y'),
                        hovertemplate="<b>Intervalo fin: %{text}</b><br>Obs: %{x:.3f}<br>Sim: %{y:.3f}<extra></extra>"
                    ))
                    st.plotly_chart(fig_1to1.update_layout(title="Ajuste de Flujos por Eventos Reales", xaxis_title="Observado Relativo", yaxis_title="Simulado Relativo", height=380, showlegend=False, margin=dict(t=40, b=0, l=0, r=0)), width="stretch")

                with tab_acum:
                    fig_1to1_ac = go.Figure()
                    fig_1to1_ac.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='1:1', line=dict(color='gray', dash='dash')))
                    fig_1to1_ac.add_trace(go.Scatter(
                        x=df_sincronizado['Campo_Acumulado'], y=df_sincronizado['Sim_Acumulado'],
                        mode='markers', name='Acumulado',
                        marker=dict(color='#dc2626', size=12, symbol='diamond', line=dict(width=1, color='DarkRed')),
                        text=df_sincronizado['Fecha'].dt.strftime('%d-%m-%Y'),
                        hovertemplate="<b>%{text}</b><br>Obs Acum: %{x:.3f}<br>Sim Acum: %{y:.3f}<extra></extra>"
                    ))
                    st.plotly_chart(fig_1to1_ac.update_layout(title=f"Ajuste Acumulado (R²: {r2_acum:.3f} | RMSE: {rmse_acum:.3f})", xaxis_title="Obs. Acumulada", yaxis_title="Sim. Acumulada", height=380, showlegend=False, margin=dict(t=40, b=0, l=0, r=0)), width="stretch")

    with tab2:
        st.header("💧 Dinámica Hídrica del Suelo (Lartigau)")
        fig_hidrico = go.Figure()
        fig_hidrico.add_trace(go.Bar(x=df["Fecha"], y=df["Prec"], name='Lluvia Diaria (mm)', marker_color='#93c5fd', opacity=0.7))
        fig_hidrico.add_trace(go.Scatter(x=df["Fecha"], y=df["W_superficial"], name='Agua en Suelo (0-10cm)', mode='lines', line=dict(color='#0284c7', width=3), fill='tozeroy', fillcolor='rgba(2, 132, 199, 0.2)'))
        fig_hidrico.add_hline(y=w_max_val, line_dash="dot", line_color="#334155", annotation_text=f"Capacidad Máx. Suelo ({w_max_val} mm)", annotation_position="top left")
        st.plotly_chart(fig_hidrico.update_layout(title="Precipitación vs. Retención Real de Humedad", xaxis_title="Fecha", yaxis_title="Milímetros (mm)", height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)), width="stretch")

    with tab3:
        st.header("🔍 Clasificación DTW (Lartigau)")
        df_obs = df[df["Fecha"] < pd.Timestamp("2026-05-01")].copy()
        if not df_obs.empty and df_obs["EMERREL"].sum() > 0:
            jd_corte = df_obs["Julian_days"].max()
            max_e = df_obs["EMERREL"].max() if df_obs["EMERREL"].max() > 0 else 1.0
            JD_COM = cluster_model["JD_common"]
            jd_grid = JD_COM[JD_COM <= jd_corte]
            obs_norm = np.interp(jd_grid, df_obs["Julian_days"], df_obs["EMERREL"] / max_e)
            dists = [dtw_distance(obs_norm, m[JD_COM <= jd_corte] / m[JD_COM <= jd_corte].max() if m[JD_COM <= jd_corte].max() > 0 else m[JD_COM <= jd_corte]) for m in cluster_model["curves_interp"]]
            pred = int(np.argmin(dists))
            cols = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}

            c1, c2 = st.columns([3, 1])
            with c1:
                fp = go.Figure()
                fp.add_trace(go.Scatter(x=JD_COM, y=cluster_model["curves_interp"][pred], name="Patrón Histórico", line=dict(dash='dash', color=cols.get(pred))))
                fp.add_trace(go.Scatter(x=jd_grid, y=obs_norm * cluster_model["curves_interp"][pred].max(), name="2026", line=dict(color='black', width=3)))
                st.plotly_chart(fp, width="stretch")
            with c2:
                nombres_patrones = {0: "🌾 Bimodal", 1: "🌱 Temprano", 2: "🍂 Tardío"}
                st.success(f"### {nombres_patrones.get(pred, 'Desconocido')}")
                st.metric("DTW Score", f"{min(dists):.2f}")
        else:
            st.info("Datos insuficientes para clasificación por Dinámica Temporal (DTW).")

    with tab4:
        st.subheader("🧪 Curvas de Respuesta Fisiológica")
        x_temps = np.linspace(0, 45, 200)

        fig_tt = go.Figure()
        fig_tt.add_trace(
            go.Scatter(
                x=x_temps,
                y=[
                    calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica)
                    for t in x_temps
                ],
                mode="lines",
                name="Tiempo térmico diario",
                line=dict(color="#2563eb", width=4),
                fill="tozeroy",
            )
        )
        fig_tt.update_layout(
            title="Respuesta de tiempo térmico",
            xaxis_title="Temperatura (°C)",
            yaxis_title="Grados-día efectivos",
            height=380,
        )
        st.plotly_chart(fig_tt, width="stretch")

        factor_normal_plot = factor_termoinhibicion_normal(
            x_temps,
            media=media_termoinhibicion,
            desvio=desvio_termoinhibicion,
        )
        fig_term = go.Figure()
        fig_term.add_trace(
            go.Scatter(
                x=x_temps,
                y=factor_normal_plot,
                mode="lines",
                name="Factor de aptitud térmica",
                line=dict(color="#b91c1c", width=4),
                fill="tozeroy",
            )
        )
        fig_term.add_vline(
            x=media_termoinhibicion,
            line_dash="dash",
            line_color="#475569",
            annotation_text=f"Media T50 = {media_termoinhibicion:.1f} °C",
        )
        fig_term.add_hline(
            y=factor_min_inicio_termico,
            line_dash="dot",
            line_color="#ea580c",
            annotation_text=(
                f"Habilitación = {factor_min_inicio_termico:.2f} "
                f"durante {persistencia_termica_dias} días"
            ),
        )
        fig_term.update_layout(
            title=(
                "Termoinhibición normal acumulada "
                f"(media={media_termoinhibicion:.1f} °C; "
                f"σ={desvio_termoinhibicion:.1f} °C)"
            ),
            xaxis_title="Temperatura media móvil de 10 días (°C)",
            yaxis_title="Factor de aptitud térmica (0–1)",
            yaxis=dict(range=[0, 1.02]),
            height=380,
        )
        st.plotly_chart(fig_term, width="stretch")

    # REPORTE EN EXCEL
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data_Diaria')
        if df_campo is not None and not df_sincronizado.empty:
            df_campo.to_excel(writer, index=False, sheet_name='Campo_Validacion')
            val_lag = lag_inicio_dias if lag_inicio_dias is not None else "N/A"
            pd.DataFrame({
                'Métrica de Validación': ['PEC (%)', 'Lag Control (días)', 'Lead Time Control (días)', 'Pearson (Flujos)', 'NSE (Flujos Reales Evento)', 'KGE (Flujos)', 'RMSE (Acumulado)', 'R2 (Acumulado)', 'CCC (Acumulado)', 'Desfase T50 Global (días)', 'F1-Score (Coincidencia)', 'Exactitud Global', 'Hits (Aciertos)', 'Misses (Omisiones)', 'Falsos Positivos', 'Correctos Negativos', 'Desfase Primer Flujo (días)'],
                'Valor': [pec, peak_lag, lead_time, pearson_r, nse_flujos, kge_flujos, rmse_acum, r2_acum, ccc_acum, desfase_t50, f1_score_coincidencia, exactitud_global, hits_val, misses_val, falsos_pos_val, correctos_neg_val, val_lag]
            }).to_excel(writer, sheet_name='Validacion_Estadistica', index=False)
        pd.DataFrame({
            'Configuracion': [
                'T_Base',
                'T_Optima',
                'T_Critica',
                'W_Max',
                'Ke',
                'Mod_Termico',
                'Media_Termoinhibicion_T50',
                'Desvio_Termoinhibicion_Sigma',
                'Factor_Min_Inicio_Termico',
                'Persistencia_Termica_Dias',
            ],
            'Valor': [
                t_base_val,
                t_opt_max,
                t_critica,
                w_max_val,
                ke_val,
                mod_termico,
                media_termoinhibicion,
                desvio_termoinhibicion,
                factor_min_inicio_termico,
                persistencia_termica_dias,
            ],
        }).to_excel(writer, sheet_name='Bio_Params', index=False)

    st.sidebar.download_button("📥 Descargar Reporte Lartigau", output.getvalue(), "PREDWEEM_Integral_Lartigau_vK4_9_18_OptTermica.xlsx")

else:
    st.info("👋 Bienvenido a PREDWEEM. Cargue los datos climáticos de Lartigau para comenzar.")
