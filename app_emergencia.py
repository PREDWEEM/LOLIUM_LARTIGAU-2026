# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.4 — LOLIUM TRES ARROYOS 2026
# Actualización: Restricción Hídrica Sigmoide + Relajación Dinámica
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pickle
import io
import os
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM INTEGRAL vK4.4", 
    layout="wide",
    page_icon="🌾"
)

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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. FUNCIONES DE APOYO
# ---------------------------------------------------------
def sigmoid_restriction(prec_sum, threshold=15, k=0.4):
    """Calcula un factor de penalización suave entre 0 y 1."""
    return 1 / (1 + np.exp(-k * (prec_sum - threshold)))

def calculate_tt_scalar(t_media, t_base, t_opt, t_crit):
    if t_media <= t_base or t_media >= t_crit:
        return 0
    elif t_base < t_media <= t_opt:
        return t_media - t_base
    else:
        return (t_opt - t_base) * (1 - (t_media - t_opt) / (t_crit - t_opt))

class ANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        z1 = Xn @ self.IW + self.bIW
        a1 = np.tanh(z1)
        z2 = (a1 @ self.LW.T).flatten() + self.bLW
        return (np.tanh(z2) + 1) / 2

# ---------------------------------------------------------
# 3. CARGA DE MODELO Y DATOS
# ---------------------------------------------------------
st.sidebar.header("Parámetros del Sistema")
archivo_meteo = st.sidebar.file_uploader("Cargar meteo_daily.csv", type=["csv"])

try:
    IW = np.load('IW.npy')
    bIW = np.load('bias_IW.npy')
    LW = np.load('LW.npy')
    bLW = np.load('bias_out.npy')
    modelo_ann = ANNModel(IW, bIW, LW, bLW)
except:
    st.error("No se encontraron los archivos de la red neuronal (.npy)")
    st.stop()

if archivo_meteo:
    df = pd.read_csv(archivo_meteo)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    
    # --- MOTOR DE CÁLCULO (vK4.4) ---
    # 1. Predicción Base de la Red
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df["EMERREL"] = modelo_ann.predict(X)
    
    # 2. Restricción Hídrica Sigmoide
    # Calculamos lluvia acumulada en 21 días
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = sigmoid_restriction(df["Prec_sum_21d"])
    df["EMERREL"] = df["EMERREL"] * df["Hydric_Factor"]
    
    # 3. Relajación Dinámica del Calendario
    # Si la lluvia > 50mm, se anula el bloqueo de seguridad; de lo contrario, se bloquea hasta el día 25.
    jd_thresholds = np.where(df["Prec_sum_21d"] > 50, 0, 25)
    df.loc[df["Julian_days"] <= jd_thresholds, "EMERREL"] = 0.0

    # 4. Cálculo de Tiempo Térmico (TT)
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, 5.0, 25.0, 35.0))
    df["TT_cum"] = df["DG"].cumsum()

    # --- INTERFAZ ---
    st.title("🌾 Panel de Emergencia PREDWEEM vK4.4")
    
    col1, col2, col3 = st.columns(3)
    max_val = df["EMERREL"].max()
    col1.metric("Máxima Emergencia Diaria", f"{max_val:.3f}")
    col2.metric("TT Acumulado", f"{df['TT_cum'].iloc[-1]:.1f} °Cd")
    col3.metric("Lluvia Total", f"{df['Prec'].sum():.1f} mm")

    # Gráfico de Emergencia
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Fecha'], y=df['EMERREL'], name="Tasa Diaria", line=dict(color='green', width=3)))
    fig.add_trace(go.Bar(x=df['Fecha'], y=df['Prec']/df['Prec'].max() if df['Prec'].max() > 0 else 0, 
                         name="Lluvia (Normalizada)", opacity=0.2, marker_color='blue'))
    
    fig.update_layout(title="Dinámica de Emergencia Relativa", xaxis_title="Fecha", yaxis_title="Tasa", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Botón de Descarga
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Resultados')
    st.sidebar.download_button("Descargar Reporte Excel", data=output.getvalue(), file_name="predweem_results.xlsx")
