# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK4.4 — TRES ARROYOS / LARTIGAU
# Configuración: Carga automática de 'meteo_daily.csv'
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
import os
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK4.4 - Auto Load", layout="wide", page_icon="🌾")

st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    [data-testid="stSidebar"] { background-color: #dcfce7; border-right: 1px solid #bbf7d0; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. FUNCIONES DE CÁLCULO
# ---------------------------------------------------------
def sigmoid_restriction(prec_sum, threshold=15, k=0.4):
    return 1 / (1 + np.exp(-k * (prec_sum - threshold)))

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
# 3. MOTOR DE CARGA Y PROCESAMIENTO
# ---------------------------------------------------------
st.title("🌾 Predicción de Emergencia - Tres Arroyos / Lartigau")

# Ruta del archivo meteorológico fijo
METEO_FILE = "meteo_daily.csv"

# Intentar cargar pesos de la red
try:
    IW = np.load('IW.npy')
    bIW = np.load('bias_IW.npy')
    LW = np.load('LW.npy')
    bLW = np.load('bias_out.npy')
    modelo = ANNModel(IW, bIW, LW, bLW)
except Exception as e:
    st.error("Error al cargar archivos de pesos (.npy). Verifique su existencia.")
    st.stop()

# Verificar y cargar el archivo meteorológico
if os.path.exists(METEO_FILE):
    df = pd.read_csv(METEO_FILE)
    
    # Preprocesamiento básico
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)
    df['Julian_days'] = df['Fecha'].dt.dayofyear
    
    # --- EJECUCIÓN vK4.4 ---
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df["EMER_BASE"] = modelo.predict(X)
    
    # Restricción Hídrica Sigmoide
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = sigmoid_restriction(df["Prec_sum_21d"])
    df["EMERREL"] = df["EMER_BASE"] * df["Hydric_Factor"]
    
    # Relajación Dinámica del Calendario
    # Si Prec_sum_21d > 50mm, se ignora el bloqueo de seguridad (JD <= 25)
    jd_threshold = np.where(df["Prec_sum_21d"] > 50, 0, 25)
    df.loc[df["Julian_days"] <= jd_threshold, "EMERREL"] = 0.0
    
    # --- INTERFAZ ---
    st.success(f"Datos cargados exitosamente desde {METEO_FILE}")
    
    col1, col2, col3 = st.columns(3)
    max_emer = df['EMERREL'].max()
    fecha_max = df.loc[df['EMERREL'].idxmax(), 'Fecha'].strftime('%d-%m-%Y')
    col1.metric("Pico de Emergencia", f"{max_emer:.3f}")
    col2.metric("Fecha del Pico", fecha_max)
    col3.metric("Lluvia Total", f"{df['Prec'].sum():.1f} mm")

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Fecha'], y=df['EMERREL'], name="EMERREL (vK4.4)", line=dict(color='green', width=3)))
    fig.add_trace(go.Bar(x=df['Fecha'], y=df['Prec']/df['Prec'].max() if df['Prec'].max() > 0 else 0, 
                         name="Lluvia (Normalizada)", opacity=0.2, marker_color='blue'))
    
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Tasa de Emergencia", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Botón de descarga en la barra lateral
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button("Descargar Reporte Excel", data=output.getvalue(), file_name="reporte_emerrel.xlsx")

else:
    st.warning(f"No se encontró el archivo '{METEO_FILE}'. Asegúrese de que el archivo esté en la misma carpeta que el script.")
