# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM vK4.4 — OPTIMIZADO: TRES ARROYOS / LARTIGAU
# Lógica: Sigmoide Hídrica + Relajación Dinámica de Calendario
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN Y ESTILO
# ---------------------------------------------------------
st.set_page_config(page_title="PREDWEEM vK4.4 - Tres Arroyos/Lartigau", layout="wide", page_icon="🌾")

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
    """Calcula el factor hídrico entre 0 y 1 para transiciones suaves."""
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
        # Capa oculta
        z1 = Xn @ self.IW + self.bIW
        a1 = np.tanh(z1)
        # Capa de salida
        z2 = (a1 @ self.LW.T).flatten() + self.bLW
        return (np.tanh(z2) + 1) / 2

# ---------------------------------------------------------
# 3. INTERFAZ Y CARGA DE DATOS
# ---------------------------------------------------------
st.title("🌾 Predicción de Emergencia - Tres Arroyos / Lartigau")
st.sidebar.header("Carga de Datos")

archivo_meteo = st.sidebar.file_uploader("Subir archivo meteorológico (.csv)", type=["csv"])

# Carga de Pesos (Asumiendo que los archivos .npy están en el mismo directorio)
try:
    IW = np.load('IW.npy')
    bIW = np.load('bias_IW.npy')
    LW = np.load('LW.npy')
    bLW = np.load('bias_out.npy')
    modelo = ANNModel(IW, bIW, LW, bLW)
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos de pesos (.npy). Asegúrate de que IW.npy, bias_IW.npy, LW.npy y bias_out.npy estén presentes.")
    st.stop()

if archivo_meteo:
    df = pd.read_csv(archivo_meteo)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)
    df['Julian_days'] = df['Fecha'].dt.dayofyear
    
    # --- PROCESAMIENTO vK4.4 ---
    X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df["EMER_BASE"] = modelo.predict(X)
    
    # Restricción Hídrica Sigmoide (Optimizado para r=0.94)
    df["Prec_sum_21d"] = df["Prec"].rolling(window=21, min_periods=1).sum()
    df["Hydric_Factor"] = sigmoid_restriction(df["Prec_sum_21d"])
    df["EMERREL"] = df["EMER_BASE"] * df["Hydric_Factor"]
    
    # Relajación Dinámica del Calendario
    # Si la lluvia > 50mm, se anula el bloqueo de principios de enero
    jd_threshold = np.where(df["Prec_sum_21d"] > 50, 0, 25)
    df.loc[df["Julian_days"] <= jd_threshold, "EMERREL"] = 0.0
    
    # --- VISUALIZACIÓN ---
    st.subheader("Visualización de Tasa de Emergencia")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Fecha'], y=df['EMERREL'], name="Tasa de Emergencia (vK4.4)", line=dict(color='green', width=3)))
    fig.add_trace(go.Bar(x=df['Fecha'], y=df['Prec']/df['Prec'].max() if df['Prec'].max() > 0 else 0, name="Lluvia (Escalada)", opacity=0.3, marker_color='blue'))
    
    fig.update_layout(
        xaxis_title="Fecha", 
        yaxis_title="Tasa Relativa / Probabilidad",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Métricas clave
    col1, col2, col3 = st.columns(3)
    max_emer = df['EMERREL'].max()
    fecha_max = df.loc[df['EMERREL'].idxmax(), 'Fecha'].strftime('%d-%m-%Y')
    
    col1.metric("Pico de Emergencia Máximo", f"{max_emer:.2f}")
    col2.metric("Fecha del Pico", fecha_max)
    col3.metric("Lluvia Total Periodo", f"{df['Prec'].sum():.1f} mm")
    
    # Exportación
    st.sidebar.header("Exportar")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predicciones')
    
    st.sidebar.download_button(
        label="Descargar Resultados (Excel)",
        data=output.getvalue(),
        file_name="prediccion_predweem_vk44.xlsx",
        mime="application/vnd.ms-excel"
    )
else:
    st.info("Por favor, sube un archivo .csv con columnas Fecha, TMAX, TMIN, Prec para comenzar.")
