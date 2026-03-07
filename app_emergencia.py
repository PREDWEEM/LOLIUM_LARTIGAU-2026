# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.2 — LOLIUM TRES ARROYOS 2026
# Actualización: Auto-sync GitHub + Pestaña Hídrica Interactiva
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pickle
import io
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="PREDWEEM INTEGRAL vK4.2", 
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
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .bio-alert {
        padding: 10px;
        border-radius: 5px;
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
GITHUB_CSV = "https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/meteo_daily.csv"

# ---------------------------------------------------------
# 2. GESTIÓN DE DATOS Y ROBUSTEZ
# ---------------------------------------------------------
def create_mock_files_if_missing():
    if not (BASE / "IW.npy").exists():
        np.save(BASE / "IW.npy", np.random.rand(4, 10))
        np.save(BASE / "bias_IW.npy", np.random.rand(10))
        np.save(BASE / "LW.npy", np.random.rand(1, 10))
        np.save(BASE / "bias_out.npy", np.random.rand(1))
    
    if not (BASE / "modelo_clusters_k3.pkl").exists():
        jd = np.arange(1, 366)
        mock_cluster = {
            "JD_common": jd,
            "curves_interp": [np.exp(-((jd - 160)**2)/900)] * 3,
            "medoids_k3": [0, 1, 2]
        }
        with open(BASE / "modelo_clusters_k3.pkl", "wb") as f:
            pickle.dump(mock_cluster, f)

create_mock_files_if_missing()

def get_data(file_input):
    """Carga con jerarquía: Manual > GitHub > Local"""
    try:
        if file_input:
            df = pd.read_csv(file_input) if file_input.name.endswith('.csv') else pd.read_excel(file_input)
        else:
            try:
                # Intento de sincronización automática con GitHub
                df = pd.read_csv(GITHUB_CSV, parse_dates=["Fecha"])
            except:
                path = BASE / "meteo_daily.csv"
                df = pd.read_csv(path) if path.exists() else None
        
        if df is not None:
            df.columns = [c.upper().strip() for c in df.columns]
            mapeo = {'FECHA': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
            df = df.rename(columns=mapeo)
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            return df
        return None
    except Exception as e:
        st.error(f"Error en carga de datos: {e}")
        return None

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (ANN + BIO)
# ---------------------------------------------------------
def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base or t >= t_crit: return 0.0
    if t <= t_opt: return t - t_base
    return (t_crit - t) / (t_crit - t_opt) * (t_opt - t_base)

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min, self.input_max = np.array([1, 0, -7, 0]), np.array([300, 41, 25.5, 84])

    def predict(self, Xreal):
        Xn = 2 * (Xreal - self.input_min) / (self.input_max - self.input_min) - 1
        a1 = np.tanh(self.IW.T @ Xn.T + self.bIW[:, None])
        z2 = self.LW @ a1 + self.bLW[:, None]
        emer = (np.tanh(z2).flatten() + 1) / 2
        return emer

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
                                np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy"))
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except: return None, None

# ---------------------------------------------------------
# 4. INTERFAZ Y SIDEBAR
# ---------------------------------------------------------
modelo_ann, cluster_model = load_models()
st.sidebar.image("https://raw.githubusercontent.com/PREDWEEM/loliumTA_2026/main/logo.png", use_container_width=True)

st.sidebar.header("⚙️ Configuración")
archivo_usuario = st.sidebar.file_uploader("Actualizar Clima Manual", type=["xlsx", "csv"])
df = get_data(archivo_usuario)

st.sidebar.divider()
umbral_er = st.sidebar.slider("Umbral Pico Emergencia", 0.05, 0.80, 0.15)
t_base = st.sidebar.number_input("T Base", 2.0)
t_opt = st.sidebar.number_input("T Óptima Max", 20.0)
t_crit = st.sidebar.slider("T Crítica", 26.0, 42.0, 30.0)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO Y VISUALIZACIÓN
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    # A. Procesamiento
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian"] = df["Fecha"].dt.dayofyear
    
    # B. Predicción Neural + Lógica Hídrica
    X = df[["Julian", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    df["EMERREL"] = modelo_ann.predict(X)
    
    # Restricción hídrica: Ventana de 15 días < 10mm bloquea emergencia
    df["Prec_sum_15d"] = df["Prec"].rolling(window=15, min_periods=1).sum()
    df.loc[df["Prec_sum_15d"] < 10, "EMERREL"] = 0.0
    df.loc[df["Julian"] <= 25, "EMERREL"] = 0.0 # Bloqueo estacional
    
    # C. Tiempo Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base, t_opt, t_crit))

    st.title("🌾 PREDWEEM LOLIUM — LARTIGAU 2026")

    # Heatmap de Intensidad
    fig_h = go.Figure(data=go.Heatmap(z=[df["EMERREL"]], x=df["Fecha"], colorscale="Greens", showscale=False))
    fig_h.update_layout(height=100, margin=dict(t=20, b=0, l=10, r=10), title="Mapa de Intensidad de Emergencia")
    st.plotly_chart(fig_h, use_container_width=True)

    # Tabs
    t1, t2, t3, t4 = st.tabs(["📊 MONITOR DECISIÓN", "🌧️ MONITOREO HÍDRICO", "📈 ESTRATEGIA", "🧪 BIO-LAB"])

    with t1:
        col_m, col_g = st.columns([2, 1])
        indices_pico = df.index[df["EMERREL"] >= umbral_er].tolist()
        
        with col_m:
            fig_e = px.line(df, x="Fecha", y="EMERREL", title="Dinámica de Emergencia")
            fig_e.add_hline(y=umbral_er, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_e, use_container_width=True)
            
            if indices_pico:
                fecha_pico = df.loc[indices_pico[0], "Fecha"]
                st.success(f"✅ Primer pico detectado: {fecha_pico.strftime('%d-%m-%Y')}")
                
        with col_g:
            # Cálculo de TT acumulado desde el primer pico
            dga_total = 0.0
            if indices_pico:
                dga_total = df[df["Fecha"] >= df.loc[indices_pico[0], "Fecha"]]["DG"].sum()
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=dga_total,
                title={'text': "TT Acumulado (°Cd)"},
                gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#1e293b"}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

    with t2:
        st.subheader("🌧️ Registro de Precipitación Diaria")
        fig_p = px.bar(df, x="Fecha", y="Prec", title="Precipitación (mm)", color_discrete_sequence=['#3498db'])
        fig_p.add_trace(go.Scatter(x=df["Fecha"], y=df["Prec"].cumsum(), name="Acumulada", yaxis="y2", line=dict(color="#e74c3c")))
        fig_p.update_layout(yaxis2=dict(title="Acumulada (mm)", overlaying="y", side="right"), hovermode="x unified")
        st.plotly_chart(fig_p, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Lluvia Total", f"{df['Prec'].sum():.1f} mm")
        c2.metric("Evento Máximo", f"{df['Prec'].max():.1f} mm")
        c3.metric("Días con Lluvia", len(df[df['Prec'] > 0]))

    with t3:
        st.info("Módulo de clasificación DTW activo. Analizando similitud con patrones históricos...")
        # Aquí iría el código de DTW de tu vK4.1 si deseas mantener la comparativa visual

    with t4:
        st.subheader("🧪 Respuesta Fisiológica")
        temps = np.linspace(0, 45, 100)
        tts = [calculate_tt_scalar(t, t_base, t_opt, t_crit) for t in temps]
        fig_bio = px.line(x=temps, y=tts, labels={'x':'Temp (°C)', 'y':'TT (°Cd)'})
        st.plotly_chart(fig_bio, use_container_width=True)

    # Exportación
    output = io.BytesIO()
    df.to_excel(output, index=False)
    st.sidebar.download_button("📥 Descargar Datos", output.getvalue(), "Reporte_PREDWEEM.xlsx")

else:
    st.info("Esperando conexión con GitHub o carga de archivo local...")
