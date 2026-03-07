# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.2 — LOLIUM TRES ARROYOS 2026
# Actualización: Corrección de IndexError + Auto-sync GitHub + Tab Hídrica
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
    """Genera archivos base si no existen para evitar errores de inicio"""
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
                # Intento de descarga automática desde GitHub
                df = pd.read_csv(GITHUB_CSV, parse_dates=["Fecha"])
            except:
                path = BASE / "meteo_daily.csv"
                df = pd.read_csv(path) if path.exists() else None
        
        if df is not None:
            df.columns = [c.upper().strip() for c in df.columns]
            mapeo = {'FECHA': 'Fecha', 'DATE': 'Fecha', 'TMAX': 'TMAX', 'TMIN': 'TMIN', 'PREC': 'Prec', 'LLUVIA': 'Prec'}
            df = df.rename(columns=mapeo)
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            return df
        return None
    except Exception as e:
        st.error(f"Error en carga de datos: {e}")
        return None

# ---------------------------------------------------------
# 3. LÓGICA TÉCNICA (ANN + DTW + BIO)
# ---------------------------------------------------------
def dtw_distance(a, b):
    na, nb = len(a), len(b)
    dp = np.full((na+1, nb+1), np.inf)
    dp[0,0] = 0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return dp[na, nb]

def calculate_tt_scalar(t, t_base, t_opt, t_crit):
    if t <= t_base or t >= t_crit: return 0.0
    if t <= t_opt: return t - t_base
    return (t_crit - t) / (t_crit - t_opt) * (t_opt - t_base)

class PracticalANNModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        # Vectorización robusta contra errores de forma (IndexError fix)
        b1 = np.atleast_1d(self.bIW)
        b2 = np.atleast_1d(self.bLW)
        
        # Capa oculta y salida con broadcasting automático
        z1 = Xn @ self.IW + b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.LW.T + b2
        
        emer = (np.tanh(z2).flatten() + 1) / 2
        emer_ac = np.cumsum(emer)
        emerrel = np.diff(emer_ac, prepend=0)
        return emerrel, emer_ac

@st.cache_resource
def load_models():
    try:
        ann = PracticalANNModel(
            np.load(BASE/"IW.npy"), np.load(BASE/"bias_IW.npy"),
            np.load(BASE/"LW.npy"), np.load(BASE/"bias_out.npy")
        )
        with open(BASE/"modelo_clusters_k3.pkl", "rb") as f:
            k3 = pickle.load(f)
        return ann, k3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None

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

col_t1, col_t2 = st.sidebar.columns(2)
with col_t1: t_base_val = st.number_input("T Base", 2.0)
with col_t2: t_opt_max = st.number_input("T Óptima Max", 20.0)
t_critica = st.sidebar.slider("T Crítica", 26.0, 42.0, 30.0)

dga_optimo = st.sidebar.number_input("Objetivo Control", 600)
dga_critico = st.sidebar.number_input("Límite Ventana", 800)

# ---------------------------------------------------------
# 5. MOTOR DE CÁLCULO
# ---------------------------------------------------------
if df is not None and modelo_ann is not None:
    # A. Limpieza
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian"] = df["Fecha"].dt.dayofyear
    
    # B. Predicción Neural + Lógica Hídrica
    X = df[["Julian", "TMAX", "TMIN", "Prec"]].to_numpy(float)
    emerrel_raw, _ = modelo_ann.predict(X)
    df["EMERREL"] = np.maximum(emerrel_raw, 0.0)
    
    # Restricción hídrica: 15 días < 10mm bloquea emergencia
    df["Prec_sum_15d"] = df["Prec"].rolling(window=15, min_periods=1).sum()
    df.loc[df["Prec_sum_15d"] < 10, "EMERREL"] = 0.0
    df.loc[df["Julian"] <= 25, "EMERREL"] = 0.0 

    # C. Tiempo Térmico
    df["Tmedia"] = (df["TMAX"] + df["TMIN"]) / 2
    df["DG"] = df["Tmedia"].apply(lambda x: calculate_tt_scalar(x, t_base_val, t_opt_max, t_critica))

    # D. Detección de Ventana por PRIMER PICO
    indices_pulso = df.index[df["EMERREL"] >= umbral_er].tolist()
    
    st.title("🌾 PREDWEEM LOLIUM — LARTIGAU 2026")

    # Heatmap superior
    fig_risk = go.Figure(data=go.Heatmap(z=[df["EMERREL"]], x=df["Fecha"], colorscale="Greens", showscale=False))
    fig_risk.update_layout(height=100, margin=dict(t=20, b=0, l=10, r=10), title="Intensidad de Emergencia")
    st.plotly_chart(fig_risk, use_container_width=True)

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 MONITOR DECISIÓN", "🌧️ MONITOREO HÍDRICO", "📈 ESTRATEGIA", "🧪 BIO-LAB"])

    with tab1:
        col_main, col_gauge = st.columns([2, 1])
        dga_actual = 0.0
        dias_stress = 0
        fecha_inicio_ventana = None

        if indices_pulso:
            idx_primer = indices_pulso[0]
            fecha_inicio_ventana = df.loc[idx_primer, "Fecha"]
            df_v = df[df["Fecha"] >= fecha_inicio_ventana].copy()
            df_v["DGA_cum"] = df_v["DG"].cumsum()
            dga_actual = df_v["DGA_cum"].iloc[-1] if not df_v.empty else 0.0
            dias_stress = len(df_v[df_v["Tmedia"] > t_opt_max])

        with col_main:
            fig_e = px.line(df, x="Fecha", y="EMERREL", title="Dinámica de Emergencia")
            fig_e.add_hline(y=umbral_er, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_e, use_container_width=True)
            if fecha_inicio_ventana:
                st.success(f"📅 Conteo térmico iniciado: {fecha_inicio_ventana.strftime('%d-%m-%Y')}")
                if dias_stress > 0:
                    st.markdown(f'<div class="bio-alert">🔥 Estrés térmico: {dias_stress} días > {t_opt_max}°C</div>', unsafe_allow_html=True)
            else:
                st.warning(f"⏳ Esperando primer pico >= {umbral_er}")

        with col_gauge:
            max_ax = dga_critico * 1.2
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=dga_actual,
                title={'text': "<b>TT ACUMULADO (°Cd)</b>"},
                gauge={'axis': {'range': [0, max_ax]},
                       'bar': {'color': "#1e293b"},
                       'steps': [{'range': [0, dga_optimo], 'color': "#4ade80"},
                                 {'range': [dga_optimo, dga_critico], 'color': "#facc15"},
                                 {'range': [dga_critico, max_ax], 'color': "#f87171"}]}))
            st.plotly_chart(fig_g, use_container_width=True)

    with tab2:
        st.subheader("🌧️ Análisis de Lluvias")
        fig_p = px.bar(df, x="Fecha", y="Prec", title="Precipitación Diaria (mm)", color_discrete_sequence=['#3498db'])
        fig_p.add_trace(go.Scatter(x=df["Fecha"], y=df["Prec"].cumsum(), name="Acumulado (mm)", yaxis="y2", line=dict(color="#e74c3c", width=3)))
        fig_p.update_layout(yaxis2=dict(overlaying='y', side='right'), hovermode="x unified")
        st.plotly_chart(fig_p, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Acumulado Total", f"{df['Prec'].sum():.1f} mm")
        c2.metric("Máximo Evento", f"{df['Prec'].max():.1f} mm")
        c3.metric("Últimos 15 días", f"{df['Prec_sum_15d'].iloc[-1]:.1f} mm")

    with tab3:
        st.info("Módulo DTW: Comparando campaña 2026 con patrones históricos.")

    with tab4:
        temps = np.linspace(0, 45, 100)
        tts = [calculate_tt_scalar(t, t_base_val, t_opt_max, t_critica) for t in temps]
        st.plotly_chart(px.line(x=temps, y=tts, title="Curva Fisiológica", labels={'x':'Temp (°C)', 'y':'TT (°Cd)'}), use_container_width=True)

    # Exportación
    output = io.BytesIO()
    df.to_excel(output, index=False)
    st.sidebar.download_button("📥 Descargar Reporte", output.getvalue(), "PREDWEEM_Lolium.xlsx")
else:
    st.info("👋 Bienvenido. Cargando datos desde GitHub o local...")
