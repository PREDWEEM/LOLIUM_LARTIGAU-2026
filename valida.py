import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Calibrador Pro", layout="wide", page_icon="🌾")

# ==========================================
# 1. FUNCIONES DE MÉTRICAS (KGE & PBIAS)
# ==========================================
def calcular_metricas(obs, pred):
    """Calcula NSE, KGE y PBIAS para validación biológica."""
    # Evitar divisiones por cero
    if np.std(obs) == 0: return 0, 0, 0
    
    # NSE (Nash-Sutcliffe)
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    
    # PBIAS (Percent Bias)
    pbias = 100 * (np.sum(obs - pred) / np.sum(obs))
    
    # KGE (Kling-Gupta Efficiency)
    r = np.corrcoef(obs, pred)[0, 1]
    beta = np.mean(pred) / np.mean(obs)
    # Coeficiente de variación ratio (gamma)
    cv_obs = np.std(obs) / np.mean(obs) if np.mean(obs) != 0 else 1
    cv_pred = np.std(pred) / np.mean(pred) if np.mean(pred) != 0 else 1
    gamma = cv_pred / cv_obs
    
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    return nse, kge, pbias

# ==========================================
# 2. ARQUITECTURA DE LA RED NEURONAL
# ==========================================
class PREDWEEM_ANN:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, X_raw):
        X_norm = self.normalize(X_raw)
        z1 = X_norm @ self.IW + self.bIW.T  # Operación matricial optimizada
        a1 = np.tanh(z1)
        z2 = (a1 @ self.LW.T) + self.bLW
        emer = (np.tanh(z2) + 1) / 2
        return emer.flatten()

@st.cache_data
def ejecutar_prediccion_base(_model, data):
    """Caché para evitar re-calcular la ANN innecesariamente."""
    return _model.predict(data)

# ==========================================
# 3. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🌾 PREDWEEM: Calibración y Validación Avanzada")
st.markdown("""
Esta herramienta optimiza la respuesta del modelo **PREDWEEM** comparando datos de simulación neuronal 
con observaciones de campo mediante métricas de eficiencia hidrológica/biológica.
""")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Parámetros de Ajuste")

umbral_h = st.sidebar.slider(
    "Umbral Hídrico Acumulado (mm)", 
    0, 60, 20, 
    help="Lluvia acumulada en los últimos 21 días necesaria para disparar la emergencia."
)

ventana_dias = st.sidebar.slider(
    "Ventana de Tolerancia (Días)", 
    1, 14, 7, 
    help="Días alrededor del muestreo para capturar el pico máximo simulado."
)

st.subheader("📂 Carga de Archivos")
col_a, col_b = st.columns(2)
with col_a:
    f_meteo = st.file_uploader("Subir meteo_daily.csv", type=['csv'])
with col_b:
    f_valida = st.file_uploader("Subir VALIDA.xlsx", type=['xlsx'])

# ==========================================
# 4. MOTOR DE CÁLCULO
# ==========================================
if f_meteo and f_valida:
    try:
        # Carga de datos
        df_clima = pd.read_csv(f_meteo)
        df_clima.columns = df_clima.columns.str.strip()
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        
        df_campo = pd.read_excel(f_valida, engine='openpyxl')
        df_campo.columns = df_campo.columns.str.strip()
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Carga de Pesos con verificación
        files = ['IW.npy', 'LW.npy', 'bias_IW.npy', 'bias_out.npy']
        if all(os.path.exists(f) for f in files):
            iw, lw = np.load('IW.npy'), np.load('LW.npy')
            biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')
        else:
            st.error("❌ No se encontraron los archivos .npy de la red neuronal en el directorio.")
            st.stop()

        # 4.1 Predicción Base (ANN)
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        
        # Usamos la función con caché para que al mover el slider no se recalcule la ANN
        df_clima['EMER_RAW'] = ejecutar_prediccion_base(model, X_input)

        # 4.2 Filtro Hídrico Dinámico
        # Calculamos lluvia acumulada (ventana fija de 21 días)
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        
        # Aplicamos el umbral (esto sí cambia con el slider)
        df_clima['EMERREL'] = df_clima['EMER_RAW'].copy()
        df_clima.loc[df_clima['Prec_sum'] < umbral_h, 'EMERREL'] = 0.0
        df_clima.loc[df_clima['Julian_days'] <= 25, 'EMERREL'] = 0.0 # Estabilización inicial

        # 4.3 Validación con Ventana de Tolerancia
        df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()
        resultados_adj = []
        radio = ventana_dias // 2

        for _, row in df_campo.iterrows():
            f_obs, v_obs = row['FECHA'], row['ER_obs']
            mask = (df_clima['Fecha'] >= f_obs - pd.Timedelta(days=radio)) & \
                   (df_clima['Fecha'] <= f_obs + pd.Timedelta(days=radio))
            
            max_sim = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
            resultados_adj.append({'Fecha': f_obs, 'Obs': v_obs, 'Pred_Adj': max_sim})

        df_v = pd.DataFrame(resultados_adj)

        # 4.4 Cálculo de Métricas Finales
        y_o, y_p = df_v['Obs'].values, df_v['Pred_Adj'].values
        nse, kge, pbias = calcular_metricas(y_o, y_p)

        # ==========================================
        # 5. DASHBOARD Y GRÁFICOS
        # ==========================================
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        
        # NSE: Riguroso
        m1.metric("NSE (Eficiencia)", f"{nse:.2f}", help="Ideal = 1. Castiga fuerte los desfases.")
        # KGE: El balance que buscabas
        m2.metric("KGE (Kling-Gupta)", f"{kge:.2f}", delta=None, help="Balance entre correlación, sesgo y variabilidad.")
        # PBIAS: Magnitud total
        m3.metric("PBIAS (Sesgo %)", f"{pbias:.1f}%", delta_color="inverse", help="Mide si el modelo sub o sobreestima el volumen total.")
        # Umbral
        m4.metric("Umbral Hídrico", f"{umbral_h} mm")

        # Mensaje de interpretación
        if kge > 0.6:
            st.success("✨ **Excelente desempeño:** El modelo captura la dinámica y la magnitud de los picos.")
        elif kge > 0.4:
            st.info("👍 **Desempeño aceptable:** El modelo sigue la tendencia general.")
        else:
            st.warning("⚠️ **Ajuste necesario:** Intenta modificar el umbral hídrico o la ventana de días.")

        # Gráfico principal
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], label='Simulación PREDWEEM', color='#2e7d32', lw=1.5, alpha=0.8)
        ax.fill_between(df_clima['Fecha'], 0, df_clima['EMERREL'], color='#4caf50', alpha=0.2)
        
        # Puntos de observación
        ax.scatter(df_campo['FECHA'], df_campo['ER_obs'], color='#d32f2f', s=80, label='Observaciones Campo', zorder=5, edgecolor='white')
        
        ax.set_title("Dinámica de Emergencia: Simulado vs Observado", fontsize=14)
        ax.set_ylabel("Emergencia Relativa (0-1)")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(frameon=True)
        
        st.pyplot(fig)

        # Tabla de comparación rápida
        with st.expander("Ver tabla comparativa de picos"):
            st.dataframe(df_v.style.format({'Obs': '{:.3f}', 'Pred_Adj': '{:.3f}'}))

    except Exception as e:
        st.error(f"Ocurrió un error procesando los datos: {e}")
else:
    st.info("👋 **¡Bienvenido, Guillermo!** Por favor, carga los archivos .csv y .xlsx para iniciar la calibración de PREDWEEM.")
