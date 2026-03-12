import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Calibrador de Magnitud", layout="wide", page_icon="🌾")

# ==========================================
# 1. FUNCIONES DE CATEGORIZACIÓN Y MÉTRICAS
# ==========================================
def categorizar_emergencia(valor):
    """Categoriza la magnitud según los umbrales definidos por Guillermo."""
    if valor >= 0.5:
        return "ALTA"
    elif 0.15 <= valor < 0.5:
        return "INTERMEDIA"
    else:
        return "BAJA/NULA"

def calcular_metricas_avanzadas(obs, pred):
    """Calcula el desempeño numérico y de clasificación."""
    # Evitar divisiones por cero
    if np.std(obs) == 0: return 0, 0, 0, 0, [], []
    
    # NSE (Nash-Sutcliffe)
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    
    # PBIAS (Percent Bias)
    pbias = 100 * (np.sum(obs - pred) / np.sum(obs))
    
    # KGE (Kling-Gupta Efficiency)
    r = np.corrcoef(obs, pred)[0, 1]
    beta = np.mean(pred) / np.mean(obs)
    cv_obs = np.std(obs) / np.mean(obs) if np.mean(obs) != 0 else 1
    cv_pred = np.std(pred) / np.mean(pred) if np.mean(pred) != 0 else 1
    gamma = cv_pred / cv_obs
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    # Exactitud por Categorías (Aciertos en Magnitud)
    cat_obs = [categorizar_emergencia(v) for v in obs]
    cat_pred = [categorizar_emergencia(v) for v in pred]
    aciertos = sum(1 for o, p in zip(cat_obs, cat_pred) if o == p)
    accuracy_cat = (aciertos / len(obs)) * 100
    
    return nse, kge, pbias, accuracy_cat, cat_obs, cat_pred

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
        z1 = X_norm @ self.IW + self.bIW.T
        a1 = np.tanh(z1)
        z2 = (a1 @ self.LW.T) + self.bLW
        emer = (np.tanh(z2) + 1) / 2
        return np.maximum(emer.flatten(), 0)

# ==========================================
# 3. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🌾 PREDWEEM: Análisis de Magnitud y Riesgo")
st.markdown("Calibración dinámica considerando categorías de intensidad y período de latencia inicial.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Parámetros de Calibración")
umbral_h = st.sidebar.slider("Umbral Hídrico (mm)", 0, 60, 20)
ventana_dias = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7)

st.subheader("📂 Carga de Datos")
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
        # 4.1 Procesamiento de Datos
        df_clima = pd.read_csv(f_meteo)
        df_clima.columns = df_clima.columns.str.strip()
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        
        df_campo = pd.read_excel(f_valida)
        df_campo.columns = df_campo.columns.str.strip()
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Carga de Pesos (deben estar en la carpeta del script)
        files = ['IW.npy', 'LW.npy', 'bias_IW.npy', 'bias_out.npy']
        if all(os.path.exists(f) for f in files):
            iw, lw = np.load('IW.npy'), np.load('LW.npy')
            biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')
        else:
            st.error("No se encontraron los archivos .npy en el directorio.")
            st.stop()

        # 4.2 Predicción de la Red Neuronal
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        df_clima['EMERREL'] = model.predict(X_input)

        # 4.3 APLICACIÓN DE RESTRICCIONES (UMBRALES)
        # 1. Filtro Hídrico (Lluvia acumulada 21 días)
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        df_clima.loc[df_clima['Prec_sum'] < umbral_h, 'EMERREL'] = 0.0
        
        # 2. RESTRICCIÓN DE 25 DÍAS INICIALES (Emergencia Cero)
        df_clima.loc[df_clima['Julian_days'] <= 25, 'EMERREL'] = 0.0

        # 4.4 Validación de Campo
        df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()
        resultados = []
        radio = ventana_dias // 2

        for _, row in df_campo.iterrows():
            mask = (df_clima['Fecha'] >= row['FECHA'] - pd.Timedelta(days=radio)) & \
                   (df_clima['Fecha'] <= row['FECHA'] + pd.Timedelta(days=radio))
            max_p = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
            resultados.append({'Fecha': row['FECHA'], 'Obs': row['ER_obs'], 'Pred': max_p})

        df_v = pd.DataFrame(resultados)

        # 4.5 Cálculo de Métricas
        nse, kge, pbias, acc_cat, c_obs, c_pred = calcular_metricas_avanzadas(df_v['Obs'].values, df_v['Pred'].values)

        # ==========================================
        # 5. DASHBOARD Y VISUALIZACIÓN
        # ==========================================
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KGE (Tendencia)", f"{kge:.2f}")
        c2.metric("PBIAS (Sesgo)", f"{pbias:.1f}%")
        c3.metric("Acierto Magnitud", f"{acc_cat:.1f}%")
        c4.metric("NSE (Eficiencia)", f"{nse:.2f}")

        # Gráfico principal con zonas de riesgo
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhspan(0.5, 1.0, color='red', alpha=0.08, label='Riesgo Alto (≥0.5)')
        ax.axhspan(0.15, 0.5, color='orange', alpha=0.08, label='Riesgo Intermedio (0.15-0.49)')
        ax.axhspan(0.0, 0.15, color='green', alpha=0.05, label='Riesgo Bajo (<0.15)')
        
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], color='#1b5e20', lw=2, label='Modelo PREDWEEM')
        ax.scatter(df_v['Fecha'], df_v['Obs'], color='black', s=100, label='Observaciones Campo', zorder=5)
        
        # Marcar la zona de los primeros 25 días
        ax.axvspan(df_clima['Fecha'].iloc[0], df_clima['Fecha'].iloc[0] + pd.Timedelta(days=25), 
                   color='gray', alpha=0.2, label='Período Latencia (0-25d)')

        ax.set_ylabel("Emergencia Relativa")
        ax.set_ylim(0, 1.05)
        ax.legend(loc='upper right', fontsize='small', ncol=2)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

        # Tabla Comparativa de Categorías
        with st.expander("🔍 Ver Detalle de Clasificación por Fecha"):
            df_v['Cat_Obs'] = c_obs
            df_v['Cat_Pred'] = c_pred
            
            def style_categories(val):
                if val == "ALTA": return 'background-color: #ffcdd2; color: #b71c1c'
                if val == "INTERMEDIA": return 'background-color: #fff9c4; color: #f57f17'
                return 'background-color: #c8e6c9; color: #2e7d32'

            st.dataframe(df_v.style.applymap(style_categories, subset=['Cat_Obs', 'Cat_Pred']))

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
