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
    if valor >= 0.5:
        return "ALTA"
    elif 0.15 <= valor < 0.5:
        return "INTERMEDIA"
    else:
        return "BAJA/NULA"

def calcular_metricas_avanzadas(obs, pred):
    if np.std(obs) == 0: return 0, 0, 0, 0, [], []
    
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    pbias = 100 * (np.sum(obs - pred) / np.sum(obs))
    
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else 0
    beta = np.mean(pred) / np.mean(obs)
    cv_obs = np.std(obs) / np.mean(obs) if np.mean(obs) != 0 else 1
    cv_pred = np.std(pred) / np.mean(pred) if np.mean(pred) != 0 else 1
    gamma = cv_pred / cv_obs
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
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
# 3. UI Y MOTOR DE CÁLCULO
# ==========================================
st.title("🌾 PREDWEEM: Análisis de Magnitud y Riesgo")

st.sidebar.header("⚙️ Parámetros de Calibración")
umbral_h = st.sidebar.slider("Umbral Hídrico (mm)", 0, 60, 20)
ventana_dias = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7)

f_meteo = st.file_uploader("Subir meteo_daily.csv", type=['csv'])
f_valida = st.file_uploader("Subir VALIDA.xlsx", type=['xlsx'])

if f_meteo and f_valida:
    try:
        df_clima = pd.read_csv(f_meteo)
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        
        df_campo = pd.read_excel(f_valida)
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Pesos
        iw, lw = np.load('IW.npy'), np.load('LW.npy')
        biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')

        # Predicción y Filtros (Incluyendo latencia de 25 días)
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        df_clima['EMERREL'] = model.predict(X_input)
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        
        # APLICACIÓN DE UMBRALES
        df_clima.loc[df_clima['Prec_sum'] < umbral_h, 'EMERREL'] = 0.0
        df_clima.loc[df_clima['Julian_days'] <= 25, 'EMERREL'] = 0.0

        # Validación
        df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()
        resultados = []
        radio = ventana_dias // 2

        for _, row in df_campo.iterrows():
            mask = (df_clima['Fecha'] >= row['FECHA'] - pd.Timedelta(days=radio)) & \
                   (df_clima['Fecha'] <= row['FECHA'] + pd.Timedelta(days=radio))
            max_p = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
            resultados.append({'Fecha': row['FECHA'], 'Obs': row['ER_obs'], 'Pred': max_p})

        df_v = pd.DataFrame(resultados)
        nse, kge, pbias, acc_cat, c_obs, c_pred = calcular_metricas_avanzadas(df_v['Obs'].values, df_v['Pred'].values)

        # Dashboard de Métricas
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("KGE (Tendencia)", f"{kge:.2f}")
        c2.metric("PBIAS (Sesgo)", f"{pbias:.1f}%")
        c3.metric("Acierto Magnitud", f"{acc_cat:.1f}%")
        c4.metric("NSE (Eficiencia)", f"{nse:.2f}")

        # ==========================================
        # 4. GRÁFICO MEJORADO (SIN SOLAPAMIENTO)
        # ==========================================
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Zonas de Magnitud
        ax.axhspan(0.5, 1.0, color='red', alpha=0.07, label='Riesgo Alto (≥0.5)')
        ax.axhspan(0.15, 0.5, color='orange', alpha=0.07, label='Riesgo Intermedio (0.15-0.49)')
        ax.axhspan(0.0, 0.15, color='green', alpha=0.04, label='Riesgo Bajo (<0.15)')
        
        # Latencia inicial
        ax.axvspan(df_clima['Fecha'].iloc[0], df_clima['Fecha'].iloc[0] + pd.Timedelta(days=25), 
                   color='gray', alpha=0.15, label='Latencia (0-25d)')

        # Datos
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], color='#2e7d32', lw=2, label='Modelo PREDWEEM', zorder=3)
        ax.scatter(df_v['Fecha'], df_v['Obs'], color='black', s=70, label='Observaciones Campo', zorder=5)

        # Ajustes de Ejes
        ax.set_ylabel("Emergencia Relativa", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # SOLUCIÓN AL SOLAPAMIENTO: Leyenda fuera del gráfico (arriba)
        ax.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.15), # Mueve la leyenda arriba de la figura
            ncol=3,                     # Organiza en 3 columnas
            fontsize=9, 
            frameon=False
        )

        plt.tight_layout() # Ajusta los márgenes para que la leyenda no se corte
        st.pyplot(fig)

        # Tabla de clasificación
        with st.expander("🔍 Ver Tabla de Clasificación de Riesgo"):
            df_v['Cat_Obs'], df_v['Cat_Pred'] = c_obs, c_pred
            st.dataframe(df_v)

    except Exception as e:
        st.error(f"Error: {e}")
