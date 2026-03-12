import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="PREDWEEM - Categorización de Magnitud", layout="wide", page_icon="🌾")

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
    # Métricas Continuas
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    pbias = 100 * (np.sum(obs - pred) / np.sum(obs))
    
    # KGE
    r = np.corrcoef(obs, pred)[0, 1] if np.std(obs) > 0 and np.std(pred) > 0 else 0
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else 1
    cv_obs = np.std(obs) / np.mean(obs) if np.mean(obs) != 0 else 1
    cv_pred = np.std(pred) / np.mean(pred) if np.mean(pred) != 0 else 1
    gamma = cv_pred / cv_obs
    kge = 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
    
    # Métricas de Categoría (Aciertos)
    cat_obs = [categorizar_emergencia(v) for v in obs]
    cat_pred = [categorizar_emergencia(v) for v in pred]
    aciertos = sum(1 for o, p in zip(cat_obs, cat_pred) if o == p)
    accuracy_cat = (aciertos / len(obs)) * 100 if len(obs) > 0 else 0
    
    return nse, kge, pbias, accuracy_cat, cat_obs, cat_pred

# ==========================================
# 2. MODELO NEURONAL
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
# 3. UI Y FLUJO PRINCIPAL
# ==========================================
st.title("🌾 PREDWEEM: Análisis de Magnitud y Riesgo")

st.sidebar.header("⚙️ Calibración")
umbral_h = st.sidebar.slider("Umbral Hídrico (mm)", 0, 60, 20)
ventana_dias = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7)

f_meteo = st.file_uploader("Meteo Daily", type=['csv'])
f_valida = st.file_uploader("Valida Excel", type=['xlsx'])

if f_meteo and f_valida:
    try:
        # Carga
        df_clima = pd.read_csv(f_meteo)
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        df_campo = pd.read_excel(f_valida)
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Pesos
        iw, lw = np.load('IW.npy'), np.load('LW.npy')
        biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')
        
        # Predicción
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        df_clima['EMERREL'] = model.predict(X_input)
        
        # Filtro hídrico
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        df_clima.loc[df_clima['Prec_sum'] < umbral_h, 'EMERREL'] = 0.0
        
        # Validación
        df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()
        res = []
        radio = ventana_dias // 2
        for _, row in df_campo.iterrows():
            mask = (df_clima['Fecha'] >= row['FECHA'] - pd.Timedelta(days=radio)) & \
                   (df_clima['Fecha'] <= row['FECHA'] + pd.Timedelta(days=radio))
            max_sim = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
            res.append({'Fecha': row['FECHA'], 'Obs': row['ER_obs'], 'Pred': max_sim})
        
        df_v = pd.DataFrame(res)
        
        # Métricas
        nse, kge, pbias, acc_cat, c_obs, c_pred = calcular_metricas_avanzadas(df_v['Obs'].values, df_v['Pred'].values)

        # Dashboard
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("KGE (Tendencia)", f"{kge:.2f}")
        m2.metric("PBIAS (Magnitud)", f"{pbias:.1f}%")
        m3.metric("Acierto Categoría", f"{acc_cat:.1f}%")
        m4.metric("Precisión Clasificación", "ALTA/INT" if acc_cat > 70 else "BAJA")

        # Comparativa Visual de Categorías
        st.subheader("🎯 Validación de Nivel de Emergencia")
        df_v['Cat_Obs'] = c_obs
        df_v['Cat_Pred'] = c_pred
        
        # Colores para la tabla
        def color_cat(val):
            color = '#ffcdd2' if val == 'ALTA' else '#fff9c4' if val == 'INTERMEDIA' else '#c8e6c9'
            return f'background-color: {color}'
        
        st.dataframe(df_v.style.applymap(color_cat, subset=['Cat_Obs', 'Cat_Pred']))

        # Gráfico con sombreado de magnitud
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhspan(0.5, 1.0, color='red', alpha=0.1, label='Riesgo Alto')
        ax.axhspan(0.15, 0.5, color='orange', alpha=0.1, label='Riesgo Intermedio')
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], color='green', label='Simulación')
        ax.scatter(df_v['Fecha'], df_v['Obs'], color='black', label='Campo', zorder=5)
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
