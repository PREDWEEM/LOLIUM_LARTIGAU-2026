
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Reporte y Exportación", layout="wide", page_icon="🌾")

# ==========================================
# 1. FUNCIONES DE LÓGICA Y MÉTRICAS
# ==========================================
def categorizar_emergencia(valor):
    if valor >= 0.5: return "ALTA"
    elif 0.15 <= valor < 0.5: return "INTERMEDIA"
    else: return "BAJA/NULA"

def calcular_metricas_avanzadas(obs, pred):
    if np.std(obs) == 0: return 0, 0, 0, 0, [], []
    nse = 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    pbias = 100 * (np.sum(obs - pred) / np.sum(obs))
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else 0
    kge = 1 - np.sqrt((r - 1)**2 + (np.mean(pred)/np.mean(obs) - 1)**2 + (np.std(pred)/np.std(obs) - 1)**2)
    cat_obs = [categorizar_emergencia(v) for v in obs]
    cat_pred = [categorizar_emergencia(v) for v in pred]
    accuracy_cat = (sum(1 for o, p in zip(cat_obs, cat_pred) if o == p) / len(obs)) * 100
    return nse, kge, pbias, accuracy_cat, cat_obs, cat_pred

# Función de color para la tabla (Matriz)
def estilo_matriz(val):
    if val == "ALTA": color = '#ffcdd2; color: #b71c1c; font-weight: bold'
    elif val == "INTERMEDIA": color = '#fff9c4; color: #f57f17; font-weight: bold'
    elif val == "BAJA/NULA": color = '#c8e6c9; color: #2e7d32; font-weight: bold'
    else: color = ''
    return f'background-color: {color}'

# ==========================================
# 2. MODELO NEURONAL (PREDWEEM_ANN)
# ==========================================
class PREDWEEM_ANN:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min, self.input_max = np.array([1, 0, -7, 0]), np.array([300, 41, 25.5, 84])
    def predict(self, X_raw):
        X_norm = 2 * (X_raw - self.input_min) / (self.input_max - self.input_min) - 1
        a1 = np.tanh(X_norm @ self.IW + self.bIW.T)
        return np.maximum(((np.tanh(a1 @ self.LW.T) + self.bLW) + 1) / 2, 0).flatten()

# ==========================================
# 3. INTERFAZ Y CÁLCULOS
# ==========================================
st.title("🌾 PREDWEEM: Generador de Reportes")

st.sidebar.header("⚙️ Ajustes")
umbral_h = st.sidebar.slider("Umbral Hídrico (mm)", 0, 60, 20)
ventana_dias = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7)

f_meteo = st.file_uploader("Meteo Daily", type=['csv'])
f_valida = st.file_uploader("Valida Excel", type=['xlsx'])

if f_meteo and f_valida:
    try:
        # Procesamiento
        df_clima = pd.read_csv(f_meteo)
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        df_campo = pd.read_excel(f_valida)
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Carga de ANN
        iw, lw = np.load('IW.npy'), np.load('LW.npy')
        biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        
        # Predicción y Filtros
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        df_clima['EMERREL'] = model.predict(X_input)
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        df_clima.loc[(df_clima['Prec_sum'] < umbral_h) | (df_clima['Julian_days'] <= 25), 'EMERREL'] = 0.0

        # Validación
        df_campo['ER_obs'] = df_campo.iloc[:, 1] / df_campo.iloc[:, 1].max() # Asumiendo 2da col es PLM2
        res = []
        for _, row in df_campo.iterrows():
            mask = (df_clima['Fecha'] >= row['FECHA'] - pd.Timedelta(days=ventana_dias//2)) & \
                   (df_clima['Fecha'] <= row['FECHA'] + pd.Timedelta(days=ventana_dias//2))
            max_p = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
            res.append({'Fecha': row['FECHA'], 'Obs': row['ER_obs'], 'Pred': max_p})
        
        df_v = pd.DataFrame(res)
        nse, kge, pbias, acc_cat, c_obs, c_pred = calcular_metricas_avanzadas(df_v['Obs'].values, df_v['Pred'].values)
        df_v['Cat_Obs'], df_v['Cat_Pred'] = c_obs, c_pred

        # --- DASHBOARD ---
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("KGE", f"{kge:.2f}")
        col2.metric("PBIAS", f"{pbias:.1f}%")
        col3.metric("Acierto Cat.", f"{acc_cat:.1f}%")
        col4.metric("NSE", f"{nse:.2f}")

        # --- GRÁFICO ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhspan(0.5, 1.0, color='red', alpha=0.07, label='Alto')
        ax.axhspan(0.15, 0.5, color='orange', alpha=0.07, label='Intermedio')
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], color='#2e7d32', label='Predicción')
        ax.scatter(df_v['Fecha'], df_v['Obs'], color='black', label='Campo')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False)
        plt.tight_layout()
        st.pyplot(fig)

        # --- BOTONES DE DESCARGA ---
        st.subheader("📥 Exportar Resultados")
        d_col1, d_col2 = st.columns(2)
        
        # 1. Descarga Gráfico
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=300)
        d_col1.download_button("🖼️ Descargar Gráfico (PNG)", img_buf.getvalue(), "predweem_plot.png", "image/png")

        # 2. Descarga Tabla Excel
        exc_buf = io.BytesIO()
        df_v.to_excel(exc_buf, index=False)
        d_col2.download_button("📊 Descargar Datos (Excel)", exc_buf.getvalue(), "predweem_data.xlsx", "application/vnd.ms-excel")

        # --- MATRIZ COLOREADA ---
        st.subheader("📋 Matriz de Validación de Magnitud")
        st.dataframe(df_v.style.applymap(estilo_matriz, subset=['Cat_Obs', 'Cat_Pred']).format({'Obs': '{:.2f}', 'Pred': '{:.2f}'}))

    except Exception as e:
        st.error(f"Error: {e}")
