import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Validador Científico", layout="wide")

# --- CLASE DEL MODELO ---
class PREDWEEM_ANN:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, X_raw):
        X_norm = self.normalize(X_raw)
        emer_list = []
        for x in X_norm:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = np.dot(self.LW, a1) + self.bLW
            emer_list.append((np.tanh(z2) + 1) / 2)
        emer_cum = np.cumsum(np.array(emer_list).flatten())
        return np.diff(emer_cum, prepend=0)

# --- INTERFAZ DE USUARIO ---
st.title("🌾 PREDWEEM: Módulo de Validación de Campo")
st.markdown("""
Esta herramienta compara las predicciones de la Red Neuronal contra datos reales de emergencia 
observados en el lote. Selecciona los archivos necesarios a continuación.
""")

# --- SIDEBAR: PARÁMETROS ---
st.sidebar.header("⚙️ Configuración Biológica")
t_opt = st.sidebar.slider("Temperatura Óptima (°C)", 15.0, 25.0, 20.0)
umbral_lluvia = st.sidebar.number_input("Umbral Hídrico (mm)", value=20)

# --- CARGA DE ARCHIVOS ---
col1, col2 = st.columns(2)
with col1:
    f_meteo = st.file_uploader("1. Cargar meteo_daily.csv", type=['csv'])
with col2:
    f_valida = st.file_uploader("2. Cargar VALIDA.xlsx", type=['xlsx'])

# --- PROCESAMIENTO PRINCIPAL ---
if f_meteo and f_valida:
    try:
        # Cargar datos
        df_clima = pd.read_csv(f_meteo)
        df_clima.columns = df_clima.columns.str.strip()
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        
        df_campo = pd.read_excel(f_valida, engine='openpyxl')
        df_campo.columns = df_campo.columns.str.strip()
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # Cargar Pesos (Asegúrate de que estén en la misma carpeta en GitHub)
        iw, lw = np.load('IW.npy'), np.load('LW.npy')
        biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')

        # Ejecutar Modelo
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        df_clima['EMERREL'] = np.maximum(model.predict(X_input), 0.0)

        # Filtros biológicos
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        df_clima.loc[(df_clima['Prec_sum'] < umbral_lluvia) | (df_clima['Julian_days'] <= 25), 'EMERREL'] = 0.0

        # Sincronización para métricas
        df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()
        df_val = pd.merge(df_clima[['Fecha', 'EMERREL']], df_campo[['FECHA', 'ER_obs']], 
                          left_on='Fecha', right_on='FECHA', how='inner')

        # Cálculo de Métricas
        y_o, y_p = df_val['ER_obs'].values, df_val['EMERREL'].values
        rmse = np.sqrt(np.mean((y_o - y_p)**2))
        nse = 1 - (np.sum((y_o - y_p)**2) / np.sum((y_o - np.mean(y_o))**2))

        # --- VISUALIZACIÓN ---
        st.subheader("📊 Análisis Comparativo: Modelo vs Campo")
        
        m1, m2 = st.columns(2)
        m1.metric("Precisión (RMSE)", f"{rmse:.3f}")
        m2.metric("Eficiencia (Nash-Sutcliffe)", f"{nse:.3f}")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], label='Modelo RNA', color='#2e7d32', lw=2)
        ax.fill_between(df_clima['Fecha'], 0, df_clima['EMERREL'], color='#2e7d32', alpha=0.1)
        ax.scatter(df_campo['FECHA'], df_campo['ER_obs'], color='#d32f2f', s=100, label='Observado (Normalizado)', edgecolors='white')
        
        ax.set_ylabel("Emergencia Relativa")
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.legend()
        st.pyplot(fig)

        st.dataframe(df_val.rename(columns={'EMERREL': 'Predicho', 'ER_obs': 'Observado'}))

    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
else:
    st.info("Por favor, sube ambos archivos para iniciar la validación.")
