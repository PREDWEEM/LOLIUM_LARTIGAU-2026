import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="PREDWEEM - Validación Científica", layout="wide")

# ==========================================
# 1. ARQUITECTURA DE LA RED NEURONAL
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
        emer_list = []
        for x in X_norm:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = np.dot(self.LW, a1) + self.bLW
            emer_list.append((np.tanh(z2) + 1) / 2)
        emer_cum = np.cumsum(np.array(emer_list).flatten())
        return np.diff(emer_cum, prepend=0)

# ==========================================
# 2. INTERFAZ DE USUARIO (UI)
# ==========================================
st.title("🌾 PREDWEEM: Validador de Campo con Ajuste Temporal")
st.sidebar.header("🔬 Parámetros de Validación")

# Slider para la ventana de desfase semanal
ventana_dias = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7, 
                                  help="Busca el mejor ajuste del modelo en un rango de días cercano al muestreo.")

# Widgets de carga de archivos (Aquí se definen f_meteo y f_valida)
st.subheader("📂 Carga de Datos")
col_a, col_b = st.columns(2)
with col_a:
    f_meteo = st.file_uploader("Subir meteo_daily.csv", type=['csv'])
with col_b:
    f_valida = st.file_uploader("Subir VALIDA.xlsx", type=['xlsx'])

# ==========================================
# 3. LÓGICA DE PROCESAMIENTO
# ==========================================
if f_meteo and f_valida:
    try:
        # A. Carga y Limpieza (Evita KeyError y UnicodeErrors)
        df_clima = pd.read_csv(f_meteo)
        df_clima.columns = df_clima.columns.str.strip()
        df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])
        df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear
        
        df_campo = pd.read_excel(f_valida, engine='openpyxl')
        df_campo.columns = df_campo.columns.str.strip()
        df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

        # B. Carga de Pesos (Deben estar en el repo de GitHub)
        iw, lw = np.load('IW.npy'), np.load('LW.npy')
        biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')

        # C. Predicción del Modelo
        model = PREDWEEM_ANN(iw, biw, lw, blw)
        X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
        df_clima['EMERREL'] = np.maximum(model.predict(X_input), 0.0)

        # Filtros Biológicos (21 días de lluvia y día juliano > 25)
        df_clima['Prec_sum'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
        df_clima.loc[(df_clima['Prec_sum'] < 20) | (df_clima['Julian_days'] <= 25), 'EMERREL'] = 0.0

        # D. Validación con Ventana de Tolerancia (Ajuste por Desfase Semanal)
        df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()
        resultados_adj = []
        mitad = ventana_dias // 2

        for _, row in df_campo.iterrows():
            f_obs, v_obs = row['FECHA'], row['ER_obs']
            mask = (df_clima['Fecha'] >= f_obs - pd.Timedelta(days=mitad)) & \
                   (df_clima['Fecha'] <= f_obs + pd.Timedelta(days=mitad))
            
            # Buscamos el valor máximo del modelo en la ventana semanal
            max_p = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
            resultados_adj.append({'Fecha': f_obs, 'Obs': v_obs, 'Pred_Adj': max_p})

        df_v = pd.DataFrame(resultados_adj)

        # E. Métricas Científicas
        y_o, y_p = df_v['Obs'].values, df_v['Pred_Adj'].values
        rmse = np.sqrt(np.mean((y_o - y_p)**2))
        nse = 1 - (np.sum((y_o - y_p)**2) / np.sum((y_o - np.mean(y_o))**2))

        # ==========================================
        # 4. VISUALIZACIÓN DE RESULTADOS
        # ==========================================
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Precisión (RMSE Ajustado)", f"{rmse:.3f}")
        c2.metric("Eficiencia (Nash-Sutcliffe)", f"{nse:.3f}")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_clima['Fecha'], df_clima['EMERREL'], label='Predicción Diaria RNA', color='forestgreen', alpha=0.3)
        ax.scatter(df_campo['FECHA'], df_campo['ER_obs'], color='red', s=120, label='Campo (Dato Real)', zorder=5)
        ax.scatter(df_v['Fecha'], df_v['Pred_Adj'], color='blue', marker='x', s=100, label='Ajuste por Desfase')
        
        ax.set_title(f"Validación con Ventana de {ventana_dias} días", fontsize=14)
        ax.set_ylabel("Emergencia Relativa (0-1)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        st.write("📋 **Tabla de Sincronización:**")
        st.dataframe(df_v)

    except Exception as e:
        st.error(f"Error técnico detectado: {e}")
else:
    st.info("💡 Por favor, sube los archivos .csv y .xlsx para iniciar la validación.")
