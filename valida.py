import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. ARQUITECTURA DE LA RED NEURONAL (RNA)
# ==========================================
class PREDWEEM_ANN:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        # Rangos de escalamiento del modelo original
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, X_raw):
        X_norm = self.normalize(X_raw)
        emer_list = []
        for x in X_norm:
            # Propagación: Tanh en capa oculta y salida
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = np.dot(self.LW, a1) + self.bLW
            emer_list.append((np.tanh(z2) + 1) / 2)
        
        # Tasa diaria (EMERREL)
        emer_cum = np.cumsum(np.array(emer_list).flatten())
        return np.diff(emer_cum, prepend=0)

# ==========================================
# 2. CARGA DE COMPONENTES Y DATOS
# ==========================================
# Cargar pesos neuronales
iw, lw = np.load('IW.npy'), np.load('LW.npy')
biw, blw = np.load('bias_IW.npy'), np.load('bias_out.npy')

# Cargar clima y calcular Julian_days (Evita KeyError)
df_meteo = pd.read_csv('meteo_daily.csv')
df_meteo['Fecha'] = pd.to_datetime(df_meteo['Fecha'])
df_meteo['Julian_days'] = df_meteo['Fecha'].dt.dayofyear

# CARGAR EXCEL DE CAMPO (VALIDA.xlsx)
# Nota: Requiere 'openpyxl' instalado
try:
    df_campo = pd.read_excel('VALIDA.xlsx', engine='openpyxl')
    df_campo.columns = df_campo.columns.str.strip() # Limpiar nombres de columnas
    df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])
except Exception as e:
    print(f"Error al cargar VALIDA.xlsx: {e}")
    exit()

# ==========================================
# 3. PREDICCIÓN Y RESTRICCIONES BIOLÓGICAS
# ==========================================
model = PREDWEEM_ANN(iw, biw, lw, blw)
X = df_meteo[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
df_meteo['EMERREL'] = np.maximum(model.predict(X), 0.0)

# Aplicar filtros de PREDWEEM
df_meteo['Prec_21d'] = df_meteo['Prec'].rolling(window=21, min_periods=1).sum()
df_meteo.loc[(df_meteo['Prec_21d'] < 20) | (df_meteo['Julian_days'] <= 25), 'EMERREL'] = 0.0

# ==========================================
# 4. VALIDACIÓN ESTADÍSTICA
# ==========================================
# Normalizar verdad de campo (PLM2) para comparar con EMERREL (0-1)
df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()

# Sincronizar series
df_val = pd.merge(df_meteo[['Fecha', 'EMERREL']], 
                  df_campo[['FECHA', 'ER_obs']], 
                  left_on='Fecha', right_on='FECHA', how='inner')

# Métricas Manuales (NSE y RMSE)
y_o, y_p = df_val['ER_obs'].values, df_val['EMERREL'].values
rmse = np.sqrt(np.mean((y_o - y_p)**2))
nse = 1 - (np.sum((y_o - y_p)**2) / np.sum((y_o - np.mean(y_o))**2))

# ==========================================
# 5. GENERACIÓN DEL GRÁFICO CIENTÍFICO
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df_meteo['Fecha'], df_meteo['EMERREL'], color='green', label='Predicción RNA', alpha=0.8)
plt.fill_between(df_meteo['Fecha'], 0, df_meteo['EMERREL'], color='green', alpha=0.1)
plt.scatter(df_campo['FECHA'], df_campo['ER_obs'], color='red', s=100, label='Campo (Obs)', edgecolors='black')



plt.title(f'Validación de Emergencia de Lolium\nRMSE: {rmse:.3f} | NSE: {nse:.3f}', fontsize=12)
plt.ylabel('Emergencia Relativa (0-1)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('validacion_PREDWEEM.png')
print(f"Validación Finalizada.\nRMSE: {rmse:.4f}\nNSE: {nse:.4f}")
