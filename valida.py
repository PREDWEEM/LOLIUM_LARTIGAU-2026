import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error

# 1. MOTOR DE PREDICCIÓN (Basado en app_emergencia.py)
class NeuralModel:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW, self.bIW, self.LW, self.bLW = IW, bIW, LW, bLW
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, Xreal):
        Xn = self.normalize(Xreal)
        emer = []
        for x in Xn:
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            z2 = np.dot(self.LW, a1) + self.bLW
            emer.append(np.tanh(z2))
        emer = (np.array(emer).flatten() + 1) / 2
        return np.diff(np.cumsum(emer), prepend=0)

# 2. CARGA DE DATOS Y EJECUCIÓN DEL MODELO
# Cargar pesos y clima
IW, LW = np.load('IW.npy'), np.load('LW.npy')
bIW, bLW = np.load('bias_IW.npy'), np.load('bias_out.npy')
df_meteo = pd.read_csv('meteo_daily.csv', parse_dates=['Fecha'])

# Ejecutar predicción
model = NeuralModel(IW, bIW, LW, bLW)
X = df_meteo[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
df_meteo['EMERREL'] = model.predict(X)

# Aplicar restricciones biológicas (Filtro del proyecto)
df_meteo['Prec_sum'] = df_meteo['Prec'].rolling(window=21, min_periods=1).sum()
df_meteo.loc[(df_meteo['Prec_sum'] < 20) | (df_meteo['Fecha'].dt.dayofyear <= 25), 'EMERREL'] = 0

# 3. PROCESAMIENTO DE VERDAD DE CAMPO (Ground Truth)
df_campo = pd.read_csv('VALIDA.xlsx - Hoja1.csv', parse_dates=['FECHA'])
# Normalizar densidad absoluta (PLM2) a relativa (0-1)
df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()

# Sincronizar series temporales
df_val = pd.merge(df_meteo[['Fecha', 'EMERREL']], 
                  df_campo[['FECHA', 'ER_obs']], 
                  left_on='Fecha', right_on='FECHA', how='inner')

# 4. CÁLCULO DE MÉTRICAS ESTADÍSTICAS
rmse = np.sqrt(mean_squared_error(df_val['ER_obs'], df_val['EMERREL']))
# Eficiencia de Nash-Sutcliffe (NSE)
nse = 1 - (np.sum((df_val['ER_obs'] - df_val['EMERREL'])**2) / 
           np.sum((df_val['ER_obs'] - df_val['ER_obs'].mean())**2))

# 5. DETECCIÓN DE PICOS (Frecuencia y Magnitud)
peaks_pred, _ = find_peaks(df_meteo['EMERREL'], height=0.15, distance=7)
peaks_obs, _ = find_peaks(df_val['ER_obs'], height=0.1)

# 6. VISUALIZACIÓN DE VALIDACIÓN
plt.figure(figsize=(12, 6))
plt.plot(df_meteo['Fecha'], df_meteo['EMERREL'], label='Predicción RNA (EMERREL)', color='green', alpha=0.7)
plt.scatter(df_campo['FECHA'], df_campo['ER_obs'], color='red', label='Observado (Normalizado)', zorder=5)



plt.title(f'Validación de Modelo: NSE = {nse:.2f} | RMSE = {rmse:.3f}')
plt.xlabel('Fecha')
plt.ylabel('Emergencia Relativa')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print(f"Validación Completada.\nNSE: {nse:.4f}\nRMSE: {rmse:.44f}")
