import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINICIÓN DEL MODELO (Arquitectura RNA)
# ==========================================
class PREDWEEM_Model:
    def __init__(self, IW, bIW, LW, bLW):
        self.IW = IW   # Pesos entrada
        self.bIW = bIW # Bias entrada
        self.LW = LW   # Pesos salida
        self.bLW = bLW # Bias salida
        # Rangos de normalización del modelo original
        self.input_min = np.array([1, 0, -7, 0])
        self.input_max = np.array([300, 41, 25.5, 84])

    def normalize(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, X_raw):
        # Normalización de entradas
        X_norm = self.normalize(X_raw)
        
        # Propagación hacia adelante
        emer_list = []
        for x in X_norm:
            # Capa Oculta (Tanh)
            z1 = self.IW.T @ x + self.bIW
            a1 = np.tanh(z1)
            # Capa Salida (Tanh + Escalamiento)
            z2 = np.dot(self.LW, a1) + self.bLW
            val = (np.tanh(z2) + 1) / 2
            emer_list.append(val)
        
        # Cálculo de tasa diaria (Diferencial)
        emer_cum = np.cumsum(np.array(emer_list).flatten())
        emer_rel = np.diff(emer_cum, prepend=0)
        return np.maximum(emer_rel, 0.0)

# ==========================================
# 2. CARGA Y PROCESAMIENTO DE DATOS
# ==========================================
# Cargar archivos de pesos
iw = np.load('IW.npy')
lw = np.load('LW.npy')
biw = np.load('bias_IW.npy')
blw = np.load('bias_out.npy')

# Cargar clima y asegurar nombres de columnas
df_clima = pd.read_csv('meteo_daily.csv')
df_clima.columns = df_clima.columns.str.strip()
df_clima['Fecha'] = pd.to_datetime(df_clima['Fecha'])

# --- SOLUCIÓN AL KEYERROR: Crear Julian_days ---
df_clima['Julian_days'] = df_clima['Fecha'].dt.dayofyear

# Cargar datos de campo
df_campo = pd.read_csv('VALIDA.xlsx')
df_campo.columns = df_campo.columns.str.strip()
df_campo['FECHA'] = pd.to_datetime(df_campo['FECHA'])

# ==========================================
# 3. EJECUCIÓN DE LA PREDICCIÓN
# ==========================================
model = PREDWEEM_Model(iw, biw, lw, blw)

# Selección de features: [Julian_days, TMAX, TMIN, Prec]
X_input = df_clima[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)
df_clima['EMERREL'] = model.predict(X_input)

# --- APLICAR RESTRICCIONES BIOLÓGICAS ---
# 1. Ventana Hídrica: Suma de lluvia 21 días >= 20mm
df_clima['Prec_sum_21d'] = df_clima['Prec'].rolling(window=21, min_periods=1).sum()
df_clima.loc[df_clima['Prec_sum_21d'] < 20, 'EMERREL'] = 0.0

# 2. Restricción de fecha: No hay emergencia antes del día 25
df_clima.loc[df_clima['Julian_days'] <= 25, 'EMERREL'] = 0.0

# ==========================================
# 4. MÓDULO DE VALIDACIÓN (CÁLCULO MANUAL)
# ==========================================
# Normalizar campo: PLM2 (absoluto) -> ER_obs (relativo 0-1)
df_campo['ER_obs'] = df_campo['PLM2'] / df_campo['PLM2'].max()

# Sincronizar por fecha
df_val = pd.merge(
    df_clima[['Fecha', 'EMERREL']], 
    df_campo[['FECHA', 'ER_obs']], 
    left_on='Fecha', right_on='FECHA', how='inner'
)

# Métricas Manuales (Sin Sklearn)
y_obs = df_val['ER_obs'].values
y_pred = df_val['EMERREL'].values

if len(y_obs) > 0:
    rmse = np.sqrt(np.mean((y_obs - y_pred)**2))
    nse = 1 - (np.sum((y_obs - y_pred)**2) / np.sum((y_obs - np.mean(y_obs))**2))
else:
    rmse, nse = 0, 0

# ==========================================
# 5. VISUALIZACIÓN FINAL
# ==========================================
plt.figure(figsize=(12, 6))

# Serie predicha
plt.plot(df_clima['Fecha'], df_clima['EMERREL'], color='forestgreen', 
         label='Predicción Modelo (Relativa)', linewidth=2)
plt.fill_between(df_clima['Fecha'], 0, df_clima['EMERREL'], color='green', alpha=0.1)

# Datos de campo
plt.scatter(df_campo['FECHA'], df_campo['ER_obs'], color='red', s=80, 
            label='Verdad de Campo (Normalizada)', edgecolor='black', zorder=5)

plt.title(f'Validación Científica PREDWEEM\nRMSE: {rmse:.3f} | NSE: {nse:.3f}', fontsize=14)
plt.ylabel('Tasa de Emergencia (0 - 1)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar y mostrar
plt.savefig('resultado_validacion.png')
print(f"Validación terminada.\nRMSE: {rmse:.4f}\nNSE: {nse:.4f}")
print(df_val[['Fecha', 'EMERREL', 'ER_obs']])
