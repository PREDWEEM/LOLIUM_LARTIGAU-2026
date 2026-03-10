import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ... (Clase PREDWEEM_ANN y carga de datos igual al código anterior) ...

# --- SIDEBAR: NUEVO PARÁMETRO DE VALIDACIÓN ---
st.sidebar.header("🔬 Validación Científica")
ventana_dias = st.sidebar.slider("Ventana de Tolerancia (Días)", 1, 14, 7)

if f_meteo and f_valida:
    # ... (Cálculo de EMERREL igual al código anterior) ...

    # --- LÓGICA DE AJUSTE POR DESFASE (WINDOWING) ---
    resultados_ventana = []
    mitad_ventana = ventana_dias // 2

    for _, row in df_campo.iterrows():
        fecha_obs = row['FECHA']
        valor_obs = row['ER_obs']
        
        # Definir el rango de búsqueda (± días)
        inicio = fecha_obs - pd.Timedelta(days=mitad_ventana)
        fin = fecha_obs + pd.Timedelta(days=mitad_ventana)
        
        # Buscar el valor máximo del modelo en esa semana
        mask = (df_clima['Fecha'] >= inicio) & (df_clima['Fecha'] <= fin)
        max_pred_ventana = df_clima.loc[mask, 'EMERREL'].max() if not df_clima[mask].empty else 0
        
        resultados_ventana.append({
            'Fecha': fecha_obs,
            'Observado': valor_obs,
            'Predicho_Ajustado': max_pred_ventana
        })

    df_val_final = pd.DataFrame(resultados_ventana)

    # --- MÉTRICAS RE-CALCULADAS ---
    y_o = df_val_final['Observado'].values
    y_p = df_val_final['Predicho_Ajustado'].values
    rmse_adj = np.sqrt(np.mean((y_o - y_p)**2))
    nse_adj = 1 - (np.sum((y_o - y_p)**2) / np.sum((y_o - np.mean(y_o))**2))

    # --- VISUALIZACIÓN ---
    st.subheader(f"📊 Validación con Ventana de {ventana_dias} días")
    
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("RMSE Ajustado", f"{rmse_adj:.3f}")
    col_m2.metric("NSE Ajustado", f"{nse_adj:.3f}", delta=f"{nse_adj - nse_p:.3f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_clima['Fecha'], df_clima['EMERREL'], label='Simulación Diaria', color='forestgreen', alpha=0.4)
    ax.scatter(df_campo['FECHA'], df_campo['ER_obs'], color='red', s=100, label='Campo (Semanal)')
    
    # Dibujar los puntos de "Mejor Ajuste" que el algoritmo encontró
    ax.scatter(df_val_final['Fecha'], df_val_final['Predicho_Ajustado'], color='blue', marker='x', s=80, label='Ajuste por Desfase')

    ax.set_title(f"Sincronización de Picos (NSE: {nse_adj:.3f})")
    ax.legend()
    st.pyplot(fig)
