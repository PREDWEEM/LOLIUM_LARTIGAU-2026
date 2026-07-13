# Optimizador ecofisiológico PREDWEEM — Lartigau

## Propósito

La página `pages/05_Optimizador_Ecofisiologico.py` optimiza los parámetros del motor de emergencia de `app_emergencia.py` usando un único set de observaciones de campo y validación cruzada temporal interna por bloques contiguos.

## Limitación científica

Los bloques provienen de la misma campaña/localidad. El resultado permite seleccionar parámetros provisionales y evaluar estabilidad temporal, pero no constituye validación independiente. La validación externa deberá realizarse con otra campaña, localidad o experimento.

## Correspondencia con el modelo operativo

El optimizador usa la latitud de Lartigau (`-38.6166`) y conserva las curvas de superficie del script principal:

- cobertura 0, 30, 70 y 100 %;
- Ke 0.85, 0.50, 0.25 y 0.10;
- modulador térmico 0.95, 0.90, 0.85 y 0.80.

Por esta razón se optimiza `cobertura_pct`, no Ke y modulador térmico de manera independiente.

## Parámetros

Se pueden optimizar Wmax, cobertura, forma y corte de la respuesta hídrica, recarga, latencia, ventana y umbral de termoinhibición, ventana y umbral de choque hídrico, fin y techo del choque, primer pico, persistencia y lag.

`lag_dias=0` reproduce el modelo operativo actual. Si el óptimo difiere de cero, deberá incorporarse el desplazamiento de señal en `app_emergencia.py` para reproducir exactamente la simulación seleccionada.

## Archivos

La página intenta cargar automáticamente:

- `meteo_daily.csv` o `meteo_daily.xlsx`;
- `LARTIGAU_campo.xlsx`, `LARTIGAU_campo.csv` o archivos de validación equivalentes.

También permite cargar ambos archivos manualmente.

## Métricas

La selección combina KGE y NSE de flujos, CCC y RMSE acumulados y F1 de coincidencia. El T50 global se informa en el ajuste descriptivo completo, pero no participa en el score de los bloques.
