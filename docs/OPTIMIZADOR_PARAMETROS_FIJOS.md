# Optimizador Lartigau con parámetros temporales fijos

## Objetivo

Optimizar únicamente los parámetros hídricos libres sin alterar el subsistema que determina el primer pico en `app_emergencia_vK4_9_15.py`.

## Parámetros excluidos de la búsqueda

- latencia: JD 45;
- ventana de termoinhibición: 5 días;
- umbral de termoinhibición: 24 °C;
- ventana de lluvia: 3 días;
- choque hídrico: 45 mm;
- fin del choque: JD 110;
- techo del choque: 1,0;
- umbral del primer pico: 0,70;
- persistencia: 1 día;
- lag: 0 días.

Estos valores son constantes y no aparecen como opciones del optimizador.

## Cobertura

La cobertura de rastrojo se ingresa manualmente. No se optimiza. Ke y el modulador térmico se calculan mediante las curvas del modelo operativo.

## Parámetros libres

- Wmax;
- humedad p50;
- pendiente hídrica;
- corte hídrico;
- recarga relativa.

## Validación

Se utiliza validación cruzada temporal interna por bloques contiguos. El score combina KGE, NSE, CCC, RMSE, F1 y una penalización cuando el inicio simulado cae fuera del primer intervalo observado con emergencia.

La fecha del primer pico se informa como resultado descriptivo. No se fija, no se desplaza y no constituye una variable optimizable.
