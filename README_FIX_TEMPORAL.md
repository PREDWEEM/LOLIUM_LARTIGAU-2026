# Corrección de sincronía temporal vK4.9.22

La versión `app_emergencia.py` de esta rama reemplaza el parche rígido del lag por una sustitución tolerante a espacios y saltos de línea.

- El control manual de lag permanece entre 0 y 60 días.
- El valor inicial continúa siendo +6 días.
- Si el archivo base ya fue corregido, la aplicación no se detiene.
- La versión previa se conserva como `app_emergencia_vK4_9_20.py`.

El diagnóstico automático detallado del intervalo de máximo observado se retiró de esta revisión para evitar nuevos fallos de arranque por sustituciones textuales encadenadas. La sincronización se ajusta mediante el deslizador y las métricas ya existentes del tablero.
