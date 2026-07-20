# Preparación para repositorio privado

Este repositorio está configurado para ejecutarse desde un checkout privado.

## Antes de cambiar la visibilidad

1. Autorizar a Streamlit Community Cloud para acceder a los repositorios privados de PREDWEEM.
2. Confirmar que la aplicación utiliza la rama `main` y el archivo `app_emergencia.py`.
3. Comprobar que `meteo_daily.csv`, `logo.png` y los activos del modelo estén presentes.

## Después de cambiar la visibilidad

1. Ejecutar manualmente el workflow meteorológico existente.
2. Confirmar que `meteo_daily.csv` se actualice y se genere un nuevo commit.
3. Ejecutar `Verificar despliegue privado`.
4. Revisar la aplicación Streamlit y confirmar la carga de datos y modelos.

La visibilidad de la aplicación Streamlit puede mantenerse pública aunque el repositorio sea privado, siempre que Streamlit tenga autorización de acceso.
