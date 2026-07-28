# PREDWEEM — Lolium Lartigau 2026

Repositorio correspondiente a la implementación de **PREDWEEM** para la predicción de la emergencia y la dinámica fenológica de *Lolium multiflorum* en Lartigau, provincia de Buenos Aires, Argentina.

> **Propiedad intelectual**  
> Copyright © 2026 Guillermo R. Chantre / PREDWEEM.  
> Todos los derechos reservados.
>
> Este repositorio constituye software propietario. Su disponibilidad pública no concede autorización para utilizar, copiar, modificar, redistribuir, sublicenciar, realizar ingeniería inversa ni explotar comercialmente el código, los modelos, los parámetros, los pesos neuronales, la documentación o los datos incluidos.
>
> Consulte el aviso completo en [COPYRIGHT.md](COPYRIGHT.md).

## Finalidad

PREDWEEM es una herramienta de apoyo a la toma de decisiones agronómicas basada en la integración de datos meteorológicos, modelos predictivos y filtros ecofisiológicos para anticipar los flujos de emergencia de raigrás anual.

La implementación de este repositorio está orientada a **Lartigau** y debe utilizarse considerando el dominio geográfico, climático y agronómico para el cual fue configurada, así como su estado específico de validación.

## Fuentes meteorológicas operativas

La serie meteorológica distingue explícitamente el origen y el tipo de dato:

- **ERA5-Seamless**: reanálisis histórico desde el 1 de enero de 2026. Utiliza temperatura de ERA5-Land a 0,1° y precipitación de ERA5 a 0,25°, porque ERA5-Land no ofrece precipitación en la API histórica de Open-Meteo. Es información de grilla y no una observación puntual de estación.
- **ECMWF IFS histórico**: puente provisional para fechas vencidas todavía no disponibles en ERA5-Seamless o para eventuales huecos internos.
- **MeteoBahía XML — Coronel Falcón**: pronóstico determinístico utilizado exclusivamente desde la fecha actual en adelante.

El antiguo `meteo_daily.csv`, formado por pronósticos MeteoBahía que quedaban archivados al vencer, se conserva una sola vez en `data/meteo_falcon_pronosticos_archivados_2026.csv`. No se reutiliza como meteorología histórica.

El archivo operativo incluye `Fuente`, `TipoDato` y `CalidadDato` para evitar que un pronóstico vencido vuelva a confundirse con observación o reanálisis. Las columnas `TMAX`, `TMIN` y `Prec` continúan siendo compatibles con la ANN y el motor biofísico existentes.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Despliegue privado

La aplicación está preparada para cargar datos, imágenes y activos del modelo desde el checkout local. Antes de cambiar la visibilidad, autorice a Streamlit Community Cloud para acceder a repositorios privados de PREDWEEM. El procedimiento de verificación se describe en [PRIVATE_REPOSITORY.md](PRIVATE_REPOSITORY.md).

## Autoría

**PREDWEEM by Guillermo R. Chantre**
