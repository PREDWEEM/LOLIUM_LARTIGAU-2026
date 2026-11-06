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

La serie meteorológica utiliza exclusivamente información procedente de MeteoBahía:

- **Histórico operativo**: `data/meteo_falcon_pronosticos_archivados_2026.csv`. Contiene los pronósticos de Coronel Falcón que fueron archivados durante 2026. Estos valores se utilizan como entrada histórica de PREDWEEM, pero **no son observaciones meteorológicas de estación**.
- **Hoy y fechas futuras**: XML vigente de MeteoBahía para Coronel Falcón, descargado desde `https://meteobahia.com.ar/scripts/forecast/for-cf.xml`.

En cada actualización, el tramo anterior a la fecha actual se etiqueta como `METEOBAHIA_XML_ARCHIVADO / Historico_pronostico`, mientras que el tramo desde hoy se etiqueta como `METEOBAHIA_XML_CORONEL_FALCON / Pronostico`.

ERA5, ERA5-Land, ERA5-Seamless y el puente ECMWF histórico fueron retirados del flujo operativo. El archivo `meteo_daily.csv` se reconstruye diariamente combinando el archivo MeteoBahía archivado hasta ayer con el pronóstico XML vigente desde hoy. El propio archivo histórico se amplía con cada nueva emisión para conservar los pronósticos que luego pasan a formar parte del tramo histórico.

El archivo operativo incluye `Fuente`, `TipoDato` y `CalidadDato` para evitar que los pronósticos archivados se confundan con observaciones. Las columnas `TMAX`, `TMIN` y `Prec` continúan siendo compatibles con la ANN y el motor biofísico existentes.

## Alerta de flushes

La página muestra la **alerta de flushes** de *Lolium multiflorum*: la probabilidad de que en los próximos 7 días emerja
≥ 5 % de la campaña, con dos niveles (*Atención* desde 15 %, *Alerta* desde 25 %), la emergencia acumulada esperada y la
ventana de control post-emergente (600–800 °Cd desde el inicio). El modelo (`alerta_lolmu.py`, `alerta_lolmu_params.json`,
`visualizacion_alerta_lolmu.py`) es el calibrado con 17 campañas de Bordenave y combina, con la información disponible a la
fecha de emisión:

1. el calendario medio de emergencia de LOLMU;
2. la respuesta a las lluvias ya registradas (balance de agua y tiempo térmico de suelo);
3. opcionalmente, la fecha en que se vieron las primeras plántulas del lote, que además ancla el reloj térmico (sin ese dato,
   se usa el primer pico del motor PREDWEEM como inicio estimado);
4. opcionalmente, conteos de plántulas (tabla o planilla Excel/CSV con una columna «Fecha» y una por marco).

**Validación en Lartigau.** Sin reajustar el modelo, con los conteos de Lartigau 2026 (14 intervalos entre el 01/02 y el
30/08, 3 con flush) y el clima de este repositorio: AUC 0,88 solo con clima y 0,94 con la fecha de primeras plántulas; los
tres flushes quedaron en *Alerta*, y también 4 de los 11 intervalos sin flush. Es una sola campaña; el clima son pronósticos
MeteoBahía archivados (no observaciones), y la alerta con conteos está calibrada con conteos semanales.

La barra lateral conserva solo los parámetros que usa la alerta (temperaturas base, óptima y crítica, y objetivos 600 y
800 °Cd); los del motor PREDWEEM quedan en sus valores por defecto.

## Campaña vigente y actualización automática

La campaña meteorológica va del **1 de enero al 1 de octubre** de cada año (`_campania_vigente()` en `update_meteo.py`;
las variables `CAMPANIA_INICIO` y `CAMPANIA_FIN` la fijan a mano). Después del 1 de octubre la serie queda cerrada; el
**1 de enero siguiente** la carga se reinicia sola con un archivo histórico nuevo
(`data/meteo_falcon_pronosticos_archivados_<año>.csv`). Durante los primeros 21 días se admite una serie sin histórico
o que no empiece el 1 de enero (si fallaron las primeras corridas). La app muestra la alerta cuando la campaña reúne al
menos 30 días de clima.

## Curva acumulada en tiempo térmico desde las primeras plántulas

Cuando se informa la fecha de las primeras plántulas (o hay un primer conteo con plántulas), la curva de emergencia
acumulada deja de seguir el calendario y sigue el **tiempo térmico acumulado desde esa fecha** (media y rango 10–90 % de
17 campañas de Bordenave, `curva_tt` en `alerta_lolmu_params.json`); más allá del clima disponible se suman temperaturas
medias históricas. Error medio de la emergencia acumulada en las fechas de conteo desde las primeras plántulas (puntos %):

| Localidad | Calendario por fecha | Tiempo térmico desde el inicio |
|---|---|---|
| Bordenave (17 campañas, dejando cada una afuera) | 12,9 | 11,3 |
| Lartigau 2026 | 9,9 | 5,8 |
| Olavarría 2026 | 13,7 | 10,7 |
| Pergamino 2024 y 2026 | 17,5 | 7,2 |
| San Pedro 2023, 2024 y 2026 | 19,5 | 14,4 |

Mejora sobre todo donde la fecha de inicio varía mucho entre años. En Tres Arroyos y Balcarce sigue el calendario propio
del sitio (error 6,1 y 6,9 puntos) y en Azul el calendario de Bordenave (5,3 contra 15,8 en su única campaña).

## Aviso de flujos tardíos

Del 1 de mayo al 1 de octubre la app muestra además la **probabilidad de flujos tardíos**: que en los próximos 7 días
emerja al menos 2 % de la campaña. El aviso se da desde 20 %; **del 1/07 al 1/10, con fecha de primeras plántulas o conteos,
desde 15 %** (`tardia.umbral_tramos`), porque en esos meses los flujos tardíos son pocos y con 20 % se perdía la mayoría.
En esa época los grandes flujos (≥ 5 %) son raros, pero pocas plantas por semana importan para decidir la fecha de siembra.
Es la misma alerta recalibrada por nivel de información (`tardia` en `alerta_lolmu_params.json`). Aparece en un recuadro
debajo de la alerta, en el **semáforo semanal** del gráfico (fila «Flujos tardíos»: violeta fuerte = flujo tardío probable;
la fila «Grandes flujos» repite el nivel de cada semana) y en la descarga en Excel.

| Información | AUC Bordenave (17 campañas, 210 semanas, 53 con ≥ 2 %) | Detectadas con el aviso | Semanas marcadas | Detectadas mayo–junio | Detectadas julio–septiembre |
|---|---|---|---|---|---|
| Solo clima | 0,66 | 68 % | 54 % | 76 % | 36 % |
| + primeras plántulas | 0,76 | 83 % | 52 % | 98 % | 27 % |
| + conteos semanales | 0,83 | 77 % | 42 % | 90 % | 27 % |

Cada campaña se predijo con el ajuste sin ella. Con 15 % de julio a septiembre se detecta 27 % de los flujos de esos meses (con 20 % eran 18 % con
primeras plántulas y 27 % con conteos), marcando 19–24 % de esas semanas; son solo 11 semanas con flujo tardío en 17 campañas, así que conviene
confirmar el aviso con conteos. En otras seis localidades (112 intervalos del 1/05 al 1/10, 12 con ≥ 2 % por semana) la AUC
fue 0,92 con primeras plántulas y 0,93 con conteos, con 100 % y 83 % de detección.

## Datos de la campaña 2026

`data/campania_2026/` guarda una copia fija de la emergencia de *Lolium* (planilla original y CSV con fracción y acumulada)
y del clima de la campaña 2026 (01/01–01/10). Esos son los datos con que se validó la alerta. La actualización automática no
los modifica; ver `data/campania_2026/README.md`.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Despliegue privado

La aplicación está preparada para cargar datos, imágenes y activos del modelo desde el checkout local. Antes de cambiar la visibilidad, autorice a Streamlit Community Cloud para acceder a repositorios privados de PREDWEEM. El procedimiento de verificación se describe en [PRIVATE_REPOSITORY.md](PRIVATE_REPOSITORY.md).

## Autoría

**PREDWEEM by Guillermo R. Chantre**
