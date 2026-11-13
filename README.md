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
   la ventana de control no se calcula: el inicio no se estima con el clima);
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

## Señal de lluvia (actualización de noviembre de 2026)

La señal de lluvia usa ahora el **modelo de pulsos ajustado con las 17 campañas de *Lolium multiflorum* de Bordenave**.
El tamaño de cada pulso depende del agua que entró al suelo, de cuán seco estaba y de la época del año (máximo para
lluvias de fines de mayo); la mitad de la respuesta emerge dentro de 114 °Cd de tiempo térmico de suelo y el 90 %
dentro de 252 °Cd. Con el nuevo núcleo se reajustaron los modelos logísticos, los percentiles de referencia y la
recalibración de flujos tardíos (`alerta_lolmu_params.json`). El balance de agua y la temperatura de suelo no cambian.

Validación con cada campaña de Bordenave predicha con el ajuste sin ella (incluido el núcleo de lluvia): AUC 0,77 solo
con clima, 0,83 con primeras plántulas y 0,87 con conteos. En las ocho localidades de validación, sin reajustar: 0,89,
0,92 y 0,93 (calendario solo, 0,89). En Lartigau: 0,76, 0,82 y 0,79 (antes 0,88, 0,94 y 0,88); las tablas de
validación de este sitio que figuran más arriba corresponden a la versión anterior.

## Aviso de flujos tardíos

Del 1 de mayo al 30 de junio la app muestra además la **probabilidad de flujos tardíos**: que en los próximos 7 días
emerja al menos 2 % de la campaña. El aviso se da desde 20 %; sin datos del lote, desde 25 % (`tardia.umbral_tramos`).
En esa época los grandes flujos (≥ 5 %) son raros, pero pocas plantas por semana importan para decidir la fecha de siembra.
Es la misma alerta recalibrada por nivel de información (`tardia` en `alerta_lolmu_params.json`). Aparece en un recuadro
debajo de la alerta, en el **semáforo semanal** del gráfico (fila «Flujos tardíos»: violeta fuerte = flujo tardío probable;
la fila «Grandes flujos» repite el nivel de cada semana) y en la descarga en Excel.

| Información | AUC Bordenave (17 campañas, 130 semanas de mayo–junio, 42 con ≥ 2 %) | Detectadas con el aviso | Semanas marcadas |
|---|---|---|---|
| Solo clima (aviso desde 25 %) | 0,57 | 67 % | 68 % |
| + primeras plántulas | 0,79 | 90 % | 62 % |
| + conteos semanales | 0,87 | 86 % | 51 % |

Cada campaña se predijo con el ajuste sin ella. Sin datos del lote el aviso casi no distingue semanas: conviene cargar la
fecha de primeras plántulas. En otras seis localidades (58 intervalos de mayo–junio, 10 con ≥ 2 % por semana) la AUC fue
0,85 con primeras plántulas y 0,89 con conteos, con 90 % de detección; allí la probabilidad tiende a ser alta (30 % contra
17 % observado).

**Del 1 de julio al 1 de octubre no se muestra probabilidad** (`tardia.recorrida`): en Bordenave (80 semanas, 11 con
flujo tardío) no distinguió las semanas con emergencia (AUC 0,41–0,50), y bajar el umbral solo marcaba más semanas al
azar. En su lugar, la app recomienda **recorrer los lotes cada 2 semanas**, sobre todo donde suele haber emergencia
invernal (sudoeste bonaerense), antes de definir la fecha de siembra o un control. En el semáforo esas semanas aparecen
en gris.

En las apps con dos o más campañas propias (Bordenave, Tres Arroyos, Balcarce, San Pedro y Pergamino), el recuadro de
flujos tardíos muestra además cuánto emergió **después de la fecha de emisión en cada campaña del sitio**. Si en esas
campañas, en mediana, emergió menos de 3 % de la campaña desde esa fecha, el aviso se presenta como «Flujo tardío posible
según el modelo; en <sitio> la emergencia suele cortarse antes»: la recalibración de tardíos se ajustó con Bordenave, donde
la emergencia se extiende hasta el invierno, y en sitios donde se corta antes avisa de más (en Balcarce, en mayo de 2014,
2025 y 2026, avisó 8 semanas y hubo 2 flujos tardíos). La probabilidad se sigue mostrando.

## Residual a la siembra (guía de riesgo)

Debajo de la alerta, un recuadro siempre visible muestra, para la **fecha de siembra prevista** (1/05–31/07; si no se
carga, el 1/06 como referencia), el riesgo de nacimientos en la **ventana de acción del residual: de 15 días antes a 45 días
después de la siembra**, el período de acción promedio de un herbicida residual. Tiene tres tarjetas: riesgo, nacimientos en
la ventana (mediana de las campañas, en plantas/m² y en % de la campaña) y recomendación. Se calcula con las 34 campañas de la región, como referencia, porque Lartigau tiene una sola campaña; se muestra también el valor de esa campaña (`residual` en `alerta_lolmu_params.json`). Las plantas/m² usan la densidad de cada campaña con conteos; las campañas que solo
tienen la curva relativa usan la densidad de referencia del sitio (3934 pl/m² por campaña, mediana de las
campañas con conteos, aplicada a la distribución regional):

| Siembra | Nacimientos de −15 a +45 días (mediana) | Riesgo para el residual | Campañas del sitio |
|---|---|---|---|
| 1/05 | 797 pl/m² (20,3 %) | Muy alto | 2026: 92 pl/m² |
| 15/05 | 404 pl/m² (10,3 %) | Muy alto | 2026: 78 pl/m² |
| 1/06 | 167 pl/m² (4,2 %) | Muy alto | 2026: 68 pl/m² |
| 15/06 | 126 pl/m² (3,2 %) | Muy alto | 2026: 280 pl/m² |
| 1/07 | 74 pl/m² (1,9 %) | Muy alto | 2026: 298 pl/m² |

El nivel de riesgo surge de las plantas/m² que nacieron en la ventana en una campaña típica (mediana). Se usan plantas
y no el % de la campaña porque en sitios con mucha densidad una fracción chica son muchas plantas (en San Pedro, 0,8 % de
la campaña fueron 29–42 pl/m²):

| Riesgo | Nacimientos en la ventana | Recomendación |
|---|---|---|
| 🔴 Muy alto | ≥ 50 pl/m² | Usar residual |
| 🟠 Alto | 20–50 pl/m² | Residual recomendado |
| 🟡 Medio | 5–20 pl/m² | Según infestación: residual si el lote tiene alta infestación o antecedentes de raigrás |
| 🟢 Bajo | < 5 pl/m² | Residual solo en lotes muy infestados |

Las densidades son las registradas en los ensayos del sitio: en un lote con menos raigrás el riesgo es menor.

Debajo de las tarjetas se indica el valor de cada campaña del sitio y qué proporción de las campañas cayó en cada nivel. Es
un **riesgo histórico, no una predicción de la campaña**: ni la fecha de primeras plántulas ni el clima anticiparon cuánta
emergencia queda después de la siembra.

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
