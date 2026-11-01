# Campaña 2026 de Lartigau: emergencia y clima

Copia fija de los datos de la campaña 2026. Los usa la validación de la alerta de flushes; consulte la sección
«Cómo se validó esta alerta» de la app y el README principal. La actualización automática no modifica estos archivos.
En cambio, `meteo_daily.csv` se reemplaza con la campaña nueva el 1 de enero de cada año.

## Emergencia de *Lolium multiflorum*

| Archivo | Contenido |
|---|---|
| `emergencia_lartigau_2026_original.xlsx` | Planilla aportada (hoja «Hoja1»: FECHA, PLM2), sin cambios. |
| `emergencia_lartigau_2026.csv` | Los mismos datos con columnas derivadas. |

`Plantulas_nuevas_m2` son las plántulas nuevas por m² en el intervalo que termina en esa fecha; el 01/02/2026 es el cero inicial.
Hay 15 fechas entre el 01/02/2026 y el 30/08/2026, con un total de 3933.5 plántulas/m².
`Fraccion_del_intervalo_pct` y `Emergencia_acumulada_pct` se expresan sobre ese total. La localidad la asignó el autor; la planilla no la identifica.
La misma serie, sin el cero inicial, está en `VALIDA (1).xlsx` en la raíz del repositorio.

## Clima

`clima_lartigau_2026.csv` es una copia de `meteo_daily.csv` al cierre de la campaña: 274 días, del 01/01/2026 al 01/10/2026, con 427 mm de lluvia.
Son **pronósticos MeteoBahía archivados para Coronel Falcón, no observaciones de estación** (columnas `Fuente`, `TipoDato` y `CalidadDato`).
La serie archivada original es `data/meteo_falcon_pronosticos_archivados_2026.csv`.

## Huellas SHA-256

| Archivo | SHA-256 |
|---|---|
| `emergencia_lartigau_2026_original.xlsx` | `dd81b1ef1d26e5852e71b4b28a1f3653ebf2aeabe6baa743d58d23c4fc95ec3a` |
| `emergencia_lartigau_2026.csv` | `59d3c66bc45459e77a5a03cd0372f648c8c37a78ccdd2cfbf5277778943564d8` |
| `clima_lartigau_2026.csv` | `ab0847cc7dc735a939df7502fa24cec4463071b856210dc2bd89c6db36051cc5` |
