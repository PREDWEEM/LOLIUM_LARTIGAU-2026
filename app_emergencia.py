# -*- coding: utf-8 -*-
"""
Punto de entrada de PREDWEEM Lartigau.

La aplicación científica original se conserva en ``app_emergencia_core.py``.
Este archivo la ejecuta sin alterar su lógica y agrega, al final de toda la
interfaz, una descarga Excel completa de los resultados generados.
"""
from pathlib import Path

_CORE_APP = Path(__file__).with_name("app_emergencia_core.py")
exec(
    compile(_CORE_APP.read_text(encoding="utf-8"), str(_CORE_APP), "exec"),
    globals(),
)


def _fecha_reporte(valor):
    """Convierte una fecha a texto legible y admite valores ausentes."""
    if valor is None or pd.isna(valor):
        return ""
    return pd.Timestamp(valor).strftime("%d/%m/%Y")


def _escribir_hoja(writer, dataframe, nombre):
    """Escribe una hoja y aplica un formato básico de lectura."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return

    dataframe.to_excel(writer, sheet_name=nombre, index=False)
    hoja = writer.sheets[nombre]
    hoja.freeze_panes(1, 0)
    hoja.autofilter(
        0,
        0,
        len(dataframe),
        max(0, len(dataframe.columns) - 1),
    )
    hoja.set_column(0, max(0, len(dataframe.columns) - 1), 18)


# El botón final se muestra únicamente si el motor completó la simulación.
if (
    "simulation" in globals()
    and isinstance(simulation, pd.DataFrame)
    and not simulation.empty
):
    reporte_excel_final = io.BytesIO()

    resumen_decision = pd.DataFrame(
        {
            "Indicador": [
                "Localidad",
                "Versión",
                "Fecha de generación",
                "Primer pico habilitado",
                "Fecha objetivo de control",
                "Fecha límite de la ventana",
                "TT acumulado actual (°Cd)",
                "TT pronosticado +7 días (°Cd)",
                "Cobertura de rastrojo (%)",
                "Wmax superficial (mm)",
                "Ke aplicado",
            ],
            "Valor": [
                "Lartigau (Buenos Aires)",
                globals().get("APP_VERSION", ""),
                pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
                _fecha_reporte(globals().get("first_peak_date")),
                _fecha_reporte(globals().get("control_date")),
                _fecha_reporte(globals().get("limit_date")),
                globals().get("thermal_time_today", 0.0),
                globals().get("thermal_time_7days", 0.0),
                globals().get("coverage_percent", ""),
                globals().get("w_max_value", ""),
                globals().get("ke_value", ""),
            ],
        }
    )

    metricas_base = globals().get("metrics", {}) or {}
    metricas_operativas = globals().get("operational", {}) or {}
    metricas_reporte = pd.DataFrame(
        {
            "Métrica": [
                "Pearson de flujos",
                "NSE de flujos",
                "KGE de flujos",
                "RMSE acumulado",
                "CCC acumulado",
                "R2 acumulado",
                "F1-Score de coincidencia",
                "Exactitud global",
                "Hits",
                "Misses",
                "Falsos positivos",
                "Correctos negativos",
                "PEC (%)",
                "Lag control vs. pico de campo (días)",
                "Lead time (días)",
                "Desfase T50 (días)",
                "Desfase del primer flujo (días)",
            ],
            "Valor": [
                metricas_base.get("Pearson_Flujos", 0.0),
                metricas_base.get("NSE_Flujos", 0.0),
                metricas_base.get("KGE_Flujos", 0.0),
                metricas_base.get("RMSE_Acumulado", 0.0),
                metricas_base.get("CCC_Acumulado", 0.0),
                metricas_base.get("R2_Acumulado", 0.0),
                metricas_base.get("F1_Score_Coincidencia", 0.0),
                metricas_base.get("Exactitud_Global", 0.0),
                metricas_base.get("Hits", 0),
                metricas_base.get("Misses", 0),
                metricas_base.get("Falsos_Positivos", 0),
                metricas_base.get("Correctos_Negativos", 0),
                metricas_operativas.get("PEC_Porcentaje"),
                metricas_operativas.get("Lag_Control_vs_Pico_Campo_Dias"),
                metricas_operativas.get("Lead_Time_Dias"),
                metricas_operativas.get("Desfase_T50_Dias"),
                metricas_operativas.get("Desfase_Primer_Flujo_Dias"),
            ],
        }
    )

    parametros_reporte = pd.DataFrame(
        {
            "Parámetro": [
                "Latitud",
                "Latencia fija (JD)",
                "Ventana termoinhibición (días)",
                "Umbral termoinhibición (°C)",
                "Ventana de lluvia (días)",
                "Choque hídrico (mm)",
                "Fin choque hídrico (JD)",
                "Umbral del primer pico",
                "Persistencia del primer pico (días)",
                "Cobertura de rastrojo (%)",
                "Wmax superficial (mm)",
                "Ke",
                "Modulador térmico diagnóstico",
                "Temperatura base (°C)",
                "Temperatura óptima (°C)",
                "Temperatura crítica (°C)",
                "TT objetivo de control (°Cd)",
                "TT límite de ventana (°Cd)",
                "Residualidad del herbicida (días)",
                "Umbral de alerta temprana",
            ],
            "Valor": [
                globals().get("LATITUD_LARTIGAU", ""),
                globals().get("LATENCIA_JD", ""),
                globals().get("VENTANA_TERMICA_DIAS", ""),
                globals().get("thermoinhibition_threshold", ""),
                globals().get("VENTANA_LLUVIA_DIAS", ""),
                globals().get("hydric_shock_threshold", ""),
                globals().get("FIN_CHOQUE_HIDRICO_JD", ""),
                globals().get("UMBRAL_PRIMER_PICO", ""),
                globals().get("PERSISTENCIA_PRIMER_PICO_DIAS", ""),
                globals().get("coverage_percent", ""),
                globals().get("w_max_value", ""),
                globals().get("ke_value", ""),
                globals().get("thermal_modulator", ""),
                globals().get("t_base", ""),
                globals().get("t_optimum", ""),
                globals().get("t_critical", ""),
                globals().get("tt_control", ""),
                globals().get("tt_limit", ""),
                globals().get("residual_days", ""),
                globals().get("alert_threshold", ""),
            ],
        }
    )

    with pd.ExcelWriter(
        reporte_excel_final,
        engine="xlsxwriter",
        datetime_format="dd/mm/yyyy",
        date_format="dd/mm/yyyy",
    ) as writer:
        _escribir_hoja(writer, simulation, "Resultados_Diarios")
        _escribir_hoja(
            writer,
            globals().get("synchronized"),
            "Validacion_Intervalos",
        )
        _escribir_hoja(
            writer,
            globals().get("field"),
            "Observaciones_Campo",
        )
        _escribir_hoja(writer, resumen_decision, "Resumen_Decision")
        _escribir_hoja(writer, metricas_reporte, "Metricas_Validacion")
        _escribir_hoja(writer, parametros_reporte, "Parametros_Modelo")
        _escribir_hoja(
            writer,
            globals().get("thermal_curve"),
            "Tiempo_Termico",
        )
        _escribir_hoja(
            writer,
            globals().get("optimizer_results"),
            "Optimizador_2D",
        )

    reporte_excel_final.seek(0)

    st.divider()
    st.subheader("📥 Descarga final de resultados")
    st.caption(
        "El archivo reúne la simulación diaria, validación Event-to-Event, "
        "observaciones de campo, métricas, parámetros, tiempo térmico y, "
        "cuando está disponible, el optimizador biofísico 2D."
    )
    st.download_button(
        label="📊 Descargar resultados completos en Excel",
        data=reporte_excel_final.getvalue(),
        file_name="PREDWEEM_Lartigau_Resultados_Completos.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
        key="descarga_excel_resultados_final_lartigau",
    )
