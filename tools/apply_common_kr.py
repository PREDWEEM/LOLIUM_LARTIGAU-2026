from pathlib import Path

CORE = Path("app_emergencia_core.py")
REPORT = Path("app_emergencia.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: se esperaba 1 coincidencia y se encontraron {count}"
        )
    return text.replace(old, new, 1)


source = CORE.read_text(encoding="utf-8")

source = replace_once(
    source,
    "WMAX_PREDETERMINADO = 18.816\nCOBERTURA_PREDETERMINADA = 75\n",
    "WMAX_PREDETERMINADO = 18.816\nCOBERTURA_PREDETERMINADA = 75\n"
    "EXPONENTE_KR_PREDETERMINADO = 0.0\n",
    "constante Kr",
)

old_balance = '''def surface_water_balance(
    precipitation: np.ndarray,
    et0: np.ndarray,
    w_max: float,
    ke_soil: float,
) -> np.ndarray:
    precipitation = np.asarray(precipitation, dtype=float)
    et0 = np.asarray(et0, dtype=float)

    water = np.zeros(len(precipitation), dtype=float)
    if len(water) == 0:
        return water

    water[0] = float(w_max) / 2.0
    for index in range(1, len(water)):
        actual_evaporation = et0[index] * float(ke_soil)
        water[index] = np.clip(
            water[index - 1]
            + precipitation[index]
            - actual_evaporation,
            0.0,
            float(w_max),
        )
    return water
'''
new_balance = '''def surface_water_balance(
    precipitation: np.ndarray,
    et0: np.ndarray,
    w_max: float,
    ke_soil: float,
    kr_exponent: float = EXPONENTE_KR_PREDETERMINADO,
    return_kr: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Balance común con Kr configurable.

    kr_exponent=0 conserva exactamente ET0 × Ke constante.
    kr_exponent=1 reproduce la adaptación de Tres Arroyos.
    """
    precipitation = np.asarray(precipitation, dtype=float)
    et0 = np.asarray(et0, dtype=float)

    water = np.zeros(len(precipitation), dtype=float)
    kr_daily = np.ones(len(precipitation), dtype=float)
    if len(water) == 0:
        return (water, kr_daily) if return_kr else water
    if float(w_max) <= 0.0:
        raise ValueError("Wmax debe ser mayor que cero.")

    exponent = max(float(kr_exponent), 0.0)
    water[0] = float(w_max) / 2.0
    for index in range(1, len(water)):
        relative_previous_water = float(
            np.clip(water[index - 1] / float(w_max), 0.0, 1.0)
        )
        kr = (
            1.0
            if exponent == 0.0
            else relative_previous_water ** exponent
        )
        kr_daily[index] = kr
        actual_evaporation = et0[index] * float(ke_soil) * kr
        water[index] = np.clip(
            water[index - 1]
            + precipitation[index]
            - actual_evaporation,
            0.0,
            float(w_max),
        )
    return (water, kr_daily) if return_kr else water
'''
source = replace_once(source, old_balance, new_balance, "balance común")

source = replace_once(
    source,
    "    thermoinhibition_threshold: float = UMBRAL_TERMINHIBICION,\n"
    "    hydric_shock_threshold: float = UMBRAL_CHOQUE_HIDRICO_MM,\n"
    ") -> tuple[pd.DataFrame, int | None]:\n",
    "    thermoinhibition_threshold: float = UMBRAL_TERMINHIBICION,\n"
    "    hydric_shock_threshold: float = UMBRAL_CHOQUE_HIDRICO_MM,\n"
    "    kr_exponent: float = EXPONENTE_KR_PREDETERMINADO,\n"
    ") -> tuple[pd.DataFrame, int | None]:\n",
    "firma simulación",
)

source = replace_once(
    source,
    '    data["Ke_Suelo"] = ke_value\n'
    '    data["Modulador_Termico_Diagnostico"] = thermal_modulator\n',
    '    data["Ke_Suelo"] = ke_value\n'
    '    data["Exponente_Kr"] = float(kr_exponent)\n'
    '    data["Modulador_Termico_Diagnostico"] = thermal_modulator\n',
    "diagnóstico Kr",
)

source = replace_once(
    source,
    '''    data["W_superficial"] = surface_water_balance(
        data["Prec"].to_numpy(),
        data["ET0"].to_numpy(),
        float(w_max),
        ke_value,
    )
    relative_water = data["W_superficial"] / max(float(w_max), 1e-12)
''',
    '''    water, kr_daily = surface_water_balance(
        data["Prec"].to_numpy(),
        data["ET0"].to_numpy(),
        float(w_max),
        ke_value,
        kr_exponent=float(kr_exponent),
        return_kr=True,
    )
    data["W_superficial"] = water
    data["Kr_Diario"] = kr_daily
    relative_water = data["W_superficial"] / max(float(w_max), 1e-12)
''',
    "uso Kr en simulación",
)

source = replace_once(
    source,
    "    thermoinhibition_threshold: float,\n"
    "    hydric_shock_threshold: float,\n"
    ") -> pd.DataFrame:\n",
    "    thermoinhibition_threshold: float,\n"
    "    hydric_shock_threshold: float,\n"
    "    kr_exponent: float,\n"
    ") -> pd.DataFrame:\n",
    "firma optimizador",
)

source = replace_once(
    source,
    "                hydric_shock_threshold=float(hydric_shock_threshold),\n"
    "            )\n",
    "                hydric_shock_threshold=float(hydric_shock_threshold),\n"
    "                kr_exponent=float(kr_exponent),\n"
    "            )\n",
    "Kr en optimizador",
)

source = replace_once(
    source,
    '                    "W_Max_mm": float(wmax),\n'
    '                    "Fecha_Primer_Pico": first_date,\n',
    '                    "W_Max_mm": float(wmax),\n'
    '                    "Exponente_Kr": float(kr_exponent),\n'
    '                    "Fecha_Primer_Pico": first_date,\n',
    "salida optimizador",
)

old_wmax_ui = '''w_max_value = st.sidebar.number_input(
    "Capacidad superficial Wmax (mm)",
    min_value=5.0,
    max_value=60.0,
    value=WMAX_PREDETERMINADO,
    step=0.1,
    format="%.3f",
)
thermoinhibition_threshold = st.sidebar.number_input(
'''
new_wmax_ui = '''w_max_value = st.sidebar.number_input(
    "Capacidad superficial Wmax (mm)",
    min_value=5.0,
    max_value=60.0,
    value=WMAX_PREDETERMINADO,
    step=0.1,
    format="%.3f",
)
kr_exponent = st.sidebar.slider(
    "Exponente Kr (secado superficial)",
    min_value=0.0,
    max_value=2.0,
    value=EXPONENTE_KR_PREDETERMINADO,
    step=0.1,
    help=(
        "0 conserva el balance histórico de Lartigau (ET0×Ke). "
        "1 aplica la adaptación Kr de Tres Arroyos."
    ),
)
st.sidebar.caption(f"Kr diario = (W/Wmax)^{kr_exponent:.1f}.")
thermoinhibition_threshold = st.sidebar.number_input(
'''
source = replace_once(source, old_wmax_ui, new_wmax_ui, "interfaz Kr")

source = replace_once(
    source,
    "        hydric_shock_threshold=float(hydric_shock_threshold),\n"
    "    )\n"
    "except Exception as exc:\n",
    "        hydric_shock_threshold=float(hydric_shock_threshold),\n"
    "        kr_exponent=float(kr_exponent),\n"
    "    )\n"
    "except Exception as exc:\n",
    "Kr en motor principal",
)

source = replace_once(
    source,
    "                float(thermoinhibition_threshold),\n"
    "                float(hydric_shock_threshold),\n"
    "            )\n",
    "                float(thermoinhibition_threshold),\n"
    "                float(hydric_shock_threshold),\n"
    "                float(kr_exponent),\n"
    "            )\n",
    "Kr en llamada del optimizador",
)

compile(source, str(CORE), "exec")
if source.count("kr_exponent") < 10:
    raise RuntimeError("Kr no quedó integrado en todos los componentes.")
if "EXPONENTE_KR_PREDETERMINADO = 0.0" not in source:
    raise RuntimeError("Lartigau debe conservar Kr=0 como predeterminado.")
CORE.write_text(source, encoding="utf-8")

report = REPORT.read_text(encoding="utf-8")
report = replace_once(
    report,
    '                "Ke aplicado",\n            ],\n',
    '                "Ke aplicado",\n'
    '                "Exponente Kr configurable",\n'
    '            ],\n',
    "resumen reporte etiqueta",
)
report = replace_once(
    report,
    '                globals().get("ke_value", ""),\n            ],\n',
    '                globals().get("ke_value", ""),\n'
    '                globals().get("kr_exponent", ""),\n'
    '            ],\n',
    "resumen reporte valor",
)
report = replace_once(
    report,
    '                "Umbral de alerta temprana",\n            ],\n',
    '                "Umbral de alerta temprana",\n'
    '                "Exponente Kr",\n'
    '            ],\n',
    "parámetros reporte etiqueta",
)
report = replace_once(
    report,
    '                globals().get("alert_threshold", ""),\n            ],\n',
    '                globals().get("alert_threshold", ""),\n'
    '                globals().get("kr_exponent", ""),\n'
    '            ],\n',
    "parámetros reporte valor",
)
compile(report, str(REPORT), "exec")
REPORT.write_text(report, encoding="utf-8")

Path("migration_error.txt").unlink(missing_ok=True)
