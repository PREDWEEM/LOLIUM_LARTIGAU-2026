# -*- coding: utf-8 -*-
"""
PREDWEEM Lartigau vK4.9.25.

Corrección de estabilidad del primer pico frente a cambios de cobertura:
- restaura íntegramente el subsistema temporal validado en vK4.9.15;
- la cobertura afecta el balance hídrico mediante Ke;
- el modulador térmico se conserva sólo como diagnóstico de microclima;
- la ANN recibe TMAX y TMIN meteorológicas;
- el inicio se habilita con un día de EMERREL > 0.70;
- no se fija una fecha de pico y no se aplica lag.
"""
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "app_emergencia_vK4_9_15.py"

UMBRAL_PRIMER_PICO_ESTABLE = 0.50
PERSISTENCIA_PRIMER_PICO_DIAS = 1


def replace_once(text, pattern, replacement, label, flags=0):
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(
            f"No se pudo aplicar '{label}': coincidencias={count}"
        )
    return updated


def patch(source):
    source = source.replace(
        "# 🌾 PREDWEEM INTEGRAL vK4.9.15 — LOLIUM LARTIGAU 2026",
        "# 🌾 PREDWEEM INTEGRAL vK4.9.25 — LOLIUM LARTIGAU 2026",
        1,
    )
    source = source.replace(
        "# - PRIMER PICO VÁLIDO: La campaña se habilita únicamente cuando EMERREL > 0.70.",
        "# - PRIMER PICO VÁLIDO: se habilita con 1 día de EMERREL > 0.70.",
        1,
    )
    source = source.replace(
        "# - UX DINÁMICA: Sombreados de fondo basados en las fechas reales de muestreo.",
        "# - COBERTURA DESACOPLADA DE LA ANN: actúa sobre Ke; el modulador térmico es diagnóstico.\n"
        "# - SIN FECHA OBJETIVO NI LAG: el pico resulta exclusivamente de la simulación.\n"
        "# - UX DINÁMICA: Sombreados de fondo basados en las fechas reales de muestreo.",
        1,
    )

    constants = """UMBRAL_PRIMER_PICO = 0.50
PERSISTENCIA_PRIMER_PICO_DIAS = 1
"""
    source = replace_once(
        source,
        r"UMBRAL_PRIMER_PICO\s*=\s*0\.70\s*\n",
        constants,
        "constantes del primer pico",
    )

    stable_filter = """def aplicar_filtro_primer_pico(
    df,
    umbral=UMBRAL_PRIMER_PICO,
    persistencia=PERSISTENCIA_PRIMER_PICO_DIAS,
):
    # Criterio causal: habilita el primer día que supera el umbral.
    df = df.copy()
    df["EMERREL_ANTES_FILTRO_PRIMER_PICO"] = df["EMERREL"].copy()

    supera = df["EMERREL"].gt(float(umbral))
    confirmacion = (
        supera.astype(int)
        .rolling(window=int(persistencia), min_periods=int(persistencia))
        .sum()
        .ge(int(persistencia))
    )
    candidatos = np.flatnonzero(confirmacion.to_numpy())

    if candidatos.size:
        pos_confirmacion = int(candidatos[0])
        pos_inicio = max(0, pos_confirmacion - int(persistencia) + 1)
        idx_primer_pico = df.index[pos_inicio]
        df["Primer_Pico_Habilitado"] = df.index >= idx_primer_pico
        df.loc[df.index < idx_primer_pico, "EMERREL"] = 0.0
    else:
        idx_primer_pico = None
        df["Primer_Pico_Habilitado"] = False
        df["EMERREL"] = 0.0

    df["Supera_Umbral_Primer_Pico"] = supera
    df["Persistencia_Primer_Pico_Dias"] = int(persistencia)
    return df, idx_primer_pico
"""
    source = replace_once(
        source,
        r"def aplicar_filtro_primer_pico\(df, umbral=UMBRAL_PRIMER_PICO\):.*?"
        r"\n    return df, idx_primer_pico\n",
        stable_filter,
        "filtro estable del primer pico",
        re.S,
    )

    old_ann = 'X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)'
    new_ann = 'X = df[["Julian_days", "TMAX", "TMIN", "Prec"]].to_numpy(float)'
    occurrences = source.count(old_ann)
    if occurrences != 2:
        raise RuntimeError(
            "No se pudieron desacoplar las entradas térmicas de la ANN: "
            f"coincidencias={occurrences}"
        )
    source = source.replace(old_ann, new_ann)

    source = source.replace(
        "# Simulación Térmica",
        "# Microclima térmico diagnóstico; no modifica las entradas de la ANN",
        1,
    )
    source = source.replace(
        '<span style="color: #b91c1c; font-weight: bold; font-size: 1.05rem;">{mod_termico:.2f}</span>',
        '<span style="color: #b91c1c; font-weight: bold; font-size: 1.05rem;">{mod_termico:.2f} (diagnóstico)</span>',
        1,
    )
    source = source.replace(
        'help="0% = Suelo desnudo. 100% = Cobertura total (Lartigau Calibración Óptima = 70%)."',
        'help="La cobertura modifica Ke y el balance hídrico. No altera directamente las entradas térmicas de la ANN."',
        1,
    )

    source = replace_once(
        source,
        r'st\.sidebar\.info\(\n'
        r'\s*f"El inicio de la campaña se habilita únicamente cuando "\n'
        r'\s*f"EMERREL > \{UMBRAL_PRIMER_PICO:\.2f\}\."\n'
        r'\s*\)',
        """st.sidebar.info(
    f"El inicio se habilita con 1 día de EMERREL > "
    f"{UMBRAL_PRIMER_PICO:.2f}. "
    "La fecha surge de la simulación y no se aplica lag."
)""",
        "mensaje del criterio de pico",
    )

    source = source.replace(
        "# La campaña comienza en el primer valor estrictamente superior a 0.70.",
        "# La campaña comienza en el primer día con EMERREL > 0.70.",
        1,
    )
    source = source.replace(
        'f"Pico validado > {UMBRAL_PRIMER_PICO:.2f} "',
        'f"Pico habilitado (1 día) > {UMBRAL_PRIMER_PICO:.2f} "',
        1,
    )

    return source


if not SOURCE.exists():
    raise FileNotFoundError(
        f"Falta el modelo base validado: {SOURCE.name}"
    )

original = SOURCE.read_text(encoding="utf-8")
patched = patch(original)
exec(
    compile(patched, str(SOURCE), "exec"),
    {
        "__name__": "__main__",
        "__file__": str(SOURCE),
        "__package__": None,
    },
)
