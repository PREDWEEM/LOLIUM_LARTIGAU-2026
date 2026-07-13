# -*- coding: utf-8 -*-
"""PREDWEEM Lartigau vK4.9.22 — lag manual sin parches frágiles."""
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SOURCE_WRAPPER = BASE / "app_emergencia_vK4_9_20.py"

if not SOURCE_WRAPPER.exists():
    raise FileNotFoundError(
        f"Falta la versión optimizada previa: {SOURCE_WRAPPER.name}"
    )

wrapper = SOURCE_WRAPPER.read_text(encoding="utf-8")
wrapper = wrapper.replace(
    "'PREDWEEM Lartigau vK4.9.20: parámetros óptimos de CV temporal interna.'",
    "'PREDWEEM Lartigau vK4.9.22: parámetros óptimos y lag manual.'",
    1,
)
wrapper = wrapper.replace(
    "# 🌾 PREDWEEM INTEGRAL vK4.9.20 — LOLIUM LARTIGAU 2026",
    "# 🌾 PREDWEEM INTEGRAL vK4.9.22 — LOLIUM LARTIGAU 2026",
)

panel = '''    panel = """# --- PARÁMETROS OPTIMIZADOS Y SINCRONÍA TEMPORAL ---
st.sidebar.divider()
st.sidebar.markdown("## ⏱️ 4. Sincronía del Primer Pico")
lag_emergencia_dias = st.sidebar.slider(
    "Desfase temporal de emergencia (días)",
    min_value=0,
    max_value=60,
    value=LAG_EMERGENCIA_DIAS,
    step=1,
    help=(
        "Aumente este valor cuando el pico simulado aparezca antes que el observado. "
        "El valor inicial optimizado es +6 días."
    ),
)
with st.sidebar.expander("🧬 Parámetros óptimos Lartigau", expanded=False):
    st.caption(
        "La cobertura y el desfase temporal permanecen ajustables manualmente."
    )
    st.dataframe(
        pd.DataFrame({
            "Parámetro": list(PARAMETROS_OPTIMOS_LARTIGAU_20260713),
            "Valor óptimo": list(PARAMETROS_OPTIMOS_LARTIGAU_20260713.values()),
        }),
        width="stretch",
        hide_index=True,
    )
    st.write({
        "Cobertura actual (%)": int(cobertura_pct),
        "Ke derivado": float(ke_val),
        "Modulador térmico derivado": float(mod_termico),
        "Lag aplicado (días)": int(lag_emergencia_dias),
    })
# ---------------------------------------------------------
# 6. MOTOR DE CÁLCULO
"""
'''
wrapper, panel_count = re.subn(
    r"    panel = '''# --- PARÁMETROS OPTIMIZADOS ---.*?^'''",
    panel.rstrip(),
    wrapper,
    count=1,
    flags=re.S | re.M,
)
if panel_count != 1:
    raise RuntimeError(
        f"No se pudo insertar el control de sincronía: coincidencias={panel_count}"
    )

# Sustitución robusta: funciona aunque la llamada esté dividida en varias líneas.
lag_pattern = (
    r"(df\s*=\s*aplicar_lag_emergencia\(\s*df\s*,\s*lag_dias\s*=\s*)"
    r"(?:LAG_EMERGENCIA_DIAS|lag_emergencia_dias)"
    r"(\s*,\s*col\s*=\s*['\"]EMERREL['\"]\s*\))"
)
wrapper, lag_count = re.subn(
    lag_pattern,
    r"\1lag_emergencia_dias\2",
    wrapper,
    count=1,
    flags=re.S,
)
if lag_count == 0:
    # No bloquea el arranque si el archivo ya fue corregido previamente.
    wrapper = wrapper.replace(
        "lag_dias=LAG_EMERGENCIA_DIAS",
        "lag_dias=lag_emergencia_dias",
        1,
    )

exec(
    compile(wrapper, str(SOURCE_WRAPPER), "exec"),
    {
        "__name__": "__main__",
        "__file__": str(SOURCE_WRAPPER),
        "__package__": None,
    },
)
