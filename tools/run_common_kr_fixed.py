from pathlib import Path
import runpy

script_path = Path("tools/apply_common_kr.py")
script = script_path.read_text(encoding="utf-8")
old = '''    "                float(thermoinhibition_threshold),\\n"
    "                float(hydric_shock_threshold),\\n"
    "            )\\n",
    "                float(thermoinhibition_threshold),\\n"
    "                float(hydric_shock_threshold),\\n"
    "                float(kr_exponent),\\n"
    "            )\\n",
'''
new = '''    "                    float(thermoinhibition_threshold),\\n"
    "                    float(hydric_shock_threshold),\\n"
    "                )\\n",
    "                    float(thermoinhibition_threshold),\\n"
    "                    float(hydric_shock_threshold),\\n"
    "                    float(kr_exponent),\\n"
    "                )\\n",
'''
if script.count(old) != 1:
    raise RuntimeError("No se encontró el bloque de indentación del optimizador.")
script_path.write_text(script.replace(old, new, 1), encoding="utf-8")
runpy.run_path(str(script_path), run_name="__main__")
