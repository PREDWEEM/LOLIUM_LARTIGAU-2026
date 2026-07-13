# -*- coding: utf-8 -*-
"""Datos, red neuronal y espacio paramétrico del optimizador Lartigau."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import json

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class ParameterSpec:
    low: float
    high: float
    default: float
    integer: bool = False


# El porcentaje de cobertura reemplaza a Ke y mod_termico como parámetro libre.
# Ambos se derivan con las curvas que ya utiliza app_emergencia.py.
PARAMETER_SPACE: dict[str, ParameterSpec] = {
    "w_max": ParameterSpec(10.0, 40.0, 20.0),
    "cobertura_pct": ParameterSpec(0, 100, 75, integer=True),
    "humedad_p50": ParameterSpec(0.15, 0.55, 0.30),
    "pendiente_hidrica": ParameterSpec(5.0, 20.0, 10.0),
    "humedad_corte": ParameterSpec(0.05, 0.35, 0.20),
    "recarga_relativa": ParameterSpec(0.20, 0.90, 0.50),
    "latencia_jd": ParameterSpec(20, 80, 45, integer=True),
    "ventana_termica": ParameterSpec(3, 20, 5, integer=True),
    "umbral_termoinhibicion": ParameterSpec(18.0, 30.0, 24.0),
    "ventana_lluvia": ParameterSpec(1, 7, 3, integer=True),
    "umbral_choque_hidrico": ParameterSpec(20.0, 100.0, 45.0),
    "fin_choque_jd": ParameterSpec(75, 160, 110, integer=True),
    "techo_choque": ParameterSpec(0.50, 1.00, 1.00),
    "umbral_primer_pico": ParameterSpec(0.15, 0.90, 0.70),
    "persistencia_primer_pico": ParameterSpec(1, 3, 1, integer=True),
    # El modelo operativo actual no desplaza la señal. El optimizador permite
    # cuantificar si una corrección temporal mejoraría el ajuste.
    "lag_dias": ParameterSpec(-30, 40, 0, integer=True),
}

DEFAULT_OPTIMIZED_PARAMETERS = [
    "w_max",
    "cobertura_pct",
    "humedad_p50",
    "humedad_corte",
    "recarga_relativa",
    "latencia_jd",
    "ventana_termica",
    "umbral_termoinhibicion",
    "umbral_choque_hidrico",
    "umbral_primer_pico",
    "lag_dias",
]

DEFAULT_WEIGHTS = {
    "KGE_Flujos": 0.25,
    "NSE_Flujos": 0.20,
    "CCC_Acumulado": 0.20,
    "F1_Score": 0.15,
    "RMSE_Acumulado": 0.15,
    "Desfase_T50": 0.05,
}


class PracticalANNModel:
    def __init__(self, IW: np.ndarray, bIW: np.ndarray, LW: np.ndarray, bLW: np.ndarray):
        self.IW = np.asarray(IW, dtype=float)
        self.bIW = np.asarray(bIW, dtype=float)
        self.LW = np.asarray(LW, dtype=float)
        self.bLW = np.asarray(bLW, dtype=float)
        self.input_min = np.array([1.0, 0.0, -7.0, 0.0])
        self.input_max = np.array([300.0, 41.0, 25.5, 84.0])

    def normalize(self, X: np.ndarray) -> np.ndarray:
        return 2.0 * (X - self.input_min) / (self.input_max - self.input_min) - 1.0

    def predict(self, Xreal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xn = self.normalize(np.asarray(Xreal, dtype=float))
        a1 = np.tanh(Xn @ self.IW + self.bIW)
        out = np.tanh((a1 @ self.LW.T).reshape(-1) + self.bLW.reshape(-1)[0])
        emerrel = (out + 1.0) / 2.0
        return emerrel, np.cumsum(emerrel)


def load_ann_model(base: str | Path) -> PracticalANNModel:
    base = Path(base)
    required = ["IW.npy", "bias_IW.npy", "LW.npy", "bias_out.npy"]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Faltan archivos ANN: {', '.join(missing)}")
    return PracticalANNModel(
        np.load(base / "IW.npy"),
        np.load(base / "bias_IW.npy"),
        np.load(base / "LW.npy"),
        np.load(base / "bias_out.npy"),
    )


def default_parameters() -> dict[str, float | int]:
    return {
        name: int(spec.default) if spec.integer else float(spec.default)
        for name, spec in PARAMETER_SPACE.items()
    }


def surface_parameters(cobertura_pct: float | int) -> tuple[float, float]:
    """Devuelve Ke y modulador térmico usando las curvas de app_emergencia.py."""
    coverage = float(np.clip(cobertura_pct, 0.0, 100.0))
    x = [0.0, 30.0, 70.0, 100.0]
    ke = float(np.interp(coverage, x, [0.85, 0.50, 0.25, 0.10]))
    thermal = float(np.interp(coverage, x, [0.95, 0.90, 0.85, 0.80]))
    return ke, thermal


def _canonical_columns(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().upper(): c for c in df.columns}


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    canonical = _canonical_columns(df)
    for candidate in candidates:
        if candidate.upper() in canonical:
            return canonical[candidate.upper()]
    return None


def prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("El archivo meteorológico está vacío.")
    date_col = _find_column(df, ["FECHA", "DATE", "DATETIME"])
    tmax_col = _find_column(df, ["TMAX", "T_MAX", "MAX_TEMP", "TEMPERATURA_MAXIMA"])
    tmin_col = _find_column(df, ["TMIN", "T_MIN", "MIN_TEMP", "TEMPERATURA_MINIMA"])
    prec_col = _find_column(df, ["PREC", "PRECIPITACION", "PRECIP", "LLUVIA", "RAIN"])
    missing = [
        name
        for name, col in {
            "Fecha": date_col,
            "TMAX": tmax_col,
            "TMIN": tmin_col,
            "Prec": prec_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(f"Faltan columnas meteorológicas: {', '.join(missing)}")

    out = pd.DataFrame({
        "Fecha": pd.to_datetime(df[date_col], errors="coerce"),
        "TMAX": pd.to_numeric(df[tmax_col], errors="coerce"),
        "TMIN": pd.to_numeric(df[tmin_col], errors="coerce"),
        "Prec": pd.to_numeric(df[prec_col], errors="coerce").fillna(0.0),
    })
    out = out.dropna(subset=["Fecha", "TMAX", "TMIN"]).sort_values("Fecha")
    out = out.drop_duplicates("Fecha", keep="last").reset_index(drop=True)
    out["Prec"] = out["Prec"].clip(lower=0.0)
    if len(out) < 10:
        raise ValueError("Se requieren al menos 10 días meteorológicos válidos.")
    return out


def prepare_field(
    df: pd.DataFrame,
    value_mode: str = "interval",
    date_column: str | None = None,
    value_column: str | None = None,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Estandariza un único set de observaciones de campo.

    value_mode:
      - interval: flujo/conteo desde el muestreo anterior;
      - cumulative: conteo acumulado, convertido internamente a flujo.
    """
    if df is None or df.empty:
        raise ValueError("El archivo de campo está vacío.")

    date_column = date_column or _find_column(df, ["FECHA", "DATE", "FECHA_MUESTREO"])
    value_column = value_column or _find_column(
        df,
        ["PLM2", "EMERGENCIA", "EMERREL", "OBSERVADO", "CONTEO", "PLANTAS_M2", "VALOR"],
    )
    if date_column is None:
        date_column = df.columns[0]
    if value_column is None:
        numeric_candidates = [
            c for c in df.columns
            if c != date_column and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2
        ]
        if not numeric_candidates:
            raise ValueError("No se encontró una columna numérica de emergencia de campo.")
        value_column = numeric_candidates[0]

    if group_column is None:
        group_column = _find_column(df, ["GRUPO", "SITIO", "LOCALIDAD", "CAMPANIA", "CAMPAÑA", "YEAR", "AÑO"])

    out = pd.DataFrame({
        "Fecha": pd.to_datetime(df[date_column], errors="coerce"),
        "Observado_original": pd.to_numeric(df[value_column], errors="coerce"),
    })
    out["Grupo"] = df[group_column].astype(str) if group_column else "Lartigau"
    out = out.dropna(subset=["Fecha", "Observado_original"]).sort_values(["Grupo", "Fecha"])
    out["Observado_original"] = out["Observado_original"].clip(lower=0.0)

    mode = value_mode.lower().strip()
    if mode not in {"interval", "cumulative"}:
        raise ValueError("value_mode debe ser 'interval' o 'cumulative'.")

    parts: list[pd.DataFrame] = []
    for group, part in out.groupby("Grupo", sort=False):
        part = part.sort_values("Fecha").drop_duplicates("Fecha", keep="last").copy()
        if mode == "cumulative":
            values = part["Observado_original"].to_numpy(float)
            part["Observado"] = np.clip(np.diff(np.r_[0.0, values]), 0.0, None)
        else:
            part["Observado"] = part["Observado_original"].to_numpy(float)
        part["Grupo"] = group
        parts.append(part)

    result = pd.concat(parts, ignore_index=True)
    if len(result) < 4:
        raise ValueError("Se requieren al menos 4 fechas de campo para dos bloques temporales.")
    return result[["Grupo", "Fecha", "Observado", "Observado_original"]]


def params_to_json(params: Mapping[str, object]) -> str:
    serializable: dict[str, float | int] = {}
    for name, value in params.items():
        spec = PARAMETER_SPACE.get(name)
        serializable[name] = int(value) if spec and spec.integer else float(value)
    coverage = serializable.get("cobertura_pct", default_parameters()["cobertura_pct"])
    ke, thermal = surface_parameters(coverage)
    serializable["ke_suelo_derivado"] = ke
    serializable["mod_termico_derivado"] = thermal
    return json.dumps(serializable, ensure_ascii=False, indent=2)
