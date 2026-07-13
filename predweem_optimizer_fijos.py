# -*- coding: utf-8 -*-
"""Optimizador hídrico PREDWEEM Lartigau con parámetros temporales fijos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np
import pandas as pd

EPS = 1e-12

# Subsistema temporal validado en app_emergencia_vK4_9_15.py.
FIXED_PARAMETERS: dict[str, float | int] = {
    "latencia_jd": 45,
    "ventana_termica": 5,
    "umbral_termoinhibicion": 24.0,
    "ventana_lluvia": 3,
    "umbral_choque_hidrico": 45.0,
    "fin_choque_jd": 110,
    "techo_choque": 1.0,
    "umbral_primer_pico": 0.70,
    "persistencia_primer_pico": 1,
    "lag_dias": 0,
}


@dataclass(frozen=True)
class ParameterSpec:
    low: float
    high: float
    default: float
    integer: bool = False


# Sólo parámetros libres. La cobertura es manual y no integra la búsqueda.
PARAMETER_SPACE: dict[str, ParameterSpec] = {
    "w_max": ParameterSpec(10.0, 40.0, 20.0),
    "humedad_p50": ParameterSpec(0.15, 0.55, 0.30),
    "pendiente_hidrica": ParameterSpec(5.0, 20.0, 10.0),
    "humedad_corte": ParameterSpec(0.05, 0.35, 0.20),
    "recarga_relativa": ParameterSpec(0.20, 0.90, 0.50),
}

DEFAULT_OPTIMIZED_PARAMETERS = list(PARAMETER_SPACE)

SCORE_WEIGHTS = {
    "KGE_Flujos": 0.25,
    "NSE_Flujos": 0.20,
    "CCC_Acumulado": 0.20,
    "RMSE_Acumulado": 0.15,
    "F1_Score": 0.10,
    "Sincronia_Inicio": 0.10,
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


def surface_parameters(cobertura_pct: float | int) -> tuple[float, float]:
    coverage = float(np.clip(cobertura_pct, 0.0, 100.0))
    x = [0.0, 30.0, 70.0, 100.0]
    ke = float(np.interp(coverage, x, [0.85, 0.50, 0.25, 0.10]))
    thermal = float(np.interp(coverage, x, [0.95, 0.90, 0.85, 0.80]))
    return ke, thermal


def _canonical_columns(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().upper(): c for c in df.columns}


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    canonical = _canonical_columns(df)
    for candidate in candidates:
        if candidate.upper() in canonical:
            return canonical[candidate.upper()]
    return None


def prepare_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("El archivo meteorológico está vacío.")
    date_col = _find_column(df, ["FECHA", "DATE", "DATETIME"])
    tmax_col = _find_column(df, ["TMAX", "T_MAX", "MAX_TEMP"])
    tmin_col = _find_column(df, ["TMIN", "T_MIN", "MIN_TEMP"])
    prec_col = _find_column(df, ["PREC", "PRECIPITACION", "PRECIP", "LLUVIA", "RAIN"])
    missing = [
        name for name, col in {
            "Fecha": date_col, "TMAX": tmax_col, "TMIN": tmin_col, "Prec": prec_col
        }.items() if col is None
    ]
    if missing:
        raise ValueError(f"Faltan columnas meteorológicas: {', '.join(missing)}")
    out = pd.DataFrame({
        "Fecha": pd.to_datetime(df[date_col], errors="coerce"),
        "TMAX": pd.to_numeric(df[tmax_col], errors="coerce"),
        "TMIN": pd.to_numeric(df[tmin_col], errors="coerce"),
        "Prec": pd.to_numeric(df[prec_col], errors="coerce").fillna(0.0),
    })
    out = (
        out.dropna(subset=["Fecha", "TMAX", "TMIN"])
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )
    out["Prec"] = out["Prec"].clip(lower=0.0)
    if len(out) < 30:
        raise ValueError("Se requieren al menos 30 días meteorológicos válidos.")
    return out


def prepare_field(
    df: pd.DataFrame,
    value_mode: str = "interval",
    date_column: str | None = None,
    value_column: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("El archivo de campo está vacío.")
    date_column = date_column or _find_column(df, ["FECHA", "DATE", "FECHA_MUESTREO"])
    value_column = value_column or _find_column(
        df, ["PLM2", "EMERGENCIA", "EMERREL", "OBSERVADO", "CONTEO", "PLANTAS_M2", "VALOR"]
    )
    if date_column is None:
        date_column = df.columns[0]
    if value_column is None:
        numeric = [
            c for c in df.columns
            if c != date_column and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2
        ]
        if not numeric:
            raise ValueError("No se encontró la columna de emergencia observada.")
        value_column = numeric[0]

    out = pd.DataFrame({
        "Fecha": pd.to_datetime(df[date_column], errors="coerce"),
        "Observado_original": pd.to_numeric(df[value_column], errors="coerce"),
    })
    out = (
        out.dropna()
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )
    out["Observado_original"] = out["Observado_original"].clip(lower=0.0)
    mode = value_mode.strip().lower()
    if mode == "cumulative":
        values = out["Observado_original"].to_numpy(float)
        out["Observado"] = np.clip(np.diff(np.r_[0.0, values]), 0.0, None)
    elif mode == "interval":
        out["Observado"] = out["Observado_original"]
    else:
        raise ValueError("value_mode debe ser 'interval' o 'cumulative'.")
    if len(out) < 4:
        raise ValueError("Se requieren al menos cuatro fechas de campo.")
    return out[["Fecha", "Observado", "Observado_original"]]


def calculate_et0_hargreaves(
    jday: np.ndarray,
    tmax: np.ndarray,
    tmin: np.ndarray,
    latitude: float = -38.6166,
) -> np.ndarray:
    jday = np.asarray(jday, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    lat_rad = np.radians(latitude)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * jday)
    dec = 0.409 * np.sin(2.0 * np.pi / 365.0 * jday - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(dec), -1.0, 1.0))
    ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(dec)
        + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
    )
    ra_mm = ra / 2.45
    tmean = (tmax + tmin) / 2.0
    trange = np.maximum(tmax - tmin, 0.0)
    return np.maximum(0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(trange), 0.0)


def surface_water_balance(
    prec: np.ndarray,
    et0: np.ndarray,
    w_max: float,
    ke_suelo: float,
) -> np.ndarray:
    """Balance con el mismo estado inicial de vK4.9.15."""
    prec = np.asarray(prec, dtype=float)
    et0 = np.asarray(et0, dtype=float)
    water = np.zeros(len(prec), dtype=float)
    if len(water) == 0:
        return water
    water[0] = float(w_max) / 2.0
    for i in range(1, len(water)):
        water[i] = np.clip(
            water[i - 1] + prec[i] - et0[i] * float(ke_suelo),
            0.0,
            float(w_max),
        )
    return water


def _apply_first_peak(values: np.ndarray) -> tuple[np.ndarray, int | None]:
    values = np.asarray(values, dtype=float).copy()
    candidates = np.flatnonzero(values > float(FIXED_PARAMETERS["umbral_primer_pico"]))
    if len(candidates) == 0:
        return np.zeros_like(values), None
    start = int(candidates[0])
    values[:start] = 0.0
    return values, start


def simulate_emergence(
    weather: pd.DataFrame,
    ann_model: Any,
    params: Mapping[str, float | int],
    cobertura_pct: int = 75,
    latitude: float = -38.6166,
) -> pd.DataFrame:
    missing = [name for name in PARAMETER_SPACE if name not in params]
    if missing:
        raise ValueError(f"Faltan parámetros libres: {', '.join(missing)}")

    df = prepare_weather(weather).copy()
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2.0
    amplitude = (df["TMAX"] - df["TMIN"]) / 2.0
    ke_suelo, mod_termico = surface_parameters(cobertura_pct)
    df["Cobertura_Rastrojo"] = int(cobertura_pct)
    df["Ke_Suelo"] = ke_suelo
    df["Mod_Termico"] = mod_termico
    df["TMAX_suelo"] = df["Tmedia_aire"] + amplitude * mod_termico
    df["TMIN_suelo"] = df["Tmedia_aire"] - amplitude * mod_termico

    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    raw, _ = ann_model.predict(X)
    df["EMERREL_RAW"] = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)

    # Choque hídrico fijo de vK4.9.15.
    df["Prec_3d"] = df["Prec"].rolling(
        int(FIXED_PARAMETERS["ventana_lluvia"]), min_periods=1
    ).sum()
    shock = (
        (df["Julian_days"] > int(FIXED_PARAMETERS["latencia_jd"]))
        & (df["Julian_days"] <= int(FIXED_PARAMETERS["fin_choque_jd"]))
        & (df["Prec_3d"] >= float(FIXED_PARAMETERS["umbral_choque_hidrico"]))
    )
    df.loc[shock, "EMERREL_RAW"] = np.maximum(
        df.loc[shock, "EMERREL_RAW"],
        float(FIXED_PARAMETERS["techo_choque"]),
    )

    df["ET0"] = calculate_et0_hargreaves(
        df["Julian_days"].to_numpy(),
        df["TMAX"].to_numpy(),
        df["TMIN"].to_numpy(),
        latitude,
    )
    w_max = float(params["w_max"])
    df["W_superficial"] = surface_water_balance(
        df["Prec"].to_numpy(),
        df["ET0"].to_numpy(),
        w_max,
        ke_suelo,
    )
    rel_water = df["W_superficial"].to_numpy(float) / max(w_max, EPS)
    df["Humedad_Relativa"] = rel_water

    exponent = np.clip(
        -float(params["pendiente_hidrica"])
        * (rel_water - float(params["humedad_p50"])),
        -60.0,
        60.0,
    )
    df["Factor_Hidrico"] = 1.0 / (1.0 + np.exp(exponent))
    emergence = df["EMERREL_RAW"].to_numpy() * df["Factor_Hidrico"].to_numpy()
    emergence[rel_water < float(params["humedad_corte"])] = 0.0

    recharge = pd.Series(
        rel_water >= float(params["recarga_relativa"])
    ).cummax().to_numpy(bool)
    emergence[~recharge] = 0.0
    df["Recarga_Habilitada"] = recharge

    # Termoinhibición fija.
    df["Tmedia_5d"] = df["Tmedia_aire"].rolling(
        int(FIXED_PARAMETERS["ventana_termica"]), min_periods=1
    ).mean()
    thermoinhibited = (
        df["Tmedia_5d"].to_numpy()
        >= float(FIXED_PARAMETERS["umbral_termoinhibicion"])
    )
    emergence[thermoinhibited] = 0.0
    df["Termoinhibida"] = thermoinhibited

    # Latencia fija al final, igual que el motor validado.
    emergence[
        df["Julian_days"].to_numpy() <= int(FIXED_PARAMETERS["latencia_jd"])
    ] = 0.0

    emergence, first_peak = _apply_first_peak(np.clip(emergence, 0.0, 1.0))
    df["EMERREL"] = emergence
    df["Primer_Pico_Indice"] = first_peak if first_peak is not None else -1
    df["Fecha_Primer_Pico"] = (
        df.loc[first_peak, "Fecha"] if first_peak is not None else pd.NaT
    )
    return df


def build_intervals(field: pd.DataFrame, simulation_start: pd.Timestamp) -> pd.DataFrame:
    field = field.sort_values("Fecha").reset_index(drop=True)
    rows = []
    previous = pd.Timestamp(simulation_start) - pd.Timedelta(days=1)
    for _, row in field.iterrows():
        end = pd.Timestamp(row["Fecha"])
        rows.append({
            "Inicio": previous,
            "Fecha": end,
            "Observado": float(row["Observado"]),
        })
        previous = end
    return pd.DataFrame(rows)


def synchronize_intervals(
    simulation: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    records = []
    for _, row in intervals.iterrows():
        start = pd.Timestamp(row["Inicio"])
        end = pd.Timestamp(row["Fecha"])
        sim_flow = simulation.loc[
            (simulation["Fecha"] > start) & (simulation["Fecha"] <= end),
            "EMERREL",
        ].sum()
        records.append({
            "Inicio": start,
            "Fecha": end,
            "Flujo_Obs_Abs": float(row["Observado"]),
            "Flujo_Sim_Abs": float(sim_flow),
        })
    out = pd.DataFrame(records)
    obs_total = float(out["Flujo_Obs_Abs"].sum())
    sim_total = float(out["Flujo_Sim_Abs"].sum())
    out["Acum_Obs_Abs"] = out["Flujo_Obs_Abs"].cumsum()
    out["Acum_Sim_Abs"] = out["Flujo_Sim_Abs"].cumsum()
    out["Campo_Relativo"] = out["Flujo_Obs_Abs"] / obs_total if obs_total > EPS else 0.0
    out["Sim_Relativo"] = out["Flujo_Sim_Abs"] / sim_total if sim_total > EPS else 0.0
    out["Campo_Acumulado"] = out["Acum_Obs_Abs"] / obs_total if obs_total > EPS else 0.0
    out["Sim_Acumulado"] = out["Acum_Sim_Abs"] / sim_total if sim_total > EPS else 0.0
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def validation_metrics(sync: pd.DataFrame) -> dict[str, float]:
    if sync is None or len(sync) < 2:
        return {
            "KGE_Flujos": -1.0,
            "NSE_Flujos": -1.0,
            "CCC_Acumulado": 0.0,
            "RMSE_Acumulado": 1.0,
            "F1_Score": 0.0,
        }

    active = sync[(sync["Campo_Relativo"] > 0) | (sync["Sim_Relativo"] > 0)]
    obs = active["Campo_Relativo"].to_numpy(float)
    sim = active["Sim_Relativo"].to_numpy(float)
    if len(obs) < 2:
        nse = kge = -1.0
    else:
        corr = _safe_corr(obs, sim)
        obs_mean = float(np.mean(obs))
        obs_std = float(np.std(obs))
        denom = np.sum((obs - obs_mean) ** 2)
        nse = 1.0 - np.sum((sim - obs) ** 2) / denom if denom > EPS else -1.0
        if obs_mean > EPS and obs_std > EPS:
            alpha = np.std(sim) / obs_std
            beta = np.mean(sim) / obs_mean
            kge = 1.0 - np.sqrt(
                (corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2
            )
        else:
            kge = -1.0

    obs_c = sync["Campo_Acumulado"].to_numpy(float)
    sim_c = sync["Sim_Acumulado"].to_numpy(float)
    rmse = float(np.sqrt(np.mean((obs_c - sim_c) ** 2)))
    mean_o, mean_s = np.mean(obs_c), np.mean(sim_c)
    var_o, var_s = np.var(obs_c), np.var(sim_c)
    cov = np.mean((obs_c - mean_o) * (sim_c - mean_s))
    ccc_den = var_o + var_s + (mean_o - mean_s) ** 2
    ccc = float(2.0 * cov / ccc_den) if ccc_den > EPS else 0.0

    obs_events = sync["Campo_Relativo"].to_numpy() > 0.05
    sim_events = sync["Sim_Relativo"].to_numpy() > 0.05
    hits = int(np.sum(obs_events & sim_events))
    misses = int(np.sum(obs_events & ~sim_events))
    false_pos = int(np.sum(~obs_events & sim_events))
    precision = hits / (hits + false_pos) if hits + false_pos else 0.0
    recall = hits / (hits + misses) if hits + misses else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "KGE_Flujos": float(kge),
        "NSE_Flujos": float(nse),
        "CCC_Acumulado": float(ccc),
        "RMSE_Acumulado": rmse,
        "F1_Score": float(f1),
    }


def onset_interval(field: pd.DataFrame, simulation_start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    field = field.sort_values("Fecha").reset_index(drop=True)
    positive = np.flatnonzero(field["Observado"].to_numpy(float) > 0)
    if len(positive) == 0:
        return None
    i = int(positive[0])
    start = (
        pd.Timestamp(simulation_start) - pd.Timedelta(days=1)
        if i == 0
        else pd.Timestamp(field.iloc[i - 1]["Fecha"])
    )
    end = pd.Timestamp(field.iloc[i]["Fecha"])
    return start, end


def onset_distance_days(
    simulation: pd.DataFrame,
    observed_interval: tuple[pd.Timestamp, pd.Timestamp] | None,
) -> int:
    if observed_interval is None:
        return 0
    active = simulation.index[simulation["EMERREL"] > 0].tolist()
    if not active:
        return 365
    date = pd.Timestamp(simulation.loc[active[0], "Fecha"])
    start, end = observed_interval
    if date <= start:
        return int((date - start).days)
    if date > end:
        return int((date - end).days)
    return 0


def objective_score(metrics: Mapping[str, float], onset_days: int) -> float:
    transformed = {
        "KGE_Flujos": np.clip((metrics["KGE_Flujos"] + 1.0) / 2.0, 0.0, 1.0),
        "NSE_Flujos": np.clip((metrics["NSE_Flujos"] + 1.0) / 2.0, 0.0, 1.0),
        "CCC_Acumulado": np.clip((metrics["CCC_Acumulado"] + 1.0) / 2.0, 0.0, 1.0),
        "RMSE_Acumulado": np.clip(1.0 - metrics["RMSE_Acumulado"], 0.0, 1.0),
        "F1_Score": np.clip(metrics["F1_Score"], 0.0, 1.0),
        "Sincronia_Inicio": float(np.exp(-abs(onset_days) / 14.0)),
    }
    return float(sum(SCORE_WEIGHTS[k] * transformed[k] for k in SCORE_WEIGHTS))


def make_temporal_folds(
    intervals: pd.DataFrame,
    n_folds: int = 3,
    min_intervals_per_fold: int = 2,
) -> pd.DataFrame:
    max_folds = len(intervals) // max(1, int(min_intervals_per_fold))
    folds = min(int(n_folds), max_folds)
    if folds < 2:
        raise ValueError(
            f"Se requieren al menos {2 * min_intervals_per_fold} intervalos para CV temporal."
        )
    frames = []
    for number, indices in enumerate(np.array_split(np.arange(len(intervals)), folds), start=1):
        part = intervals.iloc[indices].copy()
        part["Fold"] = f"B{number}"
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def default_free_parameters() -> dict[str, float | int]:
    return {
        name: int(spec.default) if spec.integer else float(spec.default)
        for name, spec in PARAMETER_SPACE.items()
    }


def _cast(name: str, value: float) -> float | int:
    spec = PARAMETER_SPACE[name]
    clipped = np.clip(value, spec.low, spec.high)
    return int(round(clipped)) if spec.integer else float(clipped)


def sample_candidates(
    n: int,
    optimized_parameters: Sequence[str],
    rng: np.random.Generator,
    fixed_free_parameters: Mapping[str, float | int] | None = None,
) -> list[dict[str, float | int]]:
    base = default_free_parameters()
    if fixed_free_parameters:
        base.update(fixed_free_parameters)
    draws: dict[str, np.ndarray] = {}
    for name in optimized_parameters:
        spec = PARAMETER_SPACE[name]
        bins = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(bins)
        draws[name] = spec.low + bins * (spec.high - spec.low)
    candidates = []
    for i in range(n):
        candidate = dict(base)
        for name in optimized_parameters:
            candidate[name] = _cast(name, float(draws[name][i]))
        candidates.append(candidate)
    return candidates


def local_candidates(
    seeds: Sequence[Mapping[str, float | int]],
    n_total: int,
    optimized_parameters: Sequence[str],
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    if n_total <= 0 or not seeds:
        return []
    per_seed = int(np.ceil(n_total / len(seeds)))
    candidates = []
    for seed in seeds:
        for _ in range(per_seed):
            candidate = dict(seed)
            for name in optimized_parameters:
                spec = PARAMETER_SPACE[name]
                sigma = 0.10 * (spec.high - spec.low)
                candidate[name] = _cast(name, float(seed[name]) + rng.normal(0.0, sigma))
            candidates.append(candidate)
    return candidates[:n_total]


def evaluate_candidate_cv(
    weather: pd.DataFrame,
    field: pd.DataFrame,
    folds: pd.DataFrame,
    ann_model: Any,
    params: Mapping[str, float | int],
    cobertura_pct: int,
    latitude: float,
    robustness_penalty: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    simulation = simulate_emergence(
        weather, ann_model, params, cobertura_pct=cobertura_pct, latitude=latitude
    )
    observed_onset = onset_interval(field, weather["Fecha"].min())
    onset_days = onset_distance_days(simulation, observed_onset)
    rows = []
    sync_frames = []
    for fold_name, intervals in folds.groupby("Fold", sort=False):
        sync = synchronize_intervals(simulation, intervals)
        metrics = validation_metrics(sync)
        score = objective_score(metrics, onset_days)
        rows.append({
            "Fold": fold_name,
            "N_Intervalos": len(intervals),
            "Fecha_Inicio": intervals["Inicio"].min(),
            "Fecha_Fin": intervals["Fecha"].max(),
            "Score": score,
            "Desfase_Inicio_Dias": onset_days,
            **metrics,
        })
        sync["Fold"] = fold_name
        sync_frames.append(sync)
    by_fold = pd.DataFrame(rows)
    scores = by_fold["Score"].to_numpy(float)
    summary = {
        "Score_CV": float(np.mean(scores) - robustness_penalty * np.std(scores)),
        "Score_CV_Medio": float(np.mean(scores)),
        "Score_CV_SD": float(np.std(scores)),
        "Score_CV_Peor_Bloque": float(np.min(scores)),
        "Desfase_Inicio_Dias": int(onset_days),
        "Fecha_Primer_Pico_Simulado": simulation["Fecha_Primer_Pico"].dropna().iloc[0]
        if simulation["Fecha_Primer_Pico"].notna().any() else pd.NaT,
    }
    for metric in [
        "KGE_Flujos", "NSE_Flujos", "CCC_Acumulado",
        "RMSE_Acumulado", "F1_Score",
    ]:
        summary[f"{metric}_Media"] = float(by_fold[metric].mean())
    return summary, by_fold, pd.concat(sync_frames, ignore_index=True)


def optimize_parameters_temporal_cv(
    weather_data: pd.DataFrame,
    field_data: pd.DataFrame,
    ann_model: Any,
    *,
    optimized_parameters: Sequence[str] = DEFAULT_OPTIMIZED_PARAMETERS,
    fixed_free_parameters: Mapping[str, float | int] | None = None,
    cobertura_pct: int = 75,
    n_global: int = 400,
    n_local: int = 200,
    seed: int = 42,
    latitude: float = -38.6166,
    robustness_penalty: float = 0.15,
    n_folds: int = 3,
    min_intervals_per_fold: int = 2,
) -> dict[str, Any]:
    forbidden = [name for name in optimized_parameters if name in FIXED_PARAMETERS]
    if forbidden:
        raise ValueError(
            "Los siguientes parámetros están fijados y no pueden optimizarse: "
            + ", ".join(forbidden)
        )
    unknown = [name for name in optimized_parameters if name not in PARAMETER_SPACE]
    if unknown:
        raise ValueError(f"Parámetros libres desconocidos: {', '.join(unknown)}")
    if not optimized_parameters:
        raise ValueError("Seleccione al menos un parámetro libre.")

    weather = prepare_weather(weather_data)
    field = (
        prepare_field(field_data)
        if "Observado" not in field_data.columns
        else field_data.copy()
    )
    intervals = build_intervals(field, weather["Fecha"].min())
    folds = make_temporal_folds(intervals, n_folds, min_intervals_per_fold)

    rng = np.random.default_rng(seed)
    candidates = sample_candidates(
        max(1, int(n_global)),
        optimized_parameters,
        rng,
        fixed_free_parameters,
    )
    rows = []
    for candidate in candidates:
        summary, _, _ = evaluate_candidate_cv(
            weather, field, folds, ann_model, candidate,
            int(cobertura_pct), float(latitude), float(robustness_penalty),
        )
        rows.append({**candidate, **summary, "Etapa": "global"})

    results = pd.DataFrame(rows).sort_values("Score_CV", ascending=False)
    seeds = results.head(5)[list(PARAMETER_SPACE)].to_dict("records")
    for candidate in local_candidates(seeds, int(n_local), optimized_parameters, rng):
        summary, _, _ = evaluate_candidate_cv(
            weather, field, folds, ann_model, candidate,
            int(cobertura_pct), float(latitude), float(robustness_penalty),
        )
        rows.append({**candidate, **summary, "Etapa": "local"})

    results = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=list(PARAMETER_SPACE), keep="first")
        .sort_values(["Score_CV", "Score_CV_Peor_Bloque"], ascending=False)
        .reset_index(drop=True)
    )
    best = {
        name: _cast(name, float(results.iloc[0][name]))
        for name in PARAMETER_SPACE
    }
    best_summary, by_fold, cv_sync = evaluate_candidate_cv(
        weather, field, folds, ann_model, best,
        int(cobertura_pct), float(latitude), float(robustness_penalty),
    )
    simulation = simulate_emergence(
        weather, ann_model, best, int(cobertura_pct), float(latitude)
    )
    full_sync = synchronize_intervals(simulation, intervals)
    full_metrics = validation_metrics(full_sync)
    return {
        "best_params": best,
        "best_summary": best_summary,
        "results": results,
        "cv_by_fold": by_fold,
        "cv_sync": cv_sync,
        "full_sync": full_sync,
        "full_metrics": full_metrics,
        "simulation": simulation,
        "fold_intervals": folds,
        "fixed_parameters": dict(FIXED_PARAMETERS),
        "cobertura_pct": int(cobertura_pct),
    }


def params_to_json(result: Mapping[str, Any]) -> str:
    coverage = int(result["cobertura_pct"])
    ke, thermal = surface_parameters(coverage)
    payload = {
        "parametros_optimos_libres": result["best_params"],
        "cobertura_manual_pct": coverage,
        "ke_derivado": ke,
        "modulador_termico_derivado": thermal,
        "parametros_fijos_no_optimizados": result["fixed_parameters"],
        "resumen_cv": {
            key: (
                value.isoformat()
                if isinstance(value, pd.Timestamp)
                else value.item() if hasattr(value, "item") else value
            )
            for key, value in result["best_summary"].items()
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
