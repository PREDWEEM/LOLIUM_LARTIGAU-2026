# -*- coding: utf-8 -*-
"""Motor ecofisiológico Lartigau y métricas event-to-event."""
from __future__ import annotations
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .data import (
    DEFAULT_WEIGHTS,
    EPS,
    default_parameters,
    prepare_weather,
    surface_parameters,
)


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


def surface_water_balance(prec: np.ndarray, et0: np.ndarray, w_max: float, ke_suelo: float) -> np.ndarray:
    prec = np.asarray(prec, dtype=float)
    et0 = np.asarray(et0, dtype=float)
    water = np.zeros(len(prec), dtype=float)
    if len(water) == 0:
        return water
    water[0] = np.clip(w_max / 2.0 + prec[0] - et0[0] * ke_suelo, 0.0, w_max)
    for i in range(1, len(water)):
        water[i] = np.clip(water[i - 1] + prec[i] - et0[i] * ke_suelo, 0.0, w_max)
    return water


def _apply_first_peak(values: np.ndarray, threshold: float, persistence: int) -> tuple[np.ndarray, int | None]:
    values = np.asarray(values, dtype=float).copy()
    persistence = max(1, int(persistence))
    above = values > threshold
    if persistence == 1:
        candidates = np.flatnonzero(above)
    else:
        rolling = pd.Series(above.astype(int)).rolling(
            persistence, min_periods=persistence
        ).sum().to_numpy()
        candidates = np.flatnonzero(rolling >= persistence)
    if len(candidates) == 0:
        return np.zeros_like(values), None
    start = int(candidates[0])
    values[:start] = 0.0
    return values, start


def _shift_signal(values: np.ndarray, lag_days: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lag_days = int(lag_days)
    shifted = np.zeros_like(values)
    if lag_days == 0:
        return values.copy()
    if lag_days > 0 and lag_days < len(values):
        shifted[lag_days:] = values[:-lag_days]
    elif lag_days < 0 and abs(lag_days) < len(values):
        k = abs(lag_days)
        shifted[:-k] = values[k:]
    return shifted


def simulate_emergence(
    weather: pd.DataFrame,
    ann_model: Any,
    params: Mapping[str, float | int],
    latitude: float = -38.6166,
) -> pd.DataFrame:
    p = default_parameters()
    p.update(dict(params))
    df = prepare_weather(weather).copy()
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    df["Tmedia_aire"] = (df["TMAX"] + df["TMIN"]) / 2.0
    amplitude = (df["TMAX"] - df["TMIN"]) / 2.0
    ke_suelo, mod_termico = surface_parameters(p["cobertura_pct"])
    df["Cobertura_Rastrojo"] = int(p["cobertura_pct"])
    df["Ke_Suelo"] = ke_suelo
    df["Mod_Termico"] = mod_termico
    df["TMAX_suelo"] = df["Tmedia_aire"] + amplitude * mod_termico
    df["TMIN_suelo"] = df["Tmedia_aire"] - amplitude * mod_termico

    X = df[["Julian_days", "TMAX_suelo", "TMIN_suelo", "Prec"]].to_numpy(float)
    raw, _ = ann_model.predict(X)
    df["EMERREL_RAW"] = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)

    latency = int(p["latencia_jd"])
    df.loc[df["Julian_days"] <= latency, "EMERREL_RAW"] = 0.0

    rain_window = int(p["ventana_lluvia"])
    df["Prec_Acum_Choque"] = df["Prec"].rolling(rain_window, min_periods=1).sum()
    shock = (
        (df["Julian_days"] > latency)
        & (df["Julian_days"] <= int(p["fin_choque_jd"]))
        & (df["Prec_Acum_Choque"] >= float(p["umbral_choque_hidrico"]))
    )
    df.loc[shock, "EMERREL_RAW"] = np.maximum(
        df.loc[shock, "EMERREL_RAW"], float(p["techo_choque"])
    )

    df["ET0"] = calculate_et0_hargreaves(
        df["Julian_days"].to_numpy(),
        df["TMAX"].to_numpy(),
        df["TMIN"].to_numpy(),
        latitude,
    )
    df["W_superficial"] = surface_water_balance(
        df["Prec"].to_numpy(), df["ET0"].to_numpy(), float(p["w_max"]), ke_suelo
    )
    rel_water = df["W_superficial"].to_numpy(float) / max(float(p["w_max"]), EPS)
    df["Humedad_Relativa"] = rel_water
    exponent = np.clip(
        -float(p["pendiente_hidrica"]) * (rel_water - float(p["humedad_p50"])),
        -60,
        60,
    )
    df["Factor_Hidrico"] = 1.0 / (1.0 + np.exp(exponent))
    emer = df["EMERREL_RAW"].to_numpy() * df["Factor_Hidrico"].to_numpy()
    emer[rel_water < float(p["humedad_corte"])] = 0.0

    # Recarga por estado hídrico alcanzado, no por una lluvia diaria >= Wmax.
    recharge = pd.Series(rel_water >= float(p["recarga_relativa"])).cummax().to_numpy(bool)
    emer[~recharge] = 0.0
    df["Recarga_Habilitada"] = recharge

    thermal_window = int(p["ventana_termica"])
    df["Tmedia_Movil"] = df["Tmedia_aire"].rolling(thermal_window, min_periods=1).mean()
    thermoinhibited = df["Tmedia_Movil"].to_numpy() >= float(p["umbral_termoinhibicion"])
    emer[thermoinhibited] = 0.0
    df["Termoinhibida"] = thermoinhibited

    emer = np.clip(emer, 0.0, 1.0)
    emer, first_pre_lag = _apply_first_peak(
        emer, float(p["umbral_primer_pico"]), int(p["persistencia_primer_pico"])
    )
    df["EMERREL_SIN_LAG"] = emer
    emer = _shift_signal(emer, int(p["lag_dias"]))
    emer, first_post_lag = _apply_first_peak(
        emer, float(p["umbral_primer_pico"]), int(p["persistencia_primer_pico"])
    )
    df["EMERREL"] = np.clip(emer, 0.0, 1.0)
    df["Primer_Pico_PreLag"] = first_pre_lag if first_pre_lag is not None else -1
    df["Primer_Pico_PostLag"] = first_post_lag if first_post_lag is not None else -1
    return df


def synchronize_intervals(simulation: pd.DataFrame, field: pd.DataFrame) -> pd.DataFrame:
    sim = simulation.sort_values("Fecha").copy()
    obs = field.sort_values("Fecha").copy()
    if len(obs) < 2:
        return pd.DataFrame()
    last_date = obs["Fecha"].max()
    sim_total = sim.loc[sim["Fecha"] <= last_date, "EMERREL"].sum()
    obs_total = obs["Observado"].sum()
    records: list[dict[str, Any]] = []
    obs_cum = obs["Observado"].cumsum().to_numpy(float)
    sim_start = sim["Fecha"].min() - pd.Timedelta(days=1)
    for i in range(len(obs)):
        start = sim_start if i == 0 else obs.iloc[i - 1]["Fecha"]
        end = obs.iloc[i]["Fecha"]
        sim_flow = sim.loc[(sim["Fecha"] > start) & (sim["Fecha"] <= end), "EMERREL"].sum()
        sim_cum = sim.loc[sim["Fecha"] <= end, "EMERREL"].sum()
        records.append({
            "Fecha": end,
            "Dias_Intervalo": int((end - start).days),
            "Flujo_Obs_Abs": float(obs.iloc[i]["Observado"]),
            "Flujo_Sim_Abs": float(sim_flow),
            "Acum_Obs_Abs": float(obs_cum[i]),
            "Acum_Sim_Abs": float(sim_cum),
        })
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["Campo_Relativo"] = result["Flujo_Obs_Abs"] / obs_total if obs_total > EPS else 0.0
    result["Sim_Relativo"] = result["Flujo_Sim_Abs"] / sim_total if sim_total > EPS else 0.0
    result["Campo_Acumulado"] = result["Acum_Obs_Abs"] / obs_total if obs_total > EPS else 0.0
    result["Sim_Acumulado"] = result["Acum_Sim_Abs"] / sim_total if sim_total > EPS else 0.0
    return result


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def validation_metrics(sync: pd.DataFrame, detection_threshold: float = 0.05) -> dict[str, float | int]:
    zero = {
        "Pearson_Flujos": 0.0,
        "NSE_Flujos": -1.0,
        "KGE_Flujos": -1.0,
        "RMSE_Acumulado": 1.0,
        "CCC_Acumulado": 0.0,
        "R2_Acumulado": -1.0,
        "Exactitud": 0.0,
        "F1_Score": 0.0,
        "Desfase_T50": 365.0,
        "Hits": 0,
        "Misses": 0,
        "Falsos_Positivos": 0,
        "Correctos_Negativos": 0,
    }
    if sync is None or len(sync) < 2:
        return zero

    active = sync[(sync["Campo_Relativo"] > 0) | (sync["Sim_Relativo"] > 0)]
    obs = active["Campo_Relativo"].to_numpy(float)
    sim = active["Sim_Relativo"].to_numpy(float)
    if len(obs) < 2:
        pearson, nse, kge = 0.0, -1.0, -1.0
    else:
        pearson = _safe_corr(obs, sim)
        obs_mean = float(np.mean(obs))
        obs_std = float(np.std(obs))
        obs_variance = np.sum((obs - obs_mean) ** 2)
        nse = 1.0 - np.sum((sim - obs) ** 2) / obs_variance if obs_variance > EPS else -1.0
        if obs_mean > EPS and obs_std > EPS:
            alpha = np.std(sim) / obs_std
            beta = np.mean(sim) / obs_mean
            kge = 1.0 - np.sqrt((pearson - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
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
    ss_tot = np.sum((obs_c - mean_o) ** 2)
    r2 = 1.0 - np.sum((obs_c - sim_c) ** 2) / ss_tot if ss_tot > EPS else -1.0

    obs_events = sync["Campo_Relativo"].to_numpy() > detection_threshold
    sim_events = sync["Sim_Relativo"].to_numpy() > detection_threshold
    hits = int(np.sum(obs_events & sim_events))
    misses = int(np.sum(obs_events & ~sim_events))
    false_pos = int(np.sum(~obs_events & sim_events))
    correct_neg = int(np.sum(~obs_events & ~sim_events))
    precision = hits / (hits + false_pos) if hits + false_pos else 0.0
    recall = hits / (hits + misses) if hits + misses else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (hits + correct_neg) / len(sync) if len(sync) else 0.0

    t50_obs = sync.loc[sync["Campo_Acumulado"] >= 0.5, "Fecha"]
    t50_sim = sync.loc[sync["Sim_Acumulado"] >= 0.5, "Fecha"]
    t50_lag = (
        float((t50_sim.iloc[0] - t50_obs.iloc[0]).days)
        if not t50_obs.empty and not t50_sim.empty
        else 365.0
    )
    values = {
        "Pearson_Flujos": pearson,
        "NSE_Flujos": float(nse),
        "KGE_Flujos": float(kge),
        "RMSE_Acumulado": rmse,
        "CCC_Acumulado": ccc,
        "R2_Acumulado": float(r2),
        "Exactitud": float(accuracy),
        "F1_Score": float(f1),
        "Desfase_T50": t50_lag,
        "Hits": hits,
        "Misses": misses,
        "Falsos_Positivos": false_pos,
        "Correctos_Negativos": correct_neg,
    }
    return {k: (v if np.isfinite(v) else zero[k]) for k, v in values.items()}


def objective_score(metrics: Mapping[str, float | int], weights: Mapping[str, float] | None = None) -> float:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    transformed = {
        "KGE_Flujos": np.clip((float(metrics["KGE_Flujos"]) + 1.0) / 2.0, 0.0, 1.0),
        "NSE_Flujos": np.clip((float(metrics["NSE_Flujos"]) + 1.0) / 2.0, 0.0, 1.0),
        "CCC_Acumulado": np.clip((float(metrics["CCC_Acumulado"]) + 1.0) / 2.0, 0.0, 1.0),
        "F1_Score": np.clip(float(metrics["F1_Score"]), 0.0, 1.0),
        "RMSE_Acumulado": np.clip(1.0 - float(metrics["RMSE_Acumulado"]), 0.0, 1.0),
        "Desfase_T50": np.exp(-abs(float(metrics["Desfase_T50"])) / 21.0),
    }
    denominator = sum(max(0.0, float(v)) for v in w.values()) or 1.0
    return float(
        sum(max(0.0, float(w.get(k, 0.0))) * transformed[k] for k in transformed)
        / denominator
    )
