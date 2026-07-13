# -*- coding: utf-8 -*-
"""Muestreo paramétrico y evaluación completa del único set de Lartigau."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import PARAMETER_SPACE, default_parameters
from .model import objective_score, simulate_emergence, synchronize_intervals, validation_metrics


def _cast_parameter(name: str, value: float) -> float | int:
    spec = PARAMETER_SPACE[name]
    clipped = np.clip(value, spec.low, spec.high)
    return int(round(clipped)) if spec.integer else float(clipped)


def sample_parameter_sets(
    n: int,
    optimized_parameters: Sequence[str],
    rng: np.random.Generator,
    fixed_parameters: Mapping[str, float | int] | None = None,
) -> list[dict[str, float | int]]:
    fixed = default_parameters()
    if fixed_parameters:
        fixed.update(fixed_parameters)
    draws: dict[str, np.ndarray] = {}
    for name in optimized_parameters:
        spec = PARAMETER_SPACE[name]
        bins = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(bins)
        draws[name] = spec.low + bins * (spec.high - spec.low)
    candidates: list[dict[str, float | int]] = []
    for i in range(n):
        candidate = dict(fixed)
        for name in optimized_parameters:
            candidate[name] = _cast_parameter(name, float(draws[name][i]))
        candidates.append(candidate)
    return candidates


def local_parameter_sets(
    seeds: Sequence[Mapping[str, float | int]],
    n_per_seed: int,
    optimized_parameters: Sequence[str],
    rng: np.random.Generator,
    scale: float = 0.10,
) -> list[dict[str, float | int]]:
    candidates: list[dict[str, float | int]] = []
    for seed in seeds:
        for _ in range(n_per_seed):
            candidate = dict(seed)
            for name in optimized_parameters:
                spec = PARAMETER_SPACE[name]
                sigma = (spec.high - spec.low) * scale
                candidate[name] = _cast_parameter(
                    name, float(seed[name]) + rng.normal(0.0, sigma)
                )
            candidates.append(candidate)
    return candidates


def evaluate_candidate(
    weather: pd.DataFrame,
    field: pd.DataFrame,
    ann_model: Any,
    params: Mapping[str, float | int],
    latitude: float,
    robustness_penalty: float = 0.0,
    weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    simulation = simulate_emergence(weather, ann_model, params, latitude)
    records: list[dict[str, Any]] = []
    sync_frames: list[pd.DataFrame] = []
    for group, group_field in field.groupby("Grupo", sort=False):
        sync = synchronize_intervals(simulation, group_field.reset_index(drop=True))
        metrics = validation_metrics(sync)
        score = objective_score(metrics, weights)
        records.append({"Grupo": str(group), "Score": score, **metrics})
        if not sync.empty:
            sync = sync.copy()
            sync["Grupo"] = str(group)
            sync_frames.append(sync)
    by_group = pd.DataFrame(records)
    scores = by_group["Score"].to_numpy(float) if not by_group.empty else np.array([0.0])
    summary = {
        "Score_Calibracion": float(np.mean(scores) - robustness_penalty * np.std(scores)),
        "Score_Medio": float(np.mean(scores)),
        "Score_SD": float(np.std(scores)),
        "Score_Peor_Grupo": float(np.min(scores)),
        "N_Grupos": int(len(by_group)),
    }
    for metric in [
        "KGE_Flujos",
        "NSE_Flujos",
        "CCC_Acumulado",
        "RMSE_Acumulado",
        "F1_Score",
        "Desfase_T50",
    ]:
        summary[f"{metric}_Media"] = (
            float(by_group[metric].mean()) if not by_group.empty else np.nan
        )
    sync_all = pd.concat(sync_frames, ignore_index=True) if sync_frames else pd.DataFrame()
    return summary, by_group, sync_all
