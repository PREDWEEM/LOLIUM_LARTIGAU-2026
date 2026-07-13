# -*- coding: utf-8 -*-
"""Validación cruzada temporal interna para el único set de Lartigau."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import DEFAULT_OPTIMIZED_PARAMETERS, EPS, PARAMETER_SPACE, prepare_field, prepare_weather
from .model import objective_score, simulate_emergence, validation_metrics
from .search import evaluate_candidate, local_parameter_sets, sample_parameter_sets

DEFAULT_CV_WEIGHTS = {
    "KGE_Flujos": 0.28,
    "NSE_Flujos": 0.22,
    "CCC_Acumulado": 0.20,
    "F1_Score": 0.15,
    "RMSE_Acumulado": 0.15,
    # Un T50 calculado dentro de un bloque no representa el T50 global.
    "Desfase_T50": 0.00,
}


def build_interval_table(field: pd.DataFrame, simulation_start: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_start = pd.Timestamp(simulation_start) - pd.Timedelta(days=1)
    for group, part in field.groupby("Grupo", sort=False):
        previous = first_start
        for _, row in part.sort_values("Fecha").iterrows():
            end = pd.Timestamp(row["Fecha"])
            rows.append({
                "Grupo": str(group),
                "Inicio": previous,
                "Fecha": end,
                "Observado": float(row["Observado"]),
            })
            previous = end
    return pd.DataFrame(rows)


def synchronize_selected_intervals(simulation: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    if intervals is None or intervals.empty:
        return pd.DataFrame()
    sim = simulation.sort_values("Fecha")
    records: list[dict[str, Any]] = []
    for _, row in intervals.sort_values(["Grupo", "Fecha"]).iterrows():
        start = pd.Timestamp(row["Inicio"])
        end = pd.Timestamp(row["Fecha"])
        sim_flow = sim.loc[
            (sim["Fecha"] > start) & (sim["Fecha"] <= end), "EMERREL"
        ].sum()
        records.append({
            "Grupo": str(row["Grupo"]),
            "Inicio": start,
            "Fecha": end,
            "Dias_Intervalo": int((end - start).days),
            "Flujo_Obs_Abs": float(row["Observado"]),
            "Flujo_Sim_Abs": float(sim_flow),
        })
    raw = pd.DataFrame(records)
    frames: list[pd.DataFrame] = []
    for group, part in raw.groupby("Grupo", sort=False):
        part = part.sort_values("Fecha").copy()
        obs_total = float(part["Flujo_Obs_Abs"].sum())
        sim_total = float(part["Flujo_Sim_Abs"].sum())
        part["Acum_Obs_Abs"] = part["Flujo_Obs_Abs"].cumsum()
        part["Acum_Sim_Abs"] = part["Flujo_Sim_Abs"].cumsum()
        part["Campo_Relativo"] = part["Flujo_Obs_Abs"] / obs_total if obs_total > EPS else 0.0
        part["Sim_Relativo"] = part["Flujo_Sim_Abs"] / sim_total if sim_total > EPS else 0.0
        part["Campo_Acumulado"] = part["Acum_Obs_Abs"] / obs_total if obs_total > EPS else 0.0
        part["Sim_Acumulado"] = part["Acum_Sim_Abs"] / sim_total if sim_total > EPS else 0.0
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_temporal_folds(
    field: pd.DataFrame,
    simulation_start: pd.Timestamp,
    n_folds: int = 3,
    min_intervals_per_fold: int = 2,
) -> pd.DataFrame:
    intervals = build_interval_table(field, simulation_start)
    if intervals.empty:
        raise ValueError("No se pudieron construir intervalos de campo.")
    frames: list[pd.DataFrame] = []
    for group, part in intervals.groupby("Grupo", sort=False):
        part = part.sort_values("Fecha").reset_index(drop=True)
        max_folds = len(part) // max(1, int(min_intervals_per_fold))
        folds_for_group = min(int(n_folds), max_folds)
        if folds_for_group < 2:
            raise ValueError(
                f"El grupo {group} requiere al menos {2 * min_intervals_per_fold} fechas para CV temporal."
            )
        for fold_number, indices in enumerate(np.array_split(np.arange(len(part)), folds_for_group), start=1):
            fold = part.iloc[indices].copy()
            fold["Fold"] = f"{group}_B{fold_number}"
            fold["Bloque_Numero"] = fold_number
            frames.append(fold)
    return pd.concat(frames, ignore_index=True)


def evaluate_candidate_temporal_cv(
    weather: pd.DataFrame,
    fold_intervals: pd.DataFrame,
    ann_model: Any,
    params: Mapping[str, float | int],
    latitude: float,
    robustness_penalty: float,
    weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    simulation = simulate_emergence(weather, ann_model, params, latitude)
    cv_weights = DEFAULT_CV_WEIGHTS if weights is None else weights
    fold_records: list[dict[str, Any]] = []
    sync_frames: list[pd.DataFrame] = []
    for fold_name, intervals in fold_intervals.groupby("Fold", sort=False):
        sync = synchronize_selected_intervals(simulation, intervals)
        metrics = validation_metrics(sync)
        score = objective_score(metrics, cv_weights)
        fold_records.append({
            "Fold": fold_name,
            "Grupo": str(intervals["Grupo"].iloc[0]),
            "N_Intervalos": int(len(intervals)),
            "Fecha_Inicio": intervals["Inicio"].min(),
            "Fecha_Fin": intervals["Fecha"].max(),
            "Score": score,
            **metrics,
        })
        if not sync.empty:
            sync = sync.copy()
            sync["Fold"] = fold_name
            sync_frames.append(sync)
    fold_df = pd.DataFrame(fold_records)
    scores = fold_df["Score"].to_numpy(float) if not fold_df.empty else np.array([0.0])
    summary = {
        "Score_CV": float(np.mean(scores) - robustness_penalty * np.std(scores)),
        "Score_CV_Medio": float(np.mean(scores)),
        "Score_CV_SD": float(np.std(scores)),
        "Score_CV_Peor_Bloque": float(np.min(scores)),
        "N_Bloques": int(len(fold_df)),
    }
    for metric in [
        "KGE_Flujos",
        "NSE_Flujos",
        "CCC_Acumulado",
        "RMSE_Acumulado",
        "F1_Score",
        "Desfase_T50",
    ]:
        summary[f"{metric}_Media"] = float(fold_df[metric].mean()) if not fold_df.empty else np.nan
    sync_all = pd.concat(sync_frames, ignore_index=True) if sync_frames else pd.DataFrame()
    return summary, fold_df, sync_all


def optimize_parameters_temporal_cv(
    weather_data: pd.DataFrame,
    field_data: pd.DataFrame,
    ann_model: Any,
    *,
    optimized_parameters: Sequence[str] = DEFAULT_OPTIMIZED_PARAMETERS,
    fixed_parameters: Mapping[str, float | int] | None = None,
    n_global: int = 400,
    n_local: int = 200,
    seed: int = 42,
    latitude: float = -38.6166,
    robustness_penalty: float = 0.15,
    n_folds: int = 3,
    min_intervals_per_fold: int = 2,
    weights: Mapping[str, float] | None = None,
    top_seeds: int = 5,
) -> dict[str, Any]:
    unknown = [name for name in optimized_parameters if name not in PARAMETER_SPACE]
    if unknown:
        raise ValueError(f"Parámetros desconocidos: {', '.join(unknown)}")
    weather = prepare_weather(weather_data)
    field = prepare_field(field_data) if "Observado" not in field_data.columns else field_data.copy()
    fold_intervals = make_temporal_folds(
        field, weather["Fecha"].min(), n_folds=n_folds,
        min_intervals_per_fold=min_intervals_per_fold,
    )
    rng = np.random.default_rng(seed)
    candidates = sample_parameter_sets(
        max(1, int(n_global)), optimized_parameters, rng, fixed_parameters
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        summary, _, _ = evaluate_candidate_temporal_cv(
            weather, fold_intervals, ann_model, candidate, latitude,
            robustness_penalty, weights,
        )
        rows.append({**candidate, **summary, "Etapa": "global"})
    global_df = pd.DataFrame(rows).sort_values("Score_CV", ascending=False)
    seeds = global_df.head(max(1, int(top_seeds)))[list(PARAMETER_SPACE)].to_dict("records")
    if n_local > 0:
        local_candidates = local_parameter_sets(
            seeds,
            max(1, int(np.ceil(n_local / len(seeds)))),
            optimized_parameters,
            rng,
        )[: int(n_local)]
        for candidate in local_candidates:
            summary, _, _ = evaluate_candidate_temporal_cv(
                weather, fold_intervals, ann_model, candidate, latitude,
                robustness_penalty, weights,
            )
            rows.append({**candidate, **summary, "Etapa": "local"})
    results = pd.DataFrame(rows)
    results = results.drop_duplicates(subset=list(PARAMETER_SPACE), keep="first")
    results = results.sort_values(
        ["Score_CV", "Score_CV_Peor_Bloque"], ascending=False
    ).reset_index(drop=True)
    best_params = {name: results.iloc[0][name] for name in PARAMETER_SPACE}
    for name, spec in PARAMETER_SPACE.items():
        best_params[name] = int(best_params[name]) if spec.integer else float(best_params[name])
    cv_summary, cv_by_fold, cv_sync = evaluate_candidate_temporal_cv(
        weather, fold_intervals, ann_model, best_params, latitude,
        robustness_penalty, weights,
    )
    apparent_summary, apparent_by_group, apparent_sync = evaluate_candidate(
        weather, field, ann_model, best_params, latitude, 0.0, None
    )
    return {
        "validation_design": "temporal_block_cv",
        "best_params": best_params,
        "best_summary": cv_summary,
        "results": results,
        "cv_by_fold": cv_by_fold,
        "cv_sync": cv_sync,
        "fold_intervals": fold_intervals,
        "apparent_summary": apparent_summary,
        "apparent_by_group": apparent_by_group,
        "apparent_sync": apparent_sync,
        "weather_data": weather,
        "field_data": field,
    }
