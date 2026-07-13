import numpy as np
import pandas as pd

from predweem_optimizer import (
    optimize_parameters_temporal_cv,
    prepare_field,
    prepare_weather,
    simulate_emergence,
    surface_parameters,
)


class DummyANN:
    def predict(self, X):
        jd = np.asarray(X)[:, 0]
        pulse = 0.82 * np.exp(-0.5 * ((jd - 92.0) / 12.0) ** 2)
        pulse += 0.25 * np.exp(-0.5 * ((jd - 135.0) / 18.0) ** 2)
        return np.clip(pulse, 0, 1), np.cumsum(pulse)


def synthetic_weather():
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    jd = dates.dayofyear.to_numpy()
    return pd.DataFrame({
        "Fecha": dates,
        "TMAX": 22 + 6 * np.sin(2 * np.pi * jd / 365),
        "TMIN": 10 + 4 * np.sin(2 * np.pi * jd / 365),
        "Prec": np.where(np.arange(len(dates)) % 11 == 0, 18.0, 0.8),
    })


def test_surface_curves_match_operational_model():
    assert surface_parameters(0) == (0.85, 0.95)
    assert surface_parameters(70) == (0.25, 0.85)
    assert surface_parameters(100) == (0.10, 0.80)


def test_temporal_cv_single_set_smoke():
    weather = prepare_weather(synthetic_weather())
    params = {
        "w_max": 20.0,
        "cobertura_pct": 70,
        "humedad_p50": 0.30,
        "pendiente_hidrica": 10.0,
        "humedad_corte": 0.10,
        "recarga_relativa": 0.30,
        "latencia_jd": 35,
        "ventana_termica": 5,
        "umbral_termoinhibicion": 29.0,
        "ventana_lluvia": 3,
        "umbral_choque_hidrico": 50.0,
        "fin_choque_jd": 110,
        "techo_choque": 1.0,
        "umbral_primer_pico": 0.25,
        "persistencia_primer_pico": 1,
        "lag_dias": 0,
    }
    sim = simulate_emergence(weather, DummyANN(), params)
    field_dates = pd.to_datetime([
        "2026-02-20", "2026-03-10", "2026-03-28",
        "2026-04-15", "2026-05-03", "2026-05-21",
    ])
    flows = []
    start = weather["Fecha"].min() - pd.Timedelta(days=1)
    for end in field_dates:
        flows.append(sim.loc[(sim["Fecha"] > start) & (sim["Fecha"] <= end), "EMERREL"].sum())
        start = end
    field = prepare_field(pd.DataFrame({"FECHA": field_dates, "PLM2": flows}))
    result = optimize_parameters_temporal_cv(
        weather,
        field,
        DummyANN(),
        optimized_parameters=["w_max", "cobertura_pct", "latencia_jd", "lag_dias"],
        n_global=12,
        n_local=4,
        n_folds=3,
        min_intervals_per_fold=2,
        seed=7,
    )
    assert result["validation_design"] == "temporal_block_cv"
    assert 0 <= result["best_params"]["cobertura_pct"] <= 100
    assert len(result["cv_by_fold"]) == 3
    assert not result["results"].empty
