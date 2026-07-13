import numpy as np
import pandas as pd

from predweem_optimizer_fijos import (
    FIXED_PARAMETERS,
    PARAMETER_SPACE,
    optimize_parameters_temporal_cv,
    prepare_field,
    prepare_weather,
    simulate_emergence,
)


class DummyANN:
    def predict(self, X):
        jd = np.asarray(X)[:, 0]
        pulse = 0.90 * np.exp(-0.5 * ((jd - 48.0) / 7.0) ** 2)
        pulse += 0.35 * np.exp(-0.5 * ((jd - 105.0) / 15.0) ** 2)
        return np.clip(pulse, 0, 1), np.cumsum(pulse)


def weather():
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    jd = dates.dayofyear.to_numpy()
    return pd.DataFrame({
        "Fecha": dates,
        "TMAX": 21 + 4 * np.sin(2 * np.pi * jd / 365),
        "TMIN": 9 + 3 * np.sin(2 * np.pi * jd / 365),
        "Prec": np.where(np.arange(len(dates)) % 9 == 0, 18.0, 1.0),
    })


def test_fixed_parameters_are_not_optimizable():
    assert not set(FIXED_PARAMETERS).intersection(PARAMETER_SPACE)
    assert FIXED_PARAMETERS["latencia_jd"] == 45
    assert FIXED_PARAMETERS["ventana_termica"] == 5
    assert FIXED_PARAMETERS["umbral_termoinhibicion"] == 24.0
    assert FIXED_PARAMETERS["umbral_primer_pico"] == 0.70
    assert FIXED_PARAMETERS["lag_dias"] == 0


def test_simulation_respects_fixed_timing():
    params = {name: spec.default for name, spec in PARAMETER_SPACE.items()}
    sim = simulate_emergence(prepare_weather(weather()), DummyANN(), params, cobertura_pct=75)
    assert (sim.loc[sim["Julian_days"] <= 45, "EMERREL"] == 0).all()
    active = sim[sim["EMERREL"] > 0]
    if not active.empty:
        assert active.iloc[0]["Julian_days"] > 45
        assert active.iloc[0]["EMERREL"] > 0.70


def test_optimizer_smoke_with_only_free_parameters():
    w = prepare_weather(weather())
    params = {name: spec.default for name, spec in PARAMETER_SPACE.items()}
    sim = simulate_emergence(w, DummyANN(), params, cobertura_pct=75)
    dates = pd.to_datetime([
        "2026-02-05", "2026-02-20", "2026-03-10",
        "2026-03-28", "2026-04-15", "2026-05-03",
    ])
    flows = []
    start = w["Fecha"].min() - pd.Timedelta(days=1)
    for end in dates:
        flows.append(sim.loc[(sim["Fecha"] > start) & (sim["Fecha"] <= end), "EMERREL"].sum())
        start = end
    field = prepare_field(pd.DataFrame({"FECHA": dates, "PLM2": flows}))
    result = optimize_parameters_temporal_cv(
        w,
        field,
        DummyANN(),
        optimized_parameters=["w_max", "humedad_p50"],
        cobertura_pct=75,
        n_global=12,
        n_local=4,
        n_folds=3,
        min_intervals_per_fold=2,
        seed=7,
    )
    assert set(result["best_params"]) == set(PARAMETER_SPACE)
    assert result["fixed_parameters"] == FIXED_PARAMETERS
    assert not result["results"].empty
