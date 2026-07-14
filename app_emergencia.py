# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.9.26 — LOLIUM LARTIGAU 2026
# PREDWEEM by GUILLERMO R. CHANTRE
#
# Criterios principales:
# - ANN con meteorología observada: JD, TMAX, TMIN y precipitación.
# - La cobertura modifica Ke y el balance hídrico.
# - El modulador térmico se conserva como variable diagnóstica.
# - Latencia fija hasta JD 45.
# - Termoinhibición: media móvil de 5 días >= 24 °C.
# - Choque hídrico: precipitación acumulada de 3 días >= 45 mm,
#   entre JD 46 y JD 110.
# - Inicio de campaña: primer día con EMERREL > 0.70.
# - Sin fecha objetivo y sin desplazamiento temporal artificial.
# ===============================================================

from __future__ import annotations

import base64
import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------
APP_VERSION = "vK4.9.26"
LATITUD_LARTIGAU = -38.6166

LATENCIA_JD = 45
VENTANA_TERMICA_DIAS = 5
UMBRAL_TERMINHIBICION = 24.0

VENTANA_LLUVIA_DIAS = 3
UMBRAL_CHOQUE_HIDRICO_MM = 45.0
FIN_CHOQUE_HIDRICO_JD = 110
TECHO_CHOQUE_HIDRICO = 1.0

UMBRAL_PRIMER_PICO = 0.70
PERSISTENCIA_PRIMER_PICO_DIAS = 1

WMAX_PREDETERMINADO = 18.816
COBERTURA_PREDETERMINADA = 75

P50_HIDRICO = 0.30
PENDIENTE_HIDRICA = 10.0
CORTE_HIDRICO = 0.20

T_BASE_PREDETERMINADA = 2.0
T_OPTIMA_PREDETERMINADA = 20.0
T_CRITICA_PREDETERMINADA = 30.0
TT_CONTROL_PREDETERMINADO = 600
TT_LIMITE_PREDETERMINADO = 800

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="PREDWEEM Lartigau Integral",
    page_icon="🌾",
    layout="wide",
)


# ---------------------------------------------------------------
# 2. ESTILO
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #f8fafc; }
        [data-testid="stSidebar"] {
            background-color: #dcfce7;
            border-right: 1px solid #bbf7d0;
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] p {
            color: #166534 !important;
        }
        .stMetric {
            background-color: #ffffff;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def set_background(filename: str) -> None:
    path = BASE / filename
    if not path.exists():
        return
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url(data:image/png;base64,{encoded});
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except OSError:
        pass


set_background("fondo_predweem_v3.png")


# ---------------------------------------------------------------
# 3. MODELO ANN Y ARCHIVOS
# ---------------------------------------------------------------
class PracticalANNModel:
    def __init__(
        self,
        input_weights: np.ndarray,
        input_bias: np.ndarray,
        output_weights: np.ndarray,
        output_bias: np.ndarray,
    ) -> None:
        self.IW = np.asarray(input_weights, dtype=float)
        self.bIW = np.asarray(input_bias, dtype=float)
        self.LW = np.asarray(output_weights, dtype=float)
        self.bLW = np.asarray(output_bias, dtype=float).reshape(-1)

        self.input_min = np.array([1.0, 0.0, -7.0, 0.0])
        self.input_max = np.array([300.0, 41.0, 25.5, 84.0])

    def normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        denominator = self.input_max - self.input_min
        return 2.0 * (values - self.input_min) / denominator - 1.0

    def predict(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized = self.normalize(values)
        hidden = np.tanh(normalized @ self.IW + self.bIW)
        linear_output = (hidden @ self.LW.T).reshape(-1)
        bias = float(self.bLW[0]) if self.bLW.size else 0.0
        emergence = (np.tanh(linear_output + bias) + 1.0) / 2.0
        emergence = np.clip(emergence, 0.0, 1.0)
        return emergence, np.cumsum(emergence)


@st.cache_resource
def load_models() -> tuple[PracticalANNModel | None, Any | None]:
    required = {
        "IW.npy": BASE / "IW.npy",
        "bias_IW.npy": BASE / "bias_IW.npy",
        "LW.npy": BASE / "LW.npy",
        "bias_out.npy": BASE / "bias_out.npy",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        st.error(
            "Faltan archivos del modelo neuronal: "
            + ", ".join(missing)
            + "."
        )
        return None, None

    try:
        ann = PracticalANNModel(
            np.load(required["IW.npy"]),
            np.load(required["bias_IW.npy"]),
            np.load(required["LW.npy"]),
            np.load(required["bias_out.npy"]),
        )
    except (OSError, ValueError) as exc:
        st.error(f"No se pudo cargar la ANN: {exc}")
        return None, None

    cluster_model = None
    cluster_path = BASE / "modelo_clusters_k3.pkl"
    if cluster_path.exists():
        try:
            with cluster_path.open("rb") as handle:
                cluster_model = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError):
            cluster_model = None

    return ann, cluster_model


def read_table(source: Any) -> pd.DataFrame:
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    return pd.read_csv(source)


def load_data(uploaded_file: Any, default_stem: str) -> pd.DataFrame | None:
    if uploaded_file is not None:
        try:
            return read_table(uploaded_file)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo cargado: {exc}")
            return None

    for suffix in (".csv", ".xlsx", ".xls"):
        candidate = BASE / f"{default_stem}{suffix}"
        if candidate.exists():
            try:
                return read_table(candidate)
            except Exception as exc:
                st.warning(f"No se pudo leer {candidate.name}: {exc}")

    return None


# ---------------------------------------------------------------
# 4. FUNCIONES ECOFISIOLÓGICAS
# ---------------------------------------------------------------
def calculate_tt_scalar(
    temperature: float,
    t_base: float,
    t_optimum: float,
    t_critical: float,
) -> float:
    if temperature <= t_base:
        return 0.0
    if temperature <= t_optimum:
        return temperature - t_base
    if temperature < t_critical:
        return (
            (temperature - t_base)
            * ((t_critical - temperature) / (t_critical - t_optimum))
        )
    return 0.0


def calculate_et0_hargreaves(
    julian_day: np.ndarray,
    tmax: np.ndarray,
    tmin: np.ndarray,
    latitude: float = LATITUD_LARTIGAU,
) -> np.ndarray:
    julian_day = np.asarray(julian_day, dtype=float)
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)

    latitude_radians = np.radians(latitude)
    inverse_distance = (
        1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * julian_day)
    )
    declination = 0.409 * np.sin(
        2.0 * np.pi / 365.0 * julian_day - 1.39
    )
    sunset_angle = np.arccos(
        np.clip(
            -np.tan(latitude_radians) * np.tan(declination),
            -1.0,
            1.0,
        )
    )
    extraterrestrial_radiation = (
        (24.0 * 60.0 / np.pi)
        * 0.0820
        * inverse_distance
        * (
            sunset_angle
            * np.sin(latitude_radians)
            * np.sin(declination)
            + np.cos(latitude_radians)
            * np.cos(declination)
            * np.sin(sunset_angle)
        )
    )
    radiation_mm = extraterrestrial_radiation / 2.45
    mean_temperature = (tmax + tmin) / 2.0
    thermal_range = np.maximum(tmax - tmin, 0.0)

    return np.maximum(
        0.0023
        * radiation_mm
        * (mean_temperature + 17.8)
        * np.sqrt(thermal_range),
        0.0,
    )


def surface_parameters(coverage_percent: float) -> tuple[float, float]:
    coverage = float(np.clip(coverage_percent, 0.0, 100.0))
    reference_coverage = [0.0, 30.0, 70.0, 100.0]

    ke_value = float(
        np.interp(
            coverage,
            reference_coverage,
            [0.85, 0.50, 0.25, 0.10],
        )
    )
    thermal_modulator = float(
        np.interp(
            coverage,
            reference_coverage,
            [0.95, 0.90, 0.85, 0.80],
        )
    )
    return ke_value, thermal_modulator


def surface_water_balance(
    precipitation: np.ndarray,
    et0: np.ndarray,
    w_max: float,
    ke_soil: float,
) -> np.ndarray:
    precipitation = np.asarray(precipitation, dtype=float)
    et0 = np.asarray(et0, dtype=float)

    water = np.zeros(len(precipitation), dtype=float)
    if len(water) == 0:
        return water

    water[0] = float(w_max) / 2.0
    for index in range(1, len(water)):
        actual_evaporation = et0[index] * float(ke_soil)
        water[index] = np.clip(
            water[index - 1]
            + precipitation[index]
            - actual_evaporation,
            0.0,
            float(w_max),
        )
    return water


def apply_first_peak_filter(
    dataframe: pd.DataFrame,
    threshold: float = UMBRAL_PRIMER_PICO,
) -> tuple[pd.DataFrame, int | None]:
    """
    Habilita la campaña en el primer día con EMERREL > threshold.

    No utiliza fecha objetivo, lag, interpolación temporal ni información
    futura de campo.
    """
    dataframe = dataframe.copy()
    dataframe["EMERREL_ANTES_FILTRO_PRIMER_PICO"] = dataframe[
        "EMERREL"
    ].copy()

    above_threshold = dataframe["EMERREL"].gt(float(threshold))
    candidates = dataframe.index[above_threshold].tolist()

    if candidates:
        first_peak_index = int(candidates[0])
        dataframe["Primer_Pico_Habilitado"] = (
            dataframe.index >= first_peak_index
        )
        dataframe.loc[
            dataframe.index < first_peak_index,
            "EMERREL",
        ] = 0.0
    else:
        first_peak_index = None
        dataframe["Primer_Pico_Habilitado"] = False
        dataframe["EMERREL"] = 0.0

    dataframe["Supera_Umbral_Primer_Pico"] = above_threshold
    dataframe["Persistencia_Primer_Pico_Dias"] = (
        PERSISTENCIA_PRIMER_PICO_DIAS
    )
    return dataframe, first_peak_index


# ---------------------------------------------------------------
# 5. PREPARACIÓN Y SIMULACIÓN
# ---------------------------------------------------------------
def canonicalize_weather(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("El archivo meteorológico está vacío.")

    data = raw.copy()
    data.columns = [str(column).strip().upper() for column in data.columns]
    data = data.rename(
        columns={
            "FECHA": "Fecha",
            "DATE": "Fecha",
            "DATETIME": "Fecha",
            "PREC": "Prec",
            "PRECIPITACION": "Prec",
            "PRECIPITACIÓN": "Prec",
            "LLUVIA": "Prec",
        }
    )

    required = ["Fecha", "TMAX", "TMIN", "Prec"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            "Faltan columnas meteorológicas: " + ", ".join(missing)
        )

    data["Fecha"] = pd.to_datetime(data["Fecha"], errors="coerce")
    for column in ("TMAX", "TMIN", "Prec"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.dropna(subset=required)
        .sort_values("Fecha")
        .drop_duplicates("Fecha", keep="last")
        .reset_index(drop=True)
    )
    data["Prec"] = data["Prec"].clip(lower=0.0)

    if len(data) < 30:
        raise ValueError(
            "Se requieren al menos 30 días meteorológicos válidos."
        )
    return data


def canonicalize_field(raw: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if raw is None or raw.empty:
        raise ValueError("El archivo de campo está vacío.")

    data = raw.copy()
    date_column = (
        "FECHA"
        if "FECHA" in data.columns
        else "Fecha"
        if "Fecha" in data.columns
        else data.columns[0]
    )

    value_candidates = [
        "PLM2",
        "EMERGENCIA",
        "EMERREL",
        "OBSERVADO",
        "CONTEO",
    ]
    value_column = next(
        (column for column in value_candidates if column in data.columns),
        None,
    )
    if value_column is None:
        if len(data.columns) < 2:
            raise ValueError(
                "No se encontró una columna de emergencia observada."
            )
        value_column = data.columns[1]

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )
    data[value_column] = pd.to_numeric(
        data[value_column],
        errors="coerce",
    )
    data = (
        data.dropna(subset=[date_column, value_column])
        .sort_values(date_column)
        .drop_duplicates(date_column, keep="last")
        .reset_index(drop=True)
    )
    data[value_column] = data[value_column].clip(lower=0.0)

    maximum = float(data[value_column].max())
    data["Campo_Normalizado"] = (
        data[value_column] / maximum if maximum > 0.0 else 0.0
    )
    return data, date_column, value_column


def simulate_emergence(
    raw_weather: pd.DataFrame,
    ann_model: PracticalANNModel,
    coverage_percent: int,
    w_max: float,
    thermoinhibition_threshold: float = UMBRAL_TERMINHIBICION,
    hydric_shock_threshold: float = UMBRAL_CHOQUE_HIDRICO_MM,
) -> tuple[pd.DataFrame, int | None]:
    data = canonicalize_weather(raw_weather)

    data["Julian_days"] = data["Fecha"].dt.dayofyear
    data["Tmedia_aire"] = (data["TMAX"] + data["TMIN"]) / 2.0
    thermal_amplitude = (data["TMAX"] - data["TMIN"]) / 2.0

    ke_value, thermal_modulator = surface_parameters(coverage_percent)
    data["Cobertura_Rastrojo"] = int(coverage_percent)
    data["Ke_Suelo"] = ke_value
    data["Modulador_Termico_Diagnostico"] = thermal_modulator

    # Sólo diagnóstico de microclima; no son entradas de la ANN.
    data["TMAX_suelo_diagnostica"] = (
        data["Tmedia_aire"]
        + thermal_amplitude * thermal_modulator
    )
    data["TMIN_suelo_diagnostica"] = (
        data["Tmedia_aire"]
        - thermal_amplitude * thermal_modulator
    )

    # ANN desacoplada de la cobertura.
    neural_inputs = data[
        ["Julian_days", "TMAX", "TMIN", "Prec"]
    ].to_numpy(float)
    raw_emergence, _ = ann_model.predict(neural_inputs)
    data["EMERREL_RAW_ANN"] = np.clip(raw_emergence, 0.0, 1.0)
    data["EMERREL"] = data["EMERREL_RAW_ANN"].copy()

    # Choque hídrico conservado.
    data["Prec_3d"] = data["Prec"].rolling(
        window=VENTANA_LLUVIA_DIAS,
        min_periods=1,
    ).sum()
    hydric_shock = (
        (data["Julian_days"] > LATENCIA_JD)
        & (data["Julian_days"] <= FIN_CHOQUE_HIDRICO_JD)
        & (data["Prec_3d"] >= float(hydric_shock_threshold))
    )
    data.loc[hydric_shock, "EMERREL"] = np.maximum(
        data.loc[hydric_shock, "EMERREL"],
        TECHO_CHOQUE_HIDRICO,
    )
    data["Choque_Hidrico"] = hydric_shock

    # Balance hídrico controlado por cobertura mediante Ke.
    data["ET0"] = calculate_et0_hargreaves(
        data["Julian_days"].to_numpy(),
        data["TMAX"].to_numpy(),
        data["TMIN"].to_numpy(),
        LATITUD_LARTIGAU,
    )
    data["W_superficial"] = surface_water_balance(
        data["Prec"].to_numpy(),
        data["ET0"].to_numpy(),
        float(w_max),
        ke_value,
    )
    relative_water = data["W_superficial"] / max(float(w_max), 1e-12)
    data["Humedad_Relativa"] = relative_water

    hydric_exponent = np.clip(
        -PENDIENTE_HIDRICA * (relative_water - P50_HIDRICO),
        -60.0,
        60.0,
    )
    data["Hydric_Factor"] = 1.0 / (1.0 + np.exp(hydric_exponent))
    data["EMERREL"] *= data["Hydric_Factor"]

    data.loc[
        relative_water < CORTE_HIDRICO,
        "EMERREL",
    ] = 0.0

    # Conserva el criterio operativo original de recarga por lluvia diaria.
    data["Lluvia_Recarga"] = (
        data["Prec"] >= float(w_max)
    ).cummax()
    data.loc[~data["Lluvia_Recarga"], "EMERREL"] = 0.0

    # Termoinhibición fija con temperatura media del aire.
    data["Tmedia_5d"] = data["Tmedia_aire"].rolling(
        window=VENTANA_TERMICA_DIAS,
        min_periods=1,
    ).mean()
    data["Termoinhibida"] = (
        data["Tmedia_5d"] >= float(thermoinhibition_threshold)
    )
    data.loc[data["Termoinhibida"], "EMERREL"] = 0.0

    # Latencia fija al final del conjunto de filtros.
    data.loc[
        data["Julian_days"] <= LATENCIA_JD,
        "EMERREL",
    ] = 0.0

    data["EMERREL"] = np.clip(data["EMERREL"], 0.0, 1.0)
    data, first_peak_index = apply_first_peak_filter(
        data,
        threshold=UMBRAL_PRIMER_PICO,
    )

    data["EMERAC"] = data["EMERREL"].cumsum()
    total_emergence = float(data["EMERREL"].sum())
    data["EMERAC_NORMALIZADA"] = (
        data["EMERAC"] / total_emergence
        if total_emergence > 0.0
        else 0.0
    )
    return data, first_peak_index


# ---------------------------------------------------------------
# 6. VALIDACIÓN EVENT-TO-EVENT
# ---------------------------------------------------------------
def synchronize_real_intervals(
    simulation: pd.DataFrame,
    field: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> pd.DataFrame:
    field = field.sort_values(date_column).reset_index(drop=True).copy()
    if field.empty:
        return pd.DataFrame()

    field["Campo_Acum_Abs"] = field[value_column].cumsum()
    simulation_start = (
        pd.Timestamp(simulation["Fecha"].min()) - pd.Timedelta(days=1)
    )
    records: list[dict[str, Any]] = []

    for row_index, row in field.iterrows():
        interval_start = (
            simulation_start
            if row_index == 0
            else pd.Timestamp(field.iloc[row_index - 1][date_column])
        )
        interval_end = pd.Timestamp(row[date_column])
        observed_flow = float(row[value_column])

        interval_mask = (
            (simulation["Fecha"] > interval_start)
            & (simulation["Fecha"] <= interval_end)
        )
        simulated_flow = float(
            simulation.loc[interval_mask, "EMERREL"].sum()
        )
        simulated_accumulated = float(
            simulation.loc[
                simulation["Fecha"] <= interval_end,
                "EMERREL",
            ].sum()
        )

        records.append(
            {
                "Inicio": interval_start,
                "Fecha": interval_end,
                "Dias_Intervalo": int(
                    (interval_end - interval_start).days
                ),
                "Flujo_Obs_Abs": observed_flow,
                "Flujo_Sim_Abs": simulated_flow,
                "Acum_Obs_Abs": float(row["Campo_Acum_Abs"]),
                "Acum_Sim_Abs": simulated_accumulated,
            }
        )

    result = pd.DataFrame(records)
    total_observed = float(result["Flujo_Obs_Abs"].sum())
    last_field_date = pd.Timestamp(field[date_column].max())
    total_simulated = float(
        simulation.loc[
            simulation["Fecha"] <= last_field_date,
            "EMERREL",
        ].sum()
    )

    result["Campo_Relativo"] = (
        result["Flujo_Obs_Abs"] / total_observed
        if total_observed > 0.0
        else 0.0
    )
    result["Sim_Relativo"] = (
        result["Flujo_Sim_Abs"] / total_simulated
        if total_simulated > 0.0
        else 0.0
    )
    result["Campo_Acumulado"] = (
        result["Acum_Obs_Abs"] / total_observed
        if total_observed > 0.0
        else 0.0
    )
    result["Sim_Acumulado"] = (
        result["Acum_Sim_Abs"] / total_simulated
        if total_simulated > 0.0
        else 0.0
    )
    return result


def safe_correlation(observed: np.ndarray, simulated: np.ndarray) -> float:
    if (
        len(observed) < 2
        or np.std(observed) <= 1e-12
        or np.std(simulated) <= 1e-12
    ):
        return 0.0
    correlation = float(np.corrcoef(observed, simulated)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def validation_metrics(
    synchronized: pd.DataFrame,
    detection_threshold: float = 0.05,
) -> dict[str, float | int]:
    if synchronized is None or synchronized.empty:
        return {
            "Pearson_Flujos": 0.0,
            "NSE_Flujos": 0.0,
            "KGE_Flujos": 0.0,
            "RMSE_Acumulado": 0.0,
            "CCC_Acumulado": 0.0,
            "R2_Acumulado": 0.0,
            "Exactitud_Global": 0.0,
            "F1_Score_Coincidencia": 0.0,
            "Hits": 0,
            "Misses": 0,
            "Falsos_Positivos": 0,
            "Correctos_Negativos": 0,
        }

    observed_flow = synchronized["Campo_Relativo"].to_numpy(float)
    simulated_flow = synchronized["Sim_Relativo"].to_numpy(float)

    pearson = safe_correlation(observed_flow, simulated_flow)
    observed_mean = float(np.mean(observed_flow))
    observed_variance = float(
        np.sum((observed_flow - observed_mean) ** 2)
    )
    nse = (
        1.0
        - float(
            np.sum((simulated_flow - observed_flow) ** 2)
            / observed_variance
        )
        if observed_variance > 1e-12
        else 0.0
    )

    observed_std = float(np.std(observed_flow))
    simulated_std = float(np.std(simulated_flow))
    if observed_mean > 1e-12 and observed_std > 1e-12:
        alpha = simulated_std / observed_std
        beta = float(np.mean(simulated_flow)) / observed_mean
        kge = 1.0 - float(
            np.sqrt(
                (pearson - 1.0) ** 2
                + (alpha - 1.0) ** 2
                + (beta - 1.0) ** 2
            )
        )
    else:
        kge = 0.0

    observed_accumulated = synchronized[
        "Campo_Acumulado"
    ].to_numpy(float)
    simulated_accumulated = synchronized[
        "Sim_Acumulado"
    ].to_numpy(float)

    rmse = float(
        np.sqrt(
            np.mean(
                (observed_accumulated - simulated_accumulated) ** 2
            )
        )
    )

    observed_acc_mean = float(np.mean(observed_accumulated))
    simulated_acc_mean = float(np.mean(simulated_accumulated))
    observed_acc_var = float(np.var(observed_accumulated))
    simulated_acc_var = float(np.var(simulated_accumulated))
    covariance = float(
        np.mean(
            (observed_accumulated - observed_acc_mean)
            * (simulated_accumulated - simulated_acc_mean)
        )
    )
    ccc_denominator = (
        observed_acc_var
        + simulated_acc_var
        + (observed_acc_mean - simulated_acc_mean) ** 2
    )
    ccc = (
        2.0 * covariance / ccc_denominator
        if ccc_denominator > 1e-12
        else 0.0
    )

    ss_residual = float(
        np.sum(
            (observed_accumulated - simulated_accumulated) ** 2
        )
    )
    ss_total = float(
        np.sum(
            (observed_accumulated - observed_acc_mean) ** 2
        )
    )
    r_squared = (
        1.0 - ss_residual / ss_total
        if ss_total > 1e-12
        else 0.0
    )

    observed_events = (
        synchronized["Campo_Relativo"] > detection_threshold
    )
    simulated_events = (
        synchronized["Sim_Relativo"] > detection_threshold
    )

    hits = int(np.sum(observed_events & simulated_events))
    misses = int(np.sum(observed_events & ~simulated_events))
    false_positives = int(
        np.sum(~observed_events & simulated_events)
    )
    correct_negatives = int(
        np.sum(~observed_events & ~simulated_events)
    )

    total_intervals = len(synchronized)
    accuracy = (
        (hits + correct_negatives) / total_intervals
        if total_intervals > 0
        else 0.0
    )
    precision = (
        hits / (hits + false_positives)
        if hits + false_positives > 0
        else 0.0
    )
    recall = (
        hits / (hits + misses)
        if hits + misses > 0
        else 0.0
    )
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )

    return {
        "Pearson_Flujos": pearson,
        "NSE_Flujos": nse,
        "KGE_Flujos": kge,
        "RMSE_Acumulado": rmse,
        "CCC_Acumulado": ccc,
        "R2_Acumulado": r_squared,
        "Exactitud_Global": accuracy,
        "F1_Score_Coincidencia": f1_score,
        "Hits": hits,
        "Misses": misses,
        "Falsos_Positivos": false_positives,
        "Correctos_Negativos": correct_negatives,
    }


# ---------------------------------------------------------------
# 7. INTERFAZ
# ---------------------------------------------------------------
ann_model, cluster_model = load_models()

st.title(
    "🌾 PREDWEEM LOLIUM — LARTIGAU "
    "(BA, lat=-38.6166; lon=-61.7000)"
)
st.caption(
    f"{APP_VERSION} · primer pico resultante de la simulación · "
    "sin fecha objetivo y sin lag"
)

with st.expander("📂 1. Datos del lote", expanded=True):
    upload_column, coverage_column = st.columns(2)

    with upload_column:
        weather_upload = st.file_uploader(
            "Meteorología diaria",
            type=["csv", "xlsx", "xls"],
        )
        field_upload = st.file_uploader(
            "Observaciones de campo",
            type=["csv", "xlsx", "xls"],
        )

    with coverage_column:
        coverage_percent = st.slider(
            "Cobertura de rastrojo (%)",
            min_value=0,
            max_value=100,
            value=COBERTURA_PREDETERMINADA,
            step=1,
            help=(
                "La cobertura modifica Ke y el balance hídrico. "
                "No modifica las entradas térmicas de la ANN."
            ),
        )
        ke_value, thermal_modulator = surface_parameters(
            coverage_percent
        )

        parameter_1, parameter_2 = st.columns(2)
        parameter_1.metric("Ke aplicado", f"{ke_value:.3f}")
        parameter_2.metric(
            "Modulador térmico",
            f"{thermal_modulator:.3f}",
            help="Variable diagnóstica; no ingresa en la ANN.",
        )

weather_raw = load_data(weather_upload, "meteo_daily")
field_raw = load_data(field_upload, "LARTIGAU_campo")

st.sidebar.markdown("## ⚙️ Fisiología fija")
st.sidebar.info(
    f"Latencia hasta JD {LATENCIA_JD} · "
    f"termoinhibición {VENTANA_TERMICA_DIAS} días ≥ "
    f"{UMBRAL_TERMINHIBICION:.1f} °C · "
    f"inicio con 1 día de EMERREL > {UMBRAL_PRIMER_PICO:.2f}."
)
st.sidebar.success(
    "La fecha del primer pico surge de la simulación. "
    "No se aplica lag temporal."
)

st.sidebar.markdown("## 💧 Balance hídrico")
w_max_value = st.sidebar.number_input(
    "Capacidad superficial Wmax (mm)",
    min_value=5.0,
    max_value=60.0,
    value=WMAX_PREDETERMINADO,
    step=0.1,
    format="%.3f",
    help=(
        "Con la meteorología 2026 analizada, valores de aproximadamente "
        "18–20 mm y cobertura ≥75 % produjeron un pico en febrero."
    ),
)

thermoinhibition_threshold = st.sidebar.number_input(
    "Umbral termoinhibición (°C)",
    min_value=15.0,
    max_value=35.0,
    value=UMBRAL_TERMINHIBICION,
    step=0.5,
)
hydric_shock_threshold = st.sidebar.number_input(
    "Choque hídrico en 3 días (mm)",
    min_value=20.0,
    max_value=100.0,
    value=UMBRAL_CHOQUE_HIDRICO_MM,
    step=1.0,
)

st.sidebar.markdown("## 🌡️ Tiempo térmico")
t_base = st.sidebar.number_input(
    "Temperatura base",
    value=T_BASE_PREDETERMINADA,
    step=0.5,
)
t_optimum = st.sidebar.number_input(
    "Temperatura óptima",
    value=T_OPTIMA_PREDETERMINADA,
    step=1.0,
)
t_critical = st.sidebar.number_input(
    "Temperatura crítica",
    value=T_CRITICA_PREDETERMINADA,
    step=1.0,
)
tt_control = st.sidebar.number_input(
    "TT para control postemergente",
    min_value=0,
    value=TT_CONTROL_PREDETERMINADO,
    step=10,
)
tt_limit = st.sidebar.number_input(
    "Límite de ventana",
    min_value=0,
    value=TT_LIMITE_PREDETERMINADO,
    step=10,
)

if ann_model is None:
    st.stop()

if weather_raw is None:
    st.warning(
        "Cargue un archivo meteorológico o incorpore "
        "`meteo_daily.csv` al repositorio."
    )
    st.stop()

try:
    simulation, first_peak_index = simulate_emergence(
        weather_raw,
        ann_model,
        coverage_percent=coverage_percent,
        w_max=float(w_max_value),
        thermoinhibition_threshold=float(
            thermoinhibition_threshold
        ),
        hydric_shock_threshold=float(hydric_shock_threshold),
    )
except Exception as exc:
    st.exception(exc)
    st.stop()

simulation["DG"] = simulation["Tmedia_aire"].apply(
    lambda value: calculate_tt_scalar(
        value,
        float(t_base),
        float(t_optimum),
        float(t_critical),
    )
)

first_peak_date = (
    pd.Timestamp(simulation.loc[first_peak_index, "Fecha"])
    if first_peak_index is not None
    else None
)

control_date = None
limit_date = None
thermal_time_today = 0.0
if first_peak_date is not None:
    from_peak = simulation[
        simulation["Fecha"] >= first_peak_date
    ].copy()
    from_peak["DGA_cum"] = from_peak["DG"].cumsum()

    control_candidates = from_peak[
        from_peak["DGA_cum"] >= float(tt_control)
    ]
    if not control_candidates.empty:
        control_date = pd.Timestamp(
            control_candidates.iloc[0]["Fecha"]
        )

    limit_candidates = from_peak[
        from_peak["DGA_cum"] >= float(tt_limit)
    ]
    if not limit_candidates.empty:
        limit_date = pd.Timestamp(
            limit_candidates.iloc[0]["Fecha"]
        )

    today = pd.Timestamp.now().normalize()
    effective_today = min(
        max(today, simulation["Fecha"].min()),
        simulation["Fecha"].max(),
    )
    thermal_time_today = float(
        simulation.loc[
            (simulation["Fecha"] >= first_peak_date)
            & (simulation["Fecha"] <= effective_today),
            "DG",
        ].sum()
    )

metric_columns = st.columns(5)
metric_columns[0].metric(
    "Primer pico",
    first_peak_date.strftime("%d/%m/%Y")
    if first_peak_date is not None
    else "No habilitado",
)
metric_columns[1].metric("Cobertura", f"{coverage_percent} %")
metric_columns[2].metric("Wmax", f"{w_max_value:.3f} mm")
metric_columns[3].metric("Ke", f"{ke_value:.3f}")
metric_columns[4].metric(
    "TT acumulado",
    f"{thermal_time_today:.1f} °Cd",
)

if first_peak_date is not None:
    st.success(
        f"Inicio habilitado el {first_peak_date.strftime('%d/%m/%Y')} "
        f"por un valor diario de EMERREL > {UMBRAL_PRIMER_PICO:.2f}."
    )
else:
    st.warning(
        "Ningún día supera el umbral del primer pico después de "
        "aplicar los filtros ecofisiológicos."
    )

tabs = st.tabs(
    [
        "📊 Emergencia",
        "💧 Agua y microclima",
        "🌡️ Logística térmica",
        "🧪 Validación",
        "📥 Exportar",
    ]
)

with tabs[0]:
    emergence_figure = go.Figure()
    emergence_figure.add_trace(
        go.Scatter(
            x=simulation["Fecha"],
            y=simulation["EMERREL"],
            mode="lines",
            name="EMERREL diaria",
        )
    )
    emergence_figure.add_trace(
        go.Scatter(
            x=simulation["Fecha"],
            y=simulation["EMERAC_NORMALIZADA"],
            mode="lines",
            name="Emergencia acumulada",
            line=dict(dash="dash"),
        )
    )

    if first_peak_date is not None:
        emergence_figure.add_vline(
            x=first_peak_date,
            line_dash="dot",
            annotation_text="Primer pico habilitado",
            annotation_position="top left",
        )

    if field_raw is not None:
        try:
            field, field_date_column, field_value_column = (
                canonicalize_field(field_raw)
            )
            emergence_figure.add_trace(
                go.Scatter(
                    x=field[field_date_column],
                    y=field["Campo_Normalizado"],
                    mode="markers",
                    name="Campo normalizado",
                    marker=dict(size=9),
                )
            )
        except Exception as exc:
            st.warning(f"No se pudo incorporar el campo al gráfico: {exc}")

    emergence_figure.update_layout(
        title="Emergencia simulada",
        xaxis_title="Fecha",
        yaxis_title="Emergencia relativa",
        yaxis=dict(range=[0.0, 1.05]),
        hovermode="x unified",
        height=520,
    )
    st.plotly_chart(emergence_figure, width="stretch")

    risk_figure = go.Figure(
        data=go.Heatmap(
            z=[simulation["EMERREL"].to_numpy()],
            x=simulation["Fecha"],
            y=["Riesgo"],
            zmin=0.0,
            zmax=1.0,
            colorscale=[
                [0.00, "green"],
                [0.10, "green"],
                [0.33, "yellow"],
                [0.66, "orange"],
                [1.00, "red"],
            ],
            showscale=True,
        )
    )
    risk_figure.update_layout(
        title="Mapa diario de riesgo",
        height=180,
        margin=dict(t=45, b=20, l=20, r=20),
    )
    st.plotly_chart(risk_figure, width="stretch")

with tabs[1]:
    water_figure = go.Figure()
    water_figure.add_trace(
        go.Bar(
            x=simulation["Fecha"],
            y=simulation["Prec"],
            name="Precipitación",
            yaxis="y2",
            opacity=0.45,
        )
    )
    water_figure.add_trace(
        go.Scatter(
            x=simulation["Fecha"],
            y=simulation["W_superficial"],
            mode="lines",
            name="Agua superficial",
        )
    )
    water_figure.add_trace(
        go.Scatter(
            x=simulation["Fecha"],
            y=simulation["Hydric_Factor"],
            mode="lines",
            name="Factor hídrico",
        )
    )
    water_figure.update_layout(
        title="Balance hídrico superficial",
        xaxis_title="Fecha",
        yaxis=dict(title="Agua / factor hídrico"),
        yaxis2=dict(
            title="Precipitación (mm)",
            overlaying="y",
            side="right",
        ),
        hovermode="x unified",
        height=500,
    )
    st.plotly_chart(water_figure, width="stretch")

    thermal_figure = go.Figure()
    thermal_figure.add_trace(
        go.Scatter(
            x=simulation["Fecha"],
            y=simulation["Tmedia_aire"],
            mode="lines",
            name="Tmedia aire",
        )
    )
    thermal_figure.add_trace(
        go.Scatter(
            x=simulation["Fecha"],
            y=simulation["Tmedia_5d"],
            mode="lines",
            name="Tmedia móvil 5 días",
        )
    )
    thermal_figure.add_hline(
        y=float(thermoinhibition_threshold),
        line_dash="dash",
        annotation_text="Umbral termoinhibición",
    )
    thermal_figure.update_layout(
        title="Escudo termofisiológico",
        xaxis_title="Fecha",
        yaxis_title="Temperatura (°C)",
        hovermode="x unified",
        height=430,
    )
    st.plotly_chart(thermal_figure, width="stretch")

with tabs[2]:
    logistics_columns = st.columns(3)
    logistics_columns[0].metric(
        "Inicio",
        first_peak_date.strftime("%d/%m/%Y")
        if first_peak_date is not None
        else "N/D",
    )
    logistics_columns[1].metric(
        "Fecha objetivo de control",
        control_date.strftime("%d/%m/%Y")
        if control_date is not None
        else "N/D",
    )
    logistics_columns[2].metric(
        "Límite de ventana",
        limit_date.strftime("%d/%m/%Y")
        if limit_date is not None
        else "N/D",
    )

    if first_peak_date is not None:
        thermal_curve = simulation[
            simulation["Fecha"] >= first_peak_date
        ].copy()
        thermal_curve["DGA_cum"] = thermal_curve["DG"].cumsum()

        thermal_time_figure = go.Figure()
        thermal_time_figure.add_trace(
            go.Scatter(
                x=thermal_curve["Fecha"],
                y=thermal_curve["DGA_cum"],
                mode="lines",
                name="Tiempo térmico acumulado",
            )
        )
        thermal_time_figure.add_hline(
            y=float(tt_control),
            line_dash="dash",
            annotation_text="Control",
        )
        thermal_time_figure.add_hline(
            y=float(tt_limit),
            line_dash="dot",
            annotation_text="Límite",
        )
        thermal_time_figure.update_layout(
            title="Reloj logístico desde el primer pico",
            xaxis_title="Fecha",
            yaxis_title="Tiempo térmico (°Cd)",
            height=470,
        )
        st.plotly_chart(thermal_time_figure, width="stretch")

with tabs[3]:
    if field_raw is None:
        st.info(
            "Cargue observaciones de campo para calcular las métricas."
        )
    else:
        try:
            field, field_date_column, field_value_column = (
                canonicalize_field(field_raw)
            )
            synchronized = synchronize_real_intervals(
                simulation,
                field,
                field_date_column,
                field_value_column,
            )
            metrics = validation_metrics(synchronized)

            validation_columns = st.columns(6)
            validation_columns[0].metric(
                "KGE",
                f"{metrics['KGE_Flujos']:.3f}",
            )
            validation_columns[1].metric(
                "NSE",
                f"{metrics['NSE_Flujos']:.3f}",
            )
            validation_columns[2].metric(
                "CCC",
                f"{metrics['CCC_Acumulado']:.3f}",
            )
            validation_columns[3].metric(
                "RMSE",
                f"{metrics['RMSE_Acumulado']:.3f}",
            )
            validation_columns[4].metric(
                "F1",
                f"{metrics['F1_Score_Coincidencia']:.3f}",
            )
            validation_columns[5].metric(
                "Exactitud",
                f"{metrics['Exactitud_Global']:.3f}",
            )

            comparison_figure = go.Figure()
            comparison_figure.add_trace(
                go.Scatter(
                    x=synchronized["Fecha"],
                    y=synchronized["Campo_Relativo"],
                    mode="markers+lines",
                    name="Campo por intervalo",
                )
            )
            comparison_figure.add_trace(
                go.Scatter(
                    x=synchronized["Fecha"],
                    y=synchronized["Sim_Relativo"],
                    mode="markers+lines",
                    name="Simulado por intervalo",
                )
            )
            comparison_figure.update_layout(
                title="Validación Event-to-Event",
                xaxis_title="Fecha de muestreo",
                yaxis_title="Flujo relativo",
                hovermode="x unified",
                height=470,
            )
            st.plotly_chart(comparison_figure, width="stretch")
            st.dataframe(
                synchronized,
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            st.exception(exc)

with tabs[4]:
    export_columns = [
        "Fecha",
        "Julian_days",
        "TMAX",
        "TMIN",
        "Prec",
        "Tmedia_aire",
        "Tmedia_5d",
        "ET0",
        "Cobertura_Rastrojo",
        "Ke_Suelo",
        "Modulador_Termico_Diagnostico",
        "W_superficial",
        "Humedad_Relativa",
        "Hydric_Factor",
        "Termoinhibida",
        "Choque_Hidrico",
        "Lluvia_Recarga",
        "EMERREL_RAW_ANN",
        "EMERREL_ANTES_FILTRO_PRIMER_PICO",
        "EMERREL",
        "EMERAC_NORMALIZADA",
        "DG",
    ]
    export_data = simulation[
        [column for column in export_columns if column in simulation.columns]
    ].copy()

    csv_data = export_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Descargar simulación CSV",
        data=csv_data,
        file_name="PREDWEEM_Lartigau_simulacion.csv",
        mime="text/csv",
        width="stretch",
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        export_data.to_excel(
            writer,
            sheet_name="Simulacion",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "Version": APP_VERSION,
                    "Cobertura_pct": coverage_percent,
                    "Ke": ke_value,
                    "Modulador_termico_diagnostico": thermal_modulator,
                    "Wmax_mm": float(w_max_value),
                    "Latencia_JD": LATENCIA_JD,
                    "Ventana_termica_dias": VENTANA_TERMICA_DIAS,
                    "Umbral_termoinhibicion_C": float(
                        thermoinhibition_threshold
                    ),
                    "Choque_hidrico_mm_3d": float(
                        hydric_shock_threshold
                    ),
                    "Umbral_primer_pico": UMBRAL_PRIMER_PICO,
                    "Persistencia_dias": (
                        PERSISTENCIA_PRIMER_PICO_DIAS
                    ),
                    "Lag_dias": 0,
                    "Fecha_primer_pico": (
                        first_peak_date
                        if first_peak_date is not None
                        else pd.NaT
                    ),
                }
            ]
        ).to_excel(
            writer,
            sheet_name="Parametros",
            index=False,
        )

    st.download_button(
        "📊 Descargar simulación Excel",
        data=excel_buffer.getvalue(),
        file_name="PREDWEEM_Lartigau_simulacion.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
    )

st.caption(
    "PREDWEEM by GUILLERMO R. CHANTRE · "
    "Modelo híbrido ANN + ecofisiología + validación de campo."
)
