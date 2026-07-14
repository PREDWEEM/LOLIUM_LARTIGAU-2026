# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 PREDWEEM INTEGRAL vK4.9.15 ADAPTADA — LOLIUM LARTIGAU 2026
# PREDWEEM by GUILLERMO R. CHANTRE
#
# ADAPTACIÓN:
# - conserva métricas, Event-to-Event, DTW, logística térmica,
#   matriz de confusión, calibrador 2D y reporte Excel de vK4.9.15;
# - la ANN recibe JD, TMAX, TMIN y precipitación meteorológicas;
# - la cobertura modifica Ke y el balance hídrico, no la ANN;
# - el modulador térmico se mantiene únicamente como diagnóstico;
# - el inicio se habilita con 1 día de EMERREL > 0.70;
# - latencia fija JD 45 y termoinhibición 5 días >= 24 °C;
# - sin fecha objetivo y sin lag temporal.
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
APP_VERSION = "vK4.9.15 Adaptada"
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
# 7. MÉTRICAS Y FUNCIONALIDADES INTEGRALES vK4.9.15
# ---------------------------------------------------------------
def dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray) -> float:
    """Distancia Dynamic Time Warping clásica."""
    a = np.asarray(sequence_a, dtype=float)
    b = np.asarray(sequence_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("inf")

    matrix = np.full((len(a) + 1, len(b) + 1), np.inf)
    matrix[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = abs(a[i - 1] - b[j - 1])
            matrix[i, j] = cost + min(
                matrix[i - 1, j],
                matrix[i, j - 1],
                matrix[i - 1, j - 1],
            )
    return float(matrix[-1, -1])


def calculate_operational_metrics(
    simulation: pd.DataFrame,
    field: pd.DataFrame | None,
    field_date_column: str | None,
    field_value_column: str | None,
    first_peak_date: pd.Timestamp | None,
    control_date: pd.Timestamp | None,
    alert_threshold: float,
) -> dict[str, Any]:
    """Métricas operativas incluidas en vK4.9.15."""
    result: dict[str, Any] = {
        "Fecha_Primer_Flujo_Observado": pd.NaT,
        "Desfase_Primer_Flujo_Dias": None,
        "T50_Observado": pd.NaT,
        "T50_Simulado": pd.NaT,
        "Desfase_T50_Dias": None,
        "PEC_Porcentaje": None,
        "Lag_Control_vs_Pico_Campo_Dias": None,
        "Lead_Time_Dias": None,
        "Fecha_Primera_Alerta": pd.NaT,
    }
    if (
        field is None
        or field.empty
        or field_date_column is None
        or field_value_column is None
    ):
        return result

    positive = field[field[field_value_column] > 0]
    if not positive.empty:
        first_observed = pd.Timestamp(positive.iloc[0][field_date_column])
        result["Fecha_Primer_Flujo_Observado"] = first_observed
        if first_peak_date is not None:
            result["Desfase_Primer_Flujo_Dias"] = int(
                (first_peak_date - first_observed).days
            )

    total_field = float(field[field_value_column].sum())
    if total_field > 0:
        field_cumulative = field[field_value_column].cumsum() / total_field
        t50_candidates = field.loc[field_cumulative >= 0.5, field_date_column]
        if not t50_candidates.empty:
            result["T50_Observado"] = pd.Timestamp(t50_candidates.iloc[0])

        last_field_date = pd.Timestamp(field[field_date_column].max())
        simulation_cut = simulation[
            simulation["Fecha"] <= last_field_date
        ].copy()
        total_simulation = float(simulation_cut["EMERREL"].sum())
        if total_simulation > 0:
            sim_cumulative = (
                simulation_cut["EMERREL"].cumsum() / total_simulation
            )
            sim_t50_candidates = simulation_cut.loc[
                sim_cumulative >= 0.5,
                "Fecha",
            ]
            if not sim_t50_candidates.empty:
                result["T50_Simulado"] = pd.Timestamp(
                    sim_t50_candidates.iloc[0]
                )

        if (
            pd.notna(result["T50_Observado"])
            and pd.notna(result["T50_Simulado"])
        ):
            result["Desfase_T50_Dias"] = int(
                (
                    result["T50_Simulado"]
                    - result["T50_Observado"]
                ).days
            )

    alert_candidates = simulation[
        simulation["EMERREL"] >= float(alert_threshold)
    ]
    if not alert_candidates.empty:
        result["Fecha_Primera_Alerta"] = pd.Timestamp(
            alert_candidates.iloc[0]["Fecha"]
        )

    if control_date is not None and total_field > 0:
        result["PEC_Porcentaje"] = float(
            100.0
            * field.loc[
                field[field_date_column] <= control_date,
                field_value_column,
            ].sum()
            / total_field
        )
        observed_peak_date = pd.Timestamp(
            field.loc[field[field_value_column].idxmax(), field_date_column]
        )
        result["Lag_Control_vs_Pico_Campo_Dias"] = int(
            (control_date - observed_peak_date).days
        )
        if pd.notna(result["Fecha_Primera_Alerta"]):
            result["Lead_Time_Dias"] = int(
                (
                    control_date
                    - result["Fecha_Primera_Alerta"]
                ).days
            )
    return result


def classify_dtw_pattern(
    simulation: pd.DataFrame,
    cluster_model: Any,
    cutoff_date: str = "2026-05-01",
) -> dict[str, Any] | None:
    """Clasifica la dinámica temprana contra los medoides históricos."""
    if not isinstance(cluster_model, dict):
        return None
    if "JD_common" not in cluster_model or "curves_interp" not in cluster_model:
        return None

    observed = simulation[
        simulation["Fecha"] < pd.Timestamp(cutoff_date)
    ].copy()
    if observed.empty or float(observed["EMERREL"].sum()) <= 0:
        return None

    jd_common = np.asarray(cluster_model["JD_common"], dtype=float)
    curves = np.asarray(cluster_model["curves_interp"], dtype=float)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)

    medoid_indices = cluster_model.get("medoids_k3")
    if medoid_indices is not None:
        medoid_indices = np.asarray(medoid_indices, dtype=int)
        valid = medoid_indices[
            (medoid_indices >= 0) & (medoid_indices < len(curves))
        ]
        prototypes = curves[valid] if len(valid) >= 3 else curves[:3]
    else:
        prototypes = curves[:3]

    if len(prototypes) < 3:
        return None

    cutoff_jd = float(observed["Julian_days"].max())
    mask = jd_common <= cutoff_jd
    jd_grid = jd_common[mask]
    if jd_grid.size < 3:
        return None

    observed_max = float(observed["EMERREL"].max())
    if observed_max <= 0:
        return None

    observed_normalized = np.interp(
        jd_grid,
        observed["Julian_days"].to_numpy(float),
        observed["EMERREL"].to_numpy(float) / observed_max,
    )

    distances: list[float] = []
    normalized_prototypes: list[np.ndarray] = []
    for prototype in prototypes:
        prototype_cut = np.asarray(prototype, dtype=float)[mask]
        maximum = float(np.max(prototype_cut)) if prototype_cut.size else 0.0
        normalized = (
            prototype_cut / maximum if maximum > 0 else prototype_cut
        )
        normalized_prototypes.append(normalized)
        distances.append(dtw_distance(observed_normalized, normalized))

    prediction = int(np.argmin(distances))
    names = {
        0: "🌾 Intermedio / Bimodal",
        1: "🌱 Temprano / Compacto",
        2: "🍂 Tardío / Extendido",
    }
    colors = {0: "#0284c7", 1: "#16a34a", 2: "#ea580c"}
    return {
        "prediction": prediction,
        "name": names.get(prediction, "Patrón desconocido"),
        "color": colors.get(prediction, "#475569"),
        "distance": float(distances[prediction]),
        "distances": distances,
        "jd_grid": jd_grid,
        "observed_normalized": observed_normalized,
        "prototype": normalized_prototypes[prediction],
    }


def optimize_hydric_2d(
    weather_raw: pd.DataFrame,
    field_raw: pd.DataFrame,
    ann_model: PracticalANNModel,
    coverage_values: np.ndarray,
    wmax_values: np.ndarray,
    thermoinhibition_threshold: float,
    hydric_shock_threshold: float,
) -> pd.DataFrame:
    """
    Calibrador biofísico 2D adaptado.

    Conserva fijos los parámetros temporales y explora únicamente
    cobertura (que determina Ke) y Wmax.
    """
    field, date_column, value_column = canonicalize_field(field_raw)
    rows: list[dict[str, Any]] = []

    for coverage in coverage_values:
        ke_value, thermal_modulator = surface_parameters(float(coverage))
        for wmax in wmax_values:
            simulated, peak_index = simulate_emergence(
                weather_raw,
                ann_model,
                coverage_percent=int(coverage),
                w_max=float(wmax),
                thermoinhibition_threshold=float(
                    thermoinhibition_threshold
                ),
                hydric_shock_threshold=float(hydric_shock_threshold),
            )
            synchronized = synchronize_real_intervals(
                simulated,
                field,
                date_column,
                value_column,
            )
            metrics = validation_metrics(synchronized)
            first_date = (
                pd.Timestamp(simulated.loc[peak_index, "Fecha"])
                if peak_index is not None
                else pd.NaT
            )
            rows.append(
                {
                    "Cobertura_pct": int(coverage),
                    "Ke_Suelo": ke_value,
                    "Mod_Termico_Diagnostico": thermal_modulator,
                    "W_Max_mm": float(wmax),
                    "Fecha_Primer_Pico": first_date,
                    "Pearson_Flujos": metrics["Pearson_Flujos"],
                    "NSE_Flujos": metrics["NSE_Flujos"],
                    "KGE_Flujos": metrics["KGE_Flujos"],
                    "CCC_Acumulado": metrics["CCC_Acumulado"],
                    "R2_Acumulado": metrics["R2_Acumulado"],
                    "RMSE_Acumulado": metrics["RMSE_Acumulado"],
                    "F1_Score": metrics["F1_Score_Coincidencia"],
                    "Exactitud": metrics["Exactitud_Global"],
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        by=["F1_Score", "KGE_Flujos", "NSE_Flujos"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def add_interval_shading(
    figure: go.Figure,
    field: pd.DataFrame | None,
    date_column: str | None,
) -> None:
    """Añade bandas alternas suaves para los intervalos reales."""
    if field is None or field.empty or date_column is None:
        return

    dates = (
        pd.to_datetime(field[date_column], errors="coerce")
        .dropna()
        .sort_values()
        .tolist()
    )
    for index in range(1, len(dates), 2):
        figure.add_vrect(
            x0=dates[index - 1],
            x1=dates[index],
            fillcolor="rgba(148, 163, 184, 0.065)",
            layer="below",
            line_width=0,
        )


# ---------------------------------------------------------------
# 8. INTERFAZ INTEGRAL
# ---------------------------------------------------------------
ann_model, cluster_model = load_models()

st.title(
    "🌾 PREDWEEM LOLIUM — LARTIGAU "
    "(BA, lat=-38.6166; lon=-61.7000)"
)
st.caption(
    "vK4.9.15 Adaptada · VISUAL V3 · "
    "ANN desacoplada de la cobertura · sin fecha objetivo y sin lag"
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

st.sidebar.markdown("## ⚙️ Fisiología y logística")
alert_threshold = st.sidebar.slider(
    "Umbral de alerta temprana",
    min_value=0.001,
    max_value=0.80,
    value=0.005,
    step=0.001,
)
residual_days = st.sidebar.number_input(
    "Residualidad del herbicida (días)",
    min_value=0,
    max_value=60,
    value=0,
    step=1,
)

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
simulation["EMERREL_LOG"] = np.log10(simulation["EMERREL"] + 0.01)
alert_threshold_log = np.log10(float(alert_threshold) + 0.01)

first_peak_date = (
    pd.Timestamp(simulation.loc[first_peak_index, "Fecha"])
    if first_peak_index is not None
    else None
)

control_date = None
limit_date = None
thermal_time_today = 0.0
thermal_time_7days = 0.0
thermal_curve = pd.DataFrame()

if first_peak_date is not None:
    thermal_curve = simulation[
        simulation["Fecha"] >= first_peak_date
    ].copy()
    thermal_curve["DGA_cum"] = thermal_curve["DG"].cumsum()

    control_candidates = thermal_curve[
        thermal_curve["DGA_cum"] >= float(tt_control)
    ]
    if not control_candidates.empty:
        control_date = pd.Timestamp(
            control_candidates.iloc[0]["Fecha"]
        )

    limit_candidates = thermal_curve[
        thermal_curve["DGA_cum"] >= float(tt_limit)
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
    today_candidates = simulation.index[
        simulation["Fecha"] <= effective_today
    ].tolist()
    today_index = today_candidates[-1] if today_candidates else 0
    forecast_index = min(today_index + 7, len(simulation) - 1)

    thermal_time_today = float(
        simulation.loc[
            (simulation["Fecha"] >= first_peak_date)
            & (simulation.index <= today_index),
            "DG",
        ].sum()
    )
    thermal_time_7days = float(
        simulation.loc[
            (simulation["Fecha"] >= first_peak_date)
            & (simulation.index <= forecast_index),
            "DG",
        ].sum()
    )

field = None
field_date_column = None
field_value_column = None
synchronized = pd.DataFrame()
metrics = validation_metrics(synchronized)
field_error = None

if field_raw is not None:
    try:
        field, field_date_column, field_value_column = canonicalize_field(
            field_raw
        )
        field["Campo_Normalizado_LOG"] = np.log10(
            field["Campo_Normalizado"] + 0.01
        )
        synchronized = synchronize_real_intervals(
            simulation,
            field,
            field_date_column,
            field_value_column,
        )
        metrics = validation_metrics(synchronized)
    except Exception as exc:
        field_error = str(exc)

operational = calculate_operational_metrics(
    simulation,
    field,
    field_date_column,
    field_value_column,
    first_peak_date,
    control_date,
    float(alert_threshold),
)

top_metrics = st.columns(5)
top_metrics[0].metric(
    "Primer pico",
    first_peak_date.strftime("%d/%m/%Y")
    if first_peak_date is not None
    else "No habilitado",
)
top_metrics[1].metric("Cobertura", f"{coverage_percent} %")
top_metrics[2].metric("Wmax", f"{w_max_value:.3f} mm")
top_metrics[3].metric("Ke", f"{ke_value:.3f}")
top_metrics[4].metric(
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
        "📊 Monitor de decisión",
        "💧 Agua y suelo",
        "📈 Análisis estratégico",
        "🧪 Bio-calibración",
        "📥 Exportar",
    ]
)

with tabs[0]:
    if field_error:
        st.warning(f"No se pudo procesar el archivo de campo: {field_error}")

    if field is not None and not synchronized.empty:
        st.markdown("#### Fidelidad científica — Event-to-Event")
        validation_row_1 = st.columns(6)
        validation_row_1[0].metric(
            "Pearson flujos",
            f"{metrics['Pearson_Flujos']:.3f}",
        )
        validation_row_1[1].metric(
            "NSE flujos",
            f"{metrics['NSE_Flujos']:.3f}",
        )
        validation_row_1[2].metric(
            "KGE flujos",
            f"{metrics['KGE_Flujos']:.3f}",
        )
        validation_row_1[3].metric(
            "RMSE acumulado",
            f"{metrics['RMSE_Acumulado']:.3f}",
        )
        validation_row_1[4].metric(
            "CCC acumulado",
            f"{metrics['CCC_Acumulado']:.3f}",
        )
        validation_row_1[5].metric(
            "R² acumulado",
            f"{metrics['R2_Acumulado']:.3f}",
        )

        validation_row_2 = st.columns(5)
        validation_row_2[0].metric(
            "F1 coincidencia",
            f"{metrics['F1_Score_Coincidencia']:.3f}",
        )
        validation_row_2[1].metric(
            "Exactitud",
            f"{metrics['Exactitud_Global']:.3f}",
        )
        validation_row_2[2].metric(
            "Desfase inicio",
            (
                f"{operational['Desfase_Primer_Flujo_Dias']:+d} días"
                if operational["Desfase_Primer_Flujo_Dias"] is not None
                else "N/D"
            ),
        )
        validation_row_2[3].metric(
            "Desfase T50",
            (
                f"{operational['Desfase_T50_Dias']:+d} días"
                if operational["Desfase_T50_Dias"] is not None
                else "N/D"
            ),
        )
        validation_row_2[4].metric(
            "Lead time",
            (
                f"{operational['Lead_Time_Dias']} días"
                if operational["Lead_Time_Dias"] is not None
                else "N/D"
            ),
        )

        confusion = pd.DataFrame(
            [
                {
                    "Campo": "Con flujo",
                    "Modelo con flujo": metrics["Hits"],
                    "Modelo sin flujo": metrics["Misses"],
                },
                {
                    "Campo": "Sin flujo",
                    "Modelo con flujo": metrics["Falsos_Positivos"],
                    "Modelo sin flujo": metrics["Correctos_Negativos"],
                },
            ]
        )
        st.markdown("#### Matriz de confusión por intervalos")
        st.dataframe(confusion, width="stretch", hide_index=True)

        if control_date is not None:
            logistics_metrics = st.columns(3)
            logistics_metrics[0].metric(
                "Control efectivo (PEC)",
                (
                    f"{operational['PEC_Porcentaje']:.1f}%"
                    if operational["PEC_Porcentaje"] is not None
                    else "N/D"
                ),
            )
            logistics_metrics[1].metric(
                "Lag control vs pico campo",
                (
                    f"{operational['Lag_Control_vs_Pico_Campo_Dias']} días"
                    if operational[
                        "Lag_Control_vs_Pico_Campo_Dias"
                    ] is not None
                    else "N/D"
                ),
            )
            logistics_metrics[2].metric(
                "Lead time de alerta",
                (
                    f"{operational['Lead_Time_Dias']} días"
                    if operational["Lead_Time_Dias"] is not None
                    else "N/D"
                ),
            )

    main_column, gauge_column = st.columns([3.4, 1])

    with main_column:
        emergence_log_figure = go.Figure()

        # Intervalos reales de monitoreo: bandas tenues.
        add_interval_shading(
            emergence_log_figure,
            field,
            field_date_column,
        )

        # Ventana de aplicación eficiente.
        if control_date is not None and limit_date is not None:
            emergence_log_figure.add_vrect(
                x0=control_date,
                x1=limit_date,
                fillcolor="rgba(245, 158, 11, 0.085)",
                layer="below",
                line_width=0,
            )

        # Residualidad, cuando corresponde.
        if control_date is not None and residual_days > 0:
            emergence_log_figure.add_vrect(
                x0=control_date,
                x1=control_date + pd.Timedelta(days=int(residual_days)),
                fillcolor="rgba(37, 99, 235, 0.055)",
                layer="below",
                line_width=0,
            )

        # Curva simulada sin relleno, para evitar saturación visual.
        emergence_log_figure.add_trace(
            go.Scatter(
                x=simulation["Fecha"],
                y=simulation["EMERREL_LOG"],
                mode="lines",
                name="Tasa diaria simulada (log)",
                line=dict(
                    color="#075FCF",
                    width=2.4,
                ),
                hovertemplate=(
                    "<b>%{x|%d-%m-%Y}</b><br>"
                    "Simulado: %{y:.3f}<extra></extra>"
                ),
            )
        )

        # Datos de campo.
        if field is not None:
            emergence_log_figure.add_trace(
                go.Scatter(
                    x=field[field_date_column],
                    y=field["Campo_Normalizado_LOG"],
                    mode="markers+lines",
                    name="Campo normalizado (log)",
                    marker=dict(
                        size=9,
                        symbol="circle",
                        color="#60A5FA",
                        line=dict(
                            color="#FFFFFF",
                            width=1.4,
                        ),
                    ),
                    line=dict(
                        color="#60A5FA",
                        width=2.2,
                        dash="dash",
                    ),
                    hovertemplate=(
                        "<b>%{x|%d-%m-%Y}</b><br>"
                        "Campo: %{y:.3f}<extra></extra>"
                    ),
                )
            )

        # Umbral de alerta.
        emergence_log_figure.add_hline(
            y=alert_threshold_log,
            line_color="#111827",
            line_width=1.7,
            line_dash="dash",
        )
        emergence_log_figure.add_annotation(
            x=0.995,
            xref="paper",
            y=alert_threshold_log,
            yref="y",
            text=f"Alerta ({alert_threshold:.3f})",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            yshift=7,
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="rgba(148,163,184,0.50)",
            borderwidth=1,
            borderpad=4,
            font=dict(
                size=12,
                color="#374151",
            ),
        )

        # Inicio del recuento térmico.
        if first_peak_date is not None:
            emergence_log_figure.add_vline(
                x=first_peak_date,
                line_color="#111827",
                line_width=1.5,
                line_dash="dot",
            )
            emergence_log_figure.add_annotation(
                x=first_peak_date,
                xref="x",
                y=1.035,
                yref="paper",
                text="Inicio térmico",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.96)",
                bordercolor="rgba(148,163,184,0.45)",
                borderwidth=1,
                borderpad=4,
                font=dict(size=12, color="#111827"),
            )

        # Fecha de control.
        if control_date is not None:
            emergence_log_figure.add_vline(
                x=control_date,
                line_color="#111827",
                line_width=1.5,
                line_dash="dot",
            )
            emergence_log_figure.add_annotation(
                x=control_date,
                xref="x",
                y=1.105,
                yref="paper",
                text=f"Control ({tt_control} °Cd)",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.96)",
                bordercolor="rgba(148,163,184,0.45)",
                borderwidth=1,
                borderpad=4,
                font=dict(size=12, color="#111827"),
            )

        # Etiqueta de la ventana agronómica.
        if control_date is not None and limit_date is not None:
            emergence_log_figure.add_annotation(
                x=control_date + (limit_date - control_date) / 2,
                xref="x",
                y=0.975,
                yref="paper",
                text="Ventana eficiente",
                showarrow=False,
                xanchor="center",
                yanchor="top",
                bgcolor="rgba(255,251,235,0.90)",
                borderpad=3,
                font=dict(
                    size=11,
                    color="#92400E",
                ),
            )

        # Marcas mensuales en español.
        first_month = pd.Timestamp(
            simulation["Fecha"].min()
        ).to_period("M").to_timestamp()
        last_month = pd.Timestamp(
            simulation["Fecha"].max()
        ).to_period("M").to_timestamp()
        month_ticks = pd.date_range(
            first_month,
            last_month,
            freq="MS",
        )
        month_names = {
            1: "Ene",
            2: "Feb",
            3: "Mar",
            4: "Abr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Ago",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dic",
        }
        month_labels = [
            f"{month_names[date.month]} {date.year}"
            for date in month_ticks
        ]

        emergence_log_figure.update_layout(
            title=dict(
                text=(
                    "Dinámica fisiológica de emergencia "
                    "(intervalos reales de monitoreo)"
                ),
                x=0.0,
                xanchor="left",
                font=dict(
                    size=21,
                    color="#111827",
                    family="Arial, sans-serif",
                ),
            ),
            xaxis=dict(
                title=dict(
                    text="Fecha",
                    font=dict(
                        size=15,
                        color="#4B5563",
                    ),
                    standoff=14,
                ),
                tickmode="array",
                tickvals=month_ticks,
                ticktext=month_labels,
                tickfont=dict(
                    size=12,
                    color="#4B5563",
                ),
                showgrid=False,
                showline=True,
                linecolor="#6B7280",
                linewidth=1,
                ticks="outside",
                ticklen=6,
                zeroline=False,
                automargin=True,
            ),
            yaxis=dict(
                title=dict(
                    text="Log10(EMERREL + 0,01)",
                    font=dict(
                        size=15,
                        color="#4B5563",
                    ),
                    standoff=12,
                ),
                range=[-2.18, 0.12],
                tickmode="array",
                tickvals=[-2.0, -1.5, -1.0, -0.5, 0.0],
                tickfont=dict(
                    size=12,
                    color="#4B5563",
                ),
                showgrid=True,
                gridcolor="rgba(148, 163, 184, 0.28)",
                griddash="dash",
                gridwidth=1,
                showline=True,
                linecolor="#6B7280",
                linewidth=1,
                zeroline=False,
                automargin=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.17,
                xanchor="right",
                x=1.0,
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(148,163,184,0.35)",
                borderwidth=1,
                font=dict(
                    size=12,
                    color="#374151",
                ),
            ),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#CBD5E1",
                font=dict(
                    size=12,
                    color="#111827",
                ),
            ),
            height=630,
            margin=dict(
                l=82,
                r=28,
                t=132,
                b=78,
            ),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(
                family="Arial, sans-serif",
                color="#374151",
            ),
            dragmode="zoom",
        )

        emergence_log_figure.update_xaxes(
            rangeslider_visible=False,
            fixedrange=False,
        )
        emergence_log_figure.update_yaxes(fixedrange=False)

        st.plotly_chart(
            emergence_log_figure,
            width="stretch",
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "modeBarButtonsToRemove": [
                    "lasso2d",
                    "select2d",
                ],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "PREDWEEM_dinamica_emergencia",
                    "height": 1000,
                    "width": 2000,
                    "scale": 2,
                },
            },
        )

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

    with gauge_column:
        maximum_axis = max(float(tt_limit) * 1.2, 1.0)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=thermal_time_today,
                title={"text": "TT POST-EMERGENCIA (°Cd)"},
                gauge={
                    "axis": {"range": [0, maximum_axis]},
                    "steps": [
                        {
                            "range": [0, float(tt_control)],
                            "color": "#4ade80",
                        },
                        {
                            "range": [
                                float(tt_control),
                                float(tt_limit),
                            ],
                            "color": "#facc15",
                        },
                        {
                            "range": [
                                float(tt_limit),
                                maximum_axis,
                            ],
                            "color": "#f87171",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "#2563eb", "width": 6},
                        "thickness": 0.8,
                        "value": thermal_time_7days,
                    },
                },
            )
        )
        gauge.add_annotation(
            x=0.5,
            y=-0.10,
            text=(
                f"Pronóstico +7 días: "
                f"<b>{thermal_time_7days:.1f} °Cd</b>"
            ),
            showarrow=False,
        )
        gauge.update_layout(
            height=380,
            margin=dict(t=70, b=60, l=25, r=25),
        )
        st.plotly_chart(gauge, width="stretch")

    if field is not None and not synchronized.empty:
        st.markdown("#### Curvas acumuladas y rectas 1:1")
        cumulative_column, scatter_column = st.columns([2, 1])

        with cumulative_column:
            cumulative_figure = go.Figure()
            cumulative_figure.add_trace(
                go.Scatter(
                    x=synchronized["Fecha"],
                    y=synchronized["Campo_Acumulado"] * 100,
                    mode="markers+lines",
                    name="Campo acumulado",
                )
            )
            cumulative_figure.add_trace(
                go.Scatter(
                    x=synchronized["Fecha"],
                    y=synchronized["Sim_Acumulado"] * 100,
                    mode="lines",
                    name="PREDWEEM acumulado",
                    line=dict(dash="dash"),
                )
            )
            cumulative_figure.update_layout(
                title="Llenado cinético acumulado",
                xaxis_title="Fecha",
                yaxis_title="Emergencia acumulada (%)",
                hovermode="x unified",
                height=430,
            )
            st.plotly_chart(cumulative_figure, width="stretch")

        with scatter_column:
            flow_tab, accumulated_tab = st.tabs(
                ["1:1 Flujos", "1:1 Acumulado"]
            )
            with flow_tab:
                flow_figure = go.Figure()
                flow_figure.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(dash="dash"),
                        showlegend=False,
                    )
                )
                flow_figure.add_trace(
                    go.Scatter(
                        x=synchronized["Campo_Relativo"],
                        y=synchronized["Sim_Relativo"],
                        mode="markers",
                        text=synchronized["Fecha"].dt.strftime(
                            "%d-%m-%Y"
                        ),
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Obs: %{x:.3f}<br>Sim: %{y:.3f}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
                flow_figure.update_layout(
                    title="Flujos por evento",
                    xaxis_title="Observado",
                    yaxis_title="Simulado",
                    height=360,
                )
                st.plotly_chart(flow_figure, width="stretch")

            with accumulated_tab:
                accumulated_figure = go.Figure()
                accumulated_figure.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        line=dict(dash="dash"),
                        showlegend=False,
                    )
                )
                accumulated_figure.add_trace(
                    go.Scatter(
                        x=synchronized["Campo_Acumulado"],
                        y=synchronized["Sim_Acumulado"],
                        mode="markers",
                        text=synchronized["Fecha"].dt.strftime(
                            "%d-%m-%Y"
                        ),
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "Obs acum: %{x:.3f}<br>"
                            "Sim acum: %{y:.3f}"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
                accumulated_figure.update_layout(
                    title=(
                        f"Acumulado — R² "
                        f"{metrics['R2_Acumulado']:.3f}"
                    ),
                    xaxis_title="Observado acumulado",
                    yaxis_title="Simulado acumulado",
                    height=360,
                )
                st.plotly_chart(
                    accumulated_figure,
                    width="stretch",
                )

with tabs[1]:
    water_figure = go.Figure()
    water_figure.add_trace(
        go.Bar(
            x=simulation["Fecha"],
            y=simulation["Prec"],
            name="Precipitación",
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
    water_figure.add_hline(
        y=float(w_max_value),
        line_dash="dot",
        annotation_text=f"Wmax = {w_max_value:.2f} mm",
    )
    water_figure.update_layout(
        title="Precipitación y retención de agua superficial",
        xaxis_title="Fecha",
        yaxis_title="Milímetros",
        hovermode="x unified",
        height=480,
    )
    st.plotly_chart(water_figure, width="stretch")

    hydric_factor_figure = go.Figure()
    hydric_factor_figure.add_trace(
        go.Scatter(
            x=simulation["Fecha"],
            y=simulation["Hydric_Factor"],
            mode="lines",
            name="Factor hídrico",
        )
    )
    hydric_factor_figure.add_trace(
        go.Scatter(
            x=simulation["Fecha"],
            y=simulation["Humedad_Relativa"],
            mode="lines",
            name="Humedad relativa",
        )
    )
    hydric_factor_figure.update_layout(
        title="Disponibilidad hídrica",
        xaxis_title="Fecha",
        yaxis_title="Proporción",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(hydric_factor_figure, width="stretch")

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
    st.subheader("Clasificación temporal DTW")
    dtw_result = classify_dtw_pattern(simulation, cluster_model)
    if dtw_result is None:
        st.info(
            "No hay información suficiente o no se encontró "
            "`modelo_clusters_k3.pkl`."
        )
    else:
        pattern_column, score_column = st.columns([3, 1])
        with pattern_column:
            dtw_figure = go.Figure()
            dtw_figure.add_trace(
                go.Scatter(
                    x=dtw_result["jd_grid"],
                    y=dtw_result["prototype"],
                    mode="lines",
                    name="Patrón histórico",
                    line=dict(dash="dash"),
                )
            )
            dtw_figure.add_trace(
                go.Scatter(
                    x=dtw_result["jd_grid"],
                    y=dtw_result["observed_normalized"],
                    mode="lines",
                    name="Simulación 2026",
                )
            )
            dtw_figure.update_layout(
                title="Comparación con el patrón histórico",
                xaxis_title="Día juliano",
                yaxis_title="Emergencia normalizada",
                height=430,
            )
            st.plotly_chart(dtw_figure, width="stretch")
        with score_column:
            st.success(f"### {dtw_result['name']}")
            st.metric(
                "DTW Score",
                f"{dtw_result['distance']:.3f}",
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Patrón": [
                            "Intermedio/Bimodal",
                            "Temprano/Compacto",
                            "Tardío/Extendido",
                        ],
                        "Distancia DTW": dtw_result["distances"][:3],
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    if field is not None and not synchronized.empty:
        st.subheader("Tabla Event-to-Event")
        st.dataframe(
            synchronized,
            width="stretch",
            hide_index=True,
        )

with tabs[3]:
    st.subheader("Curva de respuesta fisiológica")
    temperatures = np.linspace(0.0, 45.0, 200)
    response = [
        calculate_tt_scalar(
            temperature,
            float(t_base),
            float(t_optimum),
            float(t_critical),
        )
        for temperature in temperatures
    ]
    response_figure = go.Figure()
    response_figure.add_trace(
        go.Scatter(
            x=temperatures,
            y=response,
            mode="lines",
            fill="tozeroy",
            name="Grados-día diarios",
        )
    )
    response_figure.update_layout(
        xaxis_title="Temperatura media (°C)",
        yaxis_title="Aporte térmico diario (°Cd)",
        height=380,
    )
    st.plotly_chart(response_figure, width="stretch")

    st.subheader("Calibrador biofísico 2D")
    st.caption(
        "Explora únicamente Wmax y cobertura/Ke. "
        "Latencia, termoinhibición, choque hídrico y primer pico "
        "permanecen fijos."
    )

    optimizer_columns = st.columns(4)
    coverage_step = optimizer_columns[0].number_input(
        "Paso cobertura (%)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )
    wmax_min = optimizer_columns[1].number_input(
        "Wmax mínimo",
        min_value=5.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )
    wmax_max = optimizer_columns[2].number_input(
        "Wmax máximo",
        min_value=6.0,
        max_value=60.0,
        value=30.0,
        step=1.0,
    )
    wmax_step = optimizer_columns[3].number_input(
        "Paso Wmax",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
    )

    run_optimizer = st.button(
        "Ejecutar barrido hídrico 2D",
        type="primary",
        width="stretch",
    )
    if run_optimizer:
        if field_raw is None:
            st.error(
                "Se necesitan observaciones de campo para optimizar."
            )
        elif float(wmax_max) <= float(wmax_min):
            st.error("Wmax máximo debe ser mayor que Wmax mínimo.")
        else:
            with st.spinner("Evaluando combinaciones Wmax × cobertura..."):
                coverage_values = np.arange(
                    0,
                    101,
                    int(coverage_step),
                    dtype=int,
                )
                wmax_values = np.arange(
                    float(wmax_min),
                    float(wmax_max) + 0.5 * float(wmax_step),
                    float(wmax_step),
                )
                optimizer_results = optimize_hydric_2d(
                    weather_raw,
                    field_raw,
                    ann_model,
                    coverage_values,
                    wmax_values,
                    float(thermoinhibition_threshold),
                    float(hydric_shock_threshold),
                )
                st.session_state["optimizer_results_integral"] = (
                    optimizer_results
                )

    optimizer_results = st.session_state.get(
        "optimizer_results_integral"
    )
    if isinstance(optimizer_results, pd.DataFrame):
        if optimizer_results.empty:
            st.warning("El barrido no produjo resultados.")
        else:
            best = optimizer_results.iloc[0]
            best_columns = st.columns(5)
            best_columns[0].metric(
                "Mejor cobertura",
                f"{int(best['Cobertura_pct'])}%",
            )
            best_columns[1].metric(
                "Mejor Wmax",
                f"{best['W_Max_mm']:.2f} mm",
            )
            best_columns[2].metric(
                "KGE",
                f"{best['KGE_Flujos']:.3f}",
            )
            best_columns[3].metric(
                "NSE",
                f"{best['NSE_Flujos']:.3f}",
            )
            best_columns[4].metric(
                "F1",
                f"{best['F1_Score']:.3f}",
            )
            st.dataframe(
                optimizer_results.head(30),
                width="stretch",
                hide_index=True,
            )

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
        "TMAX_suelo_diagnostica",
        "TMIN_suelo_diagnostica",
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
        "EMERREL_LOG",
        "DG",
    ]
    export_data = simulation[
        [
            column
            for column in export_columns
            if column in simulation.columns
        ]
    ].copy()

    st.download_button(
        "📥 Descargar simulación CSV",
        data=export_data.to_csv(index=False).encode("utf-8"),
        file_name="PREDWEEM_Lartigau_simulacion_integral.csv",
        mime="text/csv",
        width="stretch",
    )

    statistical_rows = {
        "Pearson_Flujos": metrics["Pearson_Flujos"],
        "NSE_Flujos": metrics["NSE_Flujos"],
        "KGE_Flujos": metrics["KGE_Flujos"],
        "RMSE_Acumulado": metrics["RMSE_Acumulado"],
        "CCC_Acumulado": metrics["CCC_Acumulado"],
        "R2_Acumulado": metrics["R2_Acumulado"],
        "F1_Score_Coincidencia": metrics[
            "F1_Score_Coincidencia"
        ],
        "Exactitud_Global": metrics["Exactitud_Global"],
        "Hits": metrics["Hits"],
        "Misses": metrics["Misses"],
        "Falsos_Positivos": metrics["Falsos_Positivos"],
        "Correctos_Negativos": metrics["Correctos_Negativos"],
        "PEC_Porcentaje": operational["PEC_Porcentaje"],
        "Lag_Control_vs_Pico_Campo_Dias": operational[
            "Lag_Control_vs_Pico_Campo_Dias"
        ],
        "Lead_Time_Dias": operational["Lead_Time_Dias"],
        "Desfase_T50_Dias": operational["Desfase_T50_Dias"],
        "Desfase_Primer_Flujo_Dias": operational[
            "Desfase_Primer_Flujo_Dias"
        ],
    }
    statistical_table = pd.DataFrame(
        {
            "Métrica": list(statistical_rows),
            "Valor": list(statistical_rows.values()),
        }
    )

    parameter_table = pd.DataFrame(
        {
            "Configuración": [
                "Version",
                "Cobertura_pct",
                "Wmax_mm",
                "Ke",
                "Modulador_termico_diagnostico",
                "Latencia_JD",
                "Ventana_termica_dias",
                "Umbral_termoinhibicion_C",
                "Choque_hidrico_mm_3d",
                "Umbral_primer_pico",
                "Persistencia_dias",
                "Lag_dias",
                "T_base",
                "T_optima",
                "T_critica",
                "TT_control",
                "TT_limite",
                "Residualidad_dias",
                "Fecha_primer_pico",
                "Fecha_control",
                "Fecha_limite",
            ],
            "Valor": [
                "vK4.9.15 Adaptada",
                coverage_percent,
                float(w_max_value),
                ke_value,
                thermal_modulator,
                LATENCIA_JD,
                VENTANA_TERMICA_DIAS,
                float(thermoinhibition_threshold),
                float(hydric_shock_threshold),
                UMBRAL_PRIMER_PICO,
                PERSISTENCIA_PRIMER_PICO_DIAS,
                0,
                float(t_base),
                float(t_optimum),
                float(t_critical),
                int(tt_control),
                int(tt_limit),
                int(residual_days),
                first_peak_date,
                control_date,
                limit_date,
            ],
        }
    )

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        export_data.to_excel(
            writer,
            sheet_name="Data_Diaria",
            index=False,
        )
        parameter_table.to_excel(
            writer,
            sheet_name="Bio_Params",
            index=False,
        )
        statistical_table.to_excel(
            writer,
            sheet_name="Validacion_Estadistica",
            index=False,
        )
        if field is not None:
            field.to_excel(
                writer,
                sheet_name="Campo_Validacion",
                index=False,
            )
        if not synchronized.empty:
            synchronized.to_excel(
                writer,
                sheet_name="Event_to_Event",
                index=False,
            )
        if isinstance(optimizer_results, pd.DataFrame):
            if not optimizer_results.empty:
                optimizer_results.to_excel(
                    writer,
                    sheet_name="Optimizador_2D",
                    index=False,
                )

    st.download_button(
        "📊 Descargar reporte integral Excel",
        data=excel_buffer.getvalue(),
        file_name="PREDWEEM_Integral_Lartigau_vK4_9_15_adaptada.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
    )

st.caption(
    "PREDWEEM by GUILLERMO R. CHANTRE · "
    "Modelo híbrido ANN + ecofisiología + validación Event-to-Event."
)
