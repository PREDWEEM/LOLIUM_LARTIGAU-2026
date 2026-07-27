# -*- coding: utf-8 -*-
"""Meteorología trazable PREDWEEM Lartigau 2026.

ERA5-Land = reanálisis histórico; ECMWF IFS = puente provisional;
MeteoBahía/Coronel Falcón = pronóstico desde hoy. El CSV legado formado
por pronósticos vencidos se archiva una sola vez y no se reutiliza.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

LATITUD = float(os.getenv("LATITUD", "-38.6166"))
LONGITUD = float(os.getenv("LONGITUD", "-61.7000"))
ZONA_HORARIA = "America/Argentina/Buenos_Aires"
CAMPANIA_START = date(2026, 1, 1)
RETARDO_ERA5_LAND_DIAS = int(os.getenv("RETARDO_ERA5_LAND_DIAS", "5"))
TBASE = 2.0
URL_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
URL_XML = "https://meteobahia.com.ar/scripts/forecast/for-cf.xml"
ARCHIVO_MAESTRO = Path("meteo_daily.csv")
ARCHIVO_ERA5 = Path("data/era5_land_lartigau.csv")
ARCHIVO_ESTADO = Path("data/estado_actualizacion_meteo.json")
ARCHIVO_LEGACY = Path("data/meteo_falcon_pronosticos_archivados_2026.csv")
DIR_PRONOSTICOS = Path("data/historico_pronosticos")
COLUMNAS = [
    "Fecha", "TMAX", "TMIN", "Prec", "TMEDIA", "GD_Tb2", "Fuente",
    "TipoDato", "CalidadDato", "Latitud_grilla", "Longitud_grilla",
    "Elevacion_grilla_m", "Emision_UTC",
]


def hoy_argentina() -> date:
    return datetime.now(ZoneInfo(ZONA_HORARIA)).date()


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get(url: str, *, params=None, headers=None, timeout=90) -> requests.Response:
    ultimo: Exception | None = None
    for intento in range(1, 5):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as error:
            ultimo = error
            print(f"⚠️ HTTP {intento}/4: {error}")
            if intento < 4:
                time.sleep(5 * intento)
    raise RuntimeError(f"No fue posible consultar {url}") from ultimo


def columnas(df: pd.DataFrame) -> pd.DataFrame:
    salida = df.copy()
    for nombre in COLUMNAS:
        if nombre not in salida:
            salida[nombre] = pd.NA
    return salida[COLUMNAS]


def escribir(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.3f")
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp.replace(path)


def diario(
    fechas, tmax, tmin, tmedia, prec, *, fuente, tipo, calidad, payload
) -> pd.DataFrame:
    df = pd.DataFrame({
        "Fecha": pd.to_datetime(pd.Series(fechas), errors="coerce"),
        "TMAX": pd.to_numeric(pd.Series(tmax), errors="coerce"),
        "TMIN": pd.to_numeric(pd.Series(tmin), errors="coerce"),
        "Prec": pd.to_numeric(pd.Series(prec), errors="coerce"),
    })
    df["TMEDIA"] = (
        (df["TMAX"] + df["TMIN"]) / 2
        if tmedia is None
        else pd.to_numeric(pd.Series(tmedia), errors="coerce")
    )
    derivar = df["TMEDIA"].isna() & df["TMAX"].notna() & df["TMIN"].notna()
    df.loc[derivar, "TMEDIA"] = (df.loc[derivar, "TMAX"] + df.loc[derivar, "TMIN"]) / 2
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN", "TMEDIA", "Prec"])
    df["Fecha"] = df["Fecha"].dt.normalize()
    df = df.loc[
        df["TMAX"].between(-25, 55) & df["TMIN"].between(-35, 45)
        & df["TMEDIA"].between(-35, 55) & (df["TMAX"] >= df["TMIN"])
        & df["Prec"].between(0, 500)
    ].copy()
    df["GD_Tb2"] = (df["TMEDIA"] - TBASE).clip(lower=0)
    df["Fuente"], df["TipoDato"], df["CalidadDato"] = fuente, tipo, calidad
    df["Latitud_grilla"] = payload.get("latitude", pd.NA)
    df["Longitud_grilla"] = payload.get("longitude", pd.NA)
    df["Elevacion_grilla_m"] = payload.get("elevation", pd.NA)
    df["Emision_UTC"] = utc_iso()
    df["Fecha"] = df["Fecha"].dt.strftime("%Y-%m-%d")
    return columnas(df.drop_duplicates("Fecha", keep="last").sort_values("Fecha").reset_index(drop=True))


def open_meteo(inicio: date, fin: date, modelo: str, fuente: str, tipo: str, calidad: str) -> pd.DataFrame:
    if inicio > fin:
        return pd.DataFrame(columns=COLUMNAS)
    params = {
        "latitude": LATITUD, "longitude": LONGITUD,
        "start_date": inicio.isoformat(), "end_date": fin.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum",
        "models": modelo, "timezone": ZONA_HORARIA,
        "temperature_unit": "celsius", "precipitation_unit": "mm",
        "cell_selection": "land",
    }
    payload = get(URL_ARCHIVE, params=params).json()
    d = payload.get("daily", {})
    requeridas = {"time", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"}
    if requeridas.difference(d):
        raise ValueError(f"{modelo} no devolvió todas las variables diarias.")
    salida = diario(
        d["time"], d["temperature_2m_max"], d["temperature_2m_min"],
        d.get("temperature_2m_mean"), d["precipitation_sum"],
        fuente=fuente, tipo=tipo, calidad=calidad, payload=payload,
    )
    if salida.empty:
        raise ValueError(f"{modelo} no devolvió días válidos entre {inicio} y {fin}.")
    return salida


def descargar_era5(inicio: date, fin: date) -> pd.DataFrame:
    print(f"🌍 ERA5-Land: {inicio} a {fin}")
    return open_meteo(inicio, fin, "era5_land", "ERA5_LAND", "Reanalisis", "Reanalisis_grilla_0.1_sin_correccion_local")


def leer_cache_era5() -> pd.DataFrame:
    if not ARCHIVO_ERA5.exists():
        raise FileNotFoundError("No existe caché ERA5-Land.")
    df = columnas(pd.read_csv(ARCHIVO_ERA5))
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Fecha"]).drop_duplicates("Fecha", keep="last")
    df = df.loc[df["Fuente"].astype(str).eq("ERA5_LAND")].copy()
    if df.empty:
        raise ValueError("La caché ERA5-Land no contiene filas válidas.")
    return df.sort_values("Fecha").reset_index(drop=True)


def obtener_era5(inicio: date, fin: date) -> tuple[pd.DataFrame, str]:
    try:
        df = descargar_era5(inicio, fin)
        escribir(df, ARCHIVO_ERA5)
        return df, "ERA5_Land_remoto"
    except Exception as error:
        print(f"⚠️ Falló ERA5-Land remoto: {error}")
        return leer_cache_era5(), "ERA5_Land_cache"


def puente(inicio: date, fin: date) -> pd.DataFrame:
    print(f"🧩 ECMWF IFS provisional: {inicio} a {fin}")
    return open_meteo(inicio, fin, "ecmwf_ifs", "ECMWF_IFS_HISTORICO", "Provisional", "Provisional_hasta_disponibilidad_ERA5_Land")


def numero(valor: Any) -> float | None:
    try:
        texto = str(valor).strip().replace(",", ".")
        return float(texto) if texto else None
    except (TypeError, ValueError):
        return None


def meteobahia() -> pd.DataFrame:
    print("📡 MeteoBahía XML / Coronel Falcón")
    r = get(URL_XML, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://meteobahia.com.ar/"}, timeout=30)
    filas = []
    for d in ET.fromstring(r.content).findall(".//forecast/tabular/day"):
        def valor(tag: str):
            nodo = d.find(f"./{tag}")
            return nodo.get("value") if nodo is not None else None
        filas.append({"Fecha": valor("fecha"), "TMAX": numero(valor("tmax")), "TMIN": numero(valor("tmin")), "Prec": numero(valor("precip"))})
    if not filas:
        raise ValueError("El XML de MeteoBahía no contiene días procesables.")
    x = pd.DataFrame(filas)
    df = diario(x.Fecha, x.TMAX, x.TMIN, None, x.Prec, fuente="METEOBAHIA_XML_CORONEL_FALCON", tipo="Pronostico", calidad="Pronostico_deterministico_Coronel_Falcon", payload={})
    hoy = hoy_argentina()
    df = df.loc[pd.to_datetime(df.Fecha).dt.date >= hoy].copy()
    if df.empty or pd.to_datetime(df.Fecha).min().date() != hoy:
        raise ValueError("MeteoBahía no incluye la fecha actual.")
    DIR_PRONOSTICOS.mkdir(parents=True, exist_ok=True)
    marca = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    escribir(df, DIR_PRONOSTICOS / f"meteobahia_coronel_falcon_{marca}.csv")
    return df


def faltantes(df: pd.DataFrame, inicio: date, fin: date) -> list[date]:
    if inicio > fin:
        return []
    esperadas = pd.date_range(inicio, fin, freq="D")
    presentes = pd.DatetimeIndex(pd.to_datetime(df.Fecha, errors="coerce").dropna()).normalize()
    return [x.date() for x in esperadas.difference(presentes)]


def rangos(fechas: list[date]) -> list[tuple[date, date]]:
    if not fechas:
        return []
    fechas = sorted(set(fechas)); salida = []; inicio = anterior = fechas[0]
    for actual in fechas[1:]:
        if actual == anterior + timedelta(days=1):
            anterior = actual
        else:
            salida.append((inicio, anterior)); inicio = anterior = actual
    salida.append((inicio, anterior))
    return salida


def validar(df: pd.DataFrame, hoy: date, fin: date) -> None:
    fechas = pd.to_datetime(df.Fecha, errors="coerce")
    if df.empty or fechas.isna().any() or fechas.duplicated().any():
        raise ValueError("La serie está vacía o contiene fechas inválidas/duplicadas.")
    c = df[["TMAX", "TMIN", "TMEDIA", "Prec"]].apply(pd.to_numeric, errors="coerce")
    if c.isna().any().any() or (c.TMAX < c.TMIN).any() or (c.Prec < 0).any():
        raise ValueError("Hay valores meteorológicos nulos o físicamente inválidos.")
    huecos = faltantes(df, CAMPANIA_START, fin)
    if huecos:
        raise ValueError("La serie no es continua: " + ", ".join(x.isoformat() for x in huecos[:20]))
    pasadas, futuras = fechas.dt.date < hoy, fechas.dt.date >= hoy
    if not futuras.any():
        raise ValueError("No hay pronóstico desde hoy.")
    if df.loc[pasadas, "TipoDato"].astype(str).eq("Pronostico").any():
        raise ValueError("Persisten pronósticos vencidos en el histórico.")
    if not df.loc[futuras, "Fuente"].astype(str).eq("METEOBAHIA_XML_CORONEL_FALCON").all():
        raise ValueError("El tramo futuro no proviene exclusivamente de MeteoBahía.")


def archivar_legacy() -> bool:
    if ARCHIVO_LEGACY.exists() or not ARCHIVO_MAESTRO.exists():
        return False
    if "Fuente" in pd.read_csv(ARCHIVO_MAESTRO, nrows=0).columns:
        return False
    ARCHIVO_LEGACY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARCHIVO_MAESTRO, ARCHIVO_LEGACY)
    print(f"🗄️ Serie antigua archivada en {ARCHIVO_LEGACY}")
    return True


def ejecutar() -> pd.DataFrame:
    legacy_creado = archivar_legacy()
    hoy, ayer = hoy_argentina(), hoy_argentina() - timedelta(days=1)
    era5, estado_era5 = obtener_era5(CAMPANIA_START, hoy - timedelta(days=RETARDO_ERA5_LAND_DIAS))
    huecos = faltantes(era5, CAMPANIA_START, ayer)
    rs = rangos(huecos)
    bloques = [puente(i, f) for i, f in rs]
    prov = columnas(pd.concat(bloques, ignore_index=True)) if bloques else pd.DataFrame(columns=COLUMNAS)
    pron = meteobahia()
    total = columnas(pd.concat([era5, prov, pron], ignore_index=True))
    total["Fecha_dt"] = pd.to_datetime(total.Fecha, errors="coerce")
    total["_p"] = total.TipoDato.map({"Reanalisis": 0, "Provisional": 1, "Pronostico": 2}).fillna(9)
    total = total.dropna(subset=["Fecha_dt"]).sort_values(["Fecha_dt", "_p"]).drop_duplicates("Fecha_dt", keep="first").sort_values("Fecha_dt")
    fin = total.Fecha_dt.max().date()
    total = total.loc[(total.Fecha_dt.dt.date >= CAMPANIA_START) & (total.Fecha_dt.dt.date <= fin)].copy()
    total["Fecha"] = total.Fecha_dt.dt.strftime("%Y-%m-%d")
    total = columnas(total.drop(columns=["Fecha_dt", "_p"])).reset_index(drop=True)
    validar(total, hoy, fin)
    escribir(total, ARCHIVO_MAESTRO)
    estado = {
        "ejecucion_utc": utc_iso(), "sitio": "Lartigau", "latitud": LATITUD,
        "longitud": LONGITUD, "inicio_campania": CAMPANIA_START.isoformat(),
        "fuente_historica": "ERA5_LAND", "estado_era5_land": estado_era5,
        "tipo_historico": "Reanalisis", "fin_era5_land": str(era5.Fecha.max()),
        "retardo_era5_land_dias": RETARDO_ERA5_LAND_DIAS,
        "fuente_puente": "ECMWF_IFS_HISTORICO" if len(prov) else None,
        "rangos_provisionales": [{"inicio": i.isoformat(), "fin": f.isoformat()} for i, f in rs],
        "filas_provisionales": len(prov),
        "fuente_pronostico": "METEOBAHIA_XML_CORONEL_FALCON",
        "inicio_pronostico": str(pron.Fecha.min()), "fin_pronostico": str(pron.Fecha.max()),
        "archivo_pronosticos_legacy": str(ARCHIVO_LEGACY) if ARCHIVO_LEGACY.exists() else None,
        "archivo_pronosticos_legacy_creado_en_esta_ejecucion": legacy_creado,
        "huecos_finales": [x.isoformat() for x in faltantes(total, CAMPANIA_START, fin)],
        "advertencia": "ERA5-Land es reanalisis de grilla, no observacion de estacion. MeteoBahia XML se usa solo como pronostico.",
    }
    ARCHIVO_ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVO_ESTADO.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ ERA5={len(era5)}; provisional={len(prov)}; MeteoBahía={len(pron)}; total={len(total)}")
    return total


if __name__ == "__main__":
    try:
        ejecutar()
    except Exception as error:
        print(f"❌ Error: {error}. No se reemplazó meteo_daily.csv.", file=sys.stderr)
        raise SystemExit(1)
