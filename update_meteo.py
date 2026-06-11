
# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 NODOS CLIMÁTICOS PREDWEEM — SCON SENSOR LARTIGAU 2026
# Procesamiento e Integración Estricta de la Red MeteoBahía
# ===============================================================

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
import sys

# CONFIGURACIÓN DE RUTAS Y CONSTANTES
URL_XML = "https://meteobahia.com.ar/scripts/forecast/for-cf.xml"
ARCHIVO_CSV = Path("meteo_daily.csv")
CAMPANIA_START = datetime(2026, 1, 1).date()

def to_float(x):
    """Convierte strings del XML con coma decimal a floats limpios."""
    try:
        return float(str(x).replace(",", "."))
    except (ValueError, TypeError):
        return None

def fetch_meteobahia_dataframe():
    """Descarga el XML, corrige las etiquetas y devuelve un DataFrame limpio."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    response = requests.get(URL_XML, headers=headers, timeout=20)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    rows = []
    
    # Recorrido del árbol XML de MeteoBahía
    for d in root.findall(".//forecast/tabular/day"):
        fecha_str = d.find("fecha").get("value")  # Trae formato YYYY-MM-DD
        tmax = d.find("tmax").get("value")
        tmin = d.find("tmin").get("value")
        prec = d.find("precip").get("value")

        rows.append({
            "Fecha": pd.to_datetime(fecha_str),
            "TMAX": to_float(tmax),
            "TMIN": to_float(tmin),
            "Prec": to_float(prec),
        })

    if not rows:
        raise ValueError("El XML de MeteoBahía no contenía registros procesables.")

    df = pd.DataFrame(rows)
    return df

def actualizar_base_datos():
    """Lee el historial, fusiona con el XML aplicando la purga de duplicados por ISO-string."""
    # Seteamos zona horaria local argentina (ART = UTC-3) para evitar saltos nocturnos
    hoy_local = (datetime.utcnow() - timedelta(hours=3)).date()

    # 1) Control de inicio de campaña
    if hoy_local < CAMPANIA_START:
        print(f"⏳ Esperando fecha de inicio de campaña: {CAMPANIA_START}")
        return

    # 2) Blanqueo de ciclo anual (Punto de control 1 de Enero)
    if hoy_local == CAMPANIA_START and ARCHIVO_CSV.exists():
        ARCHIVO_CSV.unlink()
        print("🆕 Ciclo 2026: Historial previo reiniciado para nueva calibración.")

    # 3) Captura remota
    print("📡 Descargando datos frescos desde MeteoBahía...")
    df_nuevo = fetch_meteobahia_dataframe()

    # 4) Filtro de horizonte: Descartar ruidos predictivos más allá de 7 días
    limite_futuro = pd.Timestamp(hoy_local + timedelta(days=7))
    df_nuevo = df_nuevo[df_nuevo["Fecha"] <= limite_futuro].copy()

    # 5) Fusión lógica e ingeniería de datos (Merge)
    if ARCHIVO_CSV.exists():
        print(f"Leyendo historial existente desde {ARCHIVO_CSV}...")
        df_historico = pd.read_csv(ARCHIVO_CSV)
        
        # Forzamos conversión temporal a los dos bloques para un merge seguro
        df_historico["Fecha"] = pd.to_datetime(df_historico["Fecha"])
        df_nuevo["Fecha"] = pd.to_datetime(df_nuevo["Fecha"])
        
        # Concatenamos poniendo las actualizaciones al final
        df_final = pd.concat([df_historico, df_nuevo], ignore_index=True)
        
        # BLINDAJE CRÍTICO: Eliminamos duplicados basados en datetime real
        df_final = df_final.drop_duplicates(subset=["Fecha"], keep="last")
        df_final = df_final.sort_values(by="Fecha").reset_index(drop=True)
    else:
        print("📝 No se detectó historial previo. Creando archivo maestro...")
        df_final = df_nuevo

    # Purga de filas corruptas antes de la persistencia
    df_final = df_final.dropna(subset=["Fecha", "TMAX", "TMIN"])

    # SOLUCIÓN AL ERROR DE ACTUALIZACIÓN:
    # Forzamos la escritura en el CSV con formato string ISO estricto (sin horas 00:00:00)
    df_final["Fecha"] = df_final["Fecha"].dt.strftime("%Y-%m-%d")
    
    # Escritura física definitiva
    df_final.to_csv(ARCHIVO_CSV, index=False)
    print(f"✅ ¡Actualización completada! Base de datos sincronizada: {len(df_final)} registros.")
    print("\n📋 Últimas filas consolidadas en el archivo:")
    print(df_final.tail(7))

if __name__ == "__main__":
    try:
        actualizar_base_datos()
    except Exception as e:
        print(f"❌ Error crítico en el módulo de sincronización: {e}")
        sys.exit(1)
