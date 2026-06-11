
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configuración y constantes de Lartigau (Sudoeste de Buenos Aires)
URL = "https://meteobahia.com.ar/scripts/forecast/for-cf.xml"
OUT = Path("meteo_daily.csv")
START = datetime(2026, 1, 1).date()

def to_float(x):
    """Convierte strings con coma decimal a float."""
    try:
        return float(str(x).replace(",", "."))
    except (ValueError, TypeError):
        return None

def fetch_meteobahia():
    """Descarga y procesa el XML de MeteoBahía forzando tipado estricto."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    r = requests.get(URL, headers=headers, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    rows = []
    for d in root.findall(".//forecast/tabular/day"):
        fecha_str = d.find("fecha").get("value") # Formato esperado: YYYY-MM-DD
        tmax  = d.find("tmax").get("value")
        tmin  = d.find("tmin").get("value")
        prec  = d.find("precip").get("value")

        rows.append({
            "Fecha": pd.to_datetime(fecha_str),
            "TMAX": to_float(tmax),
            "TMIN": to_float(tmin),
            "Prec": to_float(prec),
        })

    df = pd.DataFrame(rows)
    # Limpieza de nulos en variables críticas
    df = df.dropna(subset=["Fecha", "TMAX", "TMIN"]).sort_values("Fecha")
    return df

def update_file():
    """Lee el CSV, aplica la ventana híbrida y guarda con formato ISO estricto."""
    # Obtenemos la fecha local argentina actual (ART = UTC-3)
    today_local = (datetime.utcnow() - timedelta(hours=3)).date()

    # 1) Restricción de fecha de inicio de campaña
    if today_local < START:
        print(f"⏳ Esperando al {START} para iniciar las actualizaciones de Lartigau.")
        return

    # 2) Reinicio anual controlado
    if today_local == START:
        if OUT.exists():
            OUT.unlink()
            print("🆕 Ciclo 2026 detectado: Historial reiniciado para calibración limpia.")

    # 3) Descarga de la ventana actual de MeteoBahía
    print("📡 Descargando datos desde MeteoBahía...")
    df_new = fetch_meteobahia()

    # Acotamos df_new a un máximo de 7 días hacia adelante desde hoy para evitar ruidos de largo plazo
    fecha_limite_futuro = pd.Timestamp(today_local + timedelta(days=7))
    df_new = df_new[df_new["Fecha"] <= fecha_limite_futuro].copy()

    # 4) Fusión de registros (Merge)
    if OUT.exists():
        print(f"Leyendo historial de Lartigau desde {OUT}...")
        df_old = pd.read_csv(OUT)
        df_old["Fecha"] = pd.to_datetime(df_old["Fecha"])
        
        # Concatenación. El df_new pisa los datos previos (observados parciales o pronósticos)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        
        # Eliminamos duplicados reales asegurando consistencia de datetime
        df_all = df_all.drop_duplicates(subset=["Fecha"], keep="last")
        df_all = df_all.sort_values("Fecha").reset_index(drop=True)
        print("🔄 Registros actualizados con la última observación/asimilación de MeteoBahía.")
    else:
        df_all = df_new
        print(f"📝 Inicializando base de datos meteorológica para Lartigau...")

    # 5) Persistencia en disco con formato ISO estricto (Evita baches de strings al re-leer)
    df_all["Fecha"] = df_all["Fecha"].dt.strftime("%Y-%m-%d")
    df_all.to_csv(OUT, index=False)
    
    print(f"[OK] Sincronización exitosa. Total de días registrados: {len(df_all)}.")
    print("Últimos 5 registros consolidados:")
    print(df_all.tail(5))

if __name__ == "__main__":
    try:
        update_file()
    except Exception as e:
        print(f"❌ Error durante la ejecución del nodo Lartigau: {e}")
        sys.exit(1)
