# -*- coding: utf-8 -*-
"""Compatibilidad: descarga solo el pronóstico vigente de Coronel Falcón.

Este módulo ya no fusiona pronósticos vencidos dentro de meteo_daily.csv.
La consolidación operativa se realiza exclusivamente en update_meteo.py.
"""
from update_meteo import meteobahia


if __name__ == "__main__":
    pronostico = meteobahia()
    print(
        "✅ Pronóstico MeteoBahía descargado sin modificar el histórico: "
        f"{pronostico['Fecha'].min()} a {pronostico['Fecha'].max()}"
    )
