"""
src/api/clima.py

Cliente para API publica de clima historica Open-Meteo
(https://open-meteo.com), endpoint /v1/archive. Descarga temperatura y
precipitacion diaria para un rango de fechas y coordenadas, y las
devuelve como un DataFrame listo para unir con el dataset de viajes.
 
Uso:
    from src.api.clima import ClienteClima
 
    cliente = ClienteClima()
    df_clima = cliente.obtener_rango("2021-01-18", "2026-06-30")
"""

import requests
import pandas as pd

class ClienteClima:
    """Descarga clima historico diario (temp max/min, precipitación)
    desde Open-Meteo para una ubicación fija
    
    La API acepta un rango de fechas en una sola llamada"""

    URL_BASE = "https://archive-api.open-meteo.com/v1/archive"

    # Coordenadas de San Jose, CR usadas como referencia para toda la GAM
    # Las rutas del INCOFER caben en un radio pequeño, por lo que una sola ubicación es una aproximación
    # razonable para toda el GAM

    LATITUD_GAM = 9.93
    LONGITUD_GAM = -84.08

    VARIABLES_DIARIAS = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",        
    ]

    def __init__(self, timeout_segundos: int = 30):
        self.timeout_segundos = timeout_segundos


    def obtener_rango(self, 
        fecha_inicio: str, 
        fecha_fin: str, 
        latitud: float | None = None,
        longitud: float | None = None,
        ) -> pd.DataFrame:
        """Descarga clima diario entre fecha_inicio y fecha_fin (formato "YYYY-MM-DD")
        Usa las coordenadas de San José por defecto."""
        params = {
            "latitude": latitud if latitud is not None else self.LATITUD_GAM,
            "longitude": longitud if longitud is not None else self.LONGITUD_GAM,
            "start_date": fecha_inicio,
            "end_date": fecha_fin,
            "daily": ",".join(self.VARIABLES_DIARIAS),
            "timezone": "America/Costa_Rica",
        }

        respuesta = requests.get(self.URL_BASE, params=params, timeout=self.timeout_segundos)
        respuesta.raise_for_status()

        datos = respuesta.json()

        df = pd.DataFrame(datos["daily"])
        df = df.rename(columns={
            "time": "fecha",
            "temperature_2m_max": "temp_max_c",
            "temperature_2m_min": "temp_min_c",
            "precipitation_sum": "precipitacion_mm",
        })
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df

def main():
    """Prueba rapida manual: descarga clima en un rango corto"""
    cliente = ClienteClima()
    df = cliente.obtener_rango("2021-01-01", "2021-01-20")
    print(df.to_string())

if __name__ == "__main__":
    main()