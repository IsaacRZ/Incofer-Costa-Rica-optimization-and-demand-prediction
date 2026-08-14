"""
src/api/cliente_api.py

Clase ClienteAPI: hace peticiones a APIs publicas externas y transforma
sus respuestas en DataFrames listos para usar. Agrupa las dos fuentes que
necesita este proyecto (feriados de Costa Rica via Nager.Date, y clima
historico via Open-Meteo) bajo una unica entidad "cliente de APIs", en
vez de una clase por API.

Uso:
    from src.api.cliente_api import ClienteAPI

    cliente = ClienteAPI()
    df_feriados = cliente.obtener_feriados_cr(2021, 2026)
    df_clima = cliente.obtener_clima_historico("2021-01-18", "2026-06-30")

"""

import requests
import pandas as pd


class ClienteAPI:
    """Cliente generico para las APIs publicas del proyecto.

    Internamente separa dos responsabilidades:
    - `_get()`: el mecanismo HTTP compartido (timeout, manejo de errores).
    - `obtener_feriados_cr()` / `obtener_clima_historico()`: el parseo
      especifico de cada fuente, ya que Nager.Date y Open-Meteo devuelven
      formatos de JSON completamente distintos entre si
    """

    URL_FERIADOS = "https://date.nager.at/api/v3/PublicHolidays"
    URL_CLIMA = "https://archive-api.open-meteo.com/v1/archive"
    CODIGO_PAIS_FERIADOS = "CR"

    # Coordenadas de San Jose, usadas como referencia unica para el GAM
    # (las rutas de INCOFER caben en un radio pequeno).
    LATITUD_GAM = 9.93
    LONGITUD_GAM = -84.08
    VARIABLES_CLIMA_DIARIAS = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    ]

    def __init__(self, timeout_segundos: int = 30):
        self.timeout_segundos = timeout_segundos
        # Cache solo para feriados (se piden por anio, se repiten seguido
        # dentro de una misma sesion de trabajo). El clima se pide una vez
        # por rango completo, asi que no vale la pena cachearlo igual.
        self._cache_feriados: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------
    # Mecanismo HTTP compartido
    # ------------------------------------------------------------------
    def _get(self, url: str, params: dict | None = None) -> dict | list:
        """Hace un GET, valida el codigo de respuesta y devuelve el JSON
        ya parseado (dict o list, segun la forma que tenga la API)."""
        respuesta = requests.get(url, params=params, timeout=self.timeout_segundos)
        respuesta.raise_for_status()
        return respuesta.json()

    # ------------------------------------------------------------------
    # Feriados de Costa Rica (Nager.Date)
    # ------------------------------------------------------------------
    def obtener_feriados_cr(self, anio_inicio: int, anio_fin: int) -> pd.DataFrame:
        """Descarga los feriados oficiales de Costa Rica entre anio_inicio
        y anio_fin (ambos inclusive) y los devuelve como DataFrame tidy."""
        filas = []
        for anio in range(anio_inicio, anio_fin + 1):
            for feriado in self._obtener_feriados_de_un_anio(anio):
                filas.append({
                    "fecha": feriado["date"],
                    "nombre_local": feriado["localName"],
                    "nombre_ingles": feriado["name"],
                    "tipo": ", ".join(feriado.get("types", [])),
                })

        df = pd.DataFrame(filas)
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["es_feriado"] = True
        return df.sort_values("fecha").reset_index(drop=True)

    def _obtener_feriados_de_un_anio(self, anio: int) -> list[dict]:
        if anio not in self._cache_feriados:
            url = f"{self.URL_FERIADOS}/{anio}/{self.CODIGO_PAIS_FERIADOS}"
            self._cache_feriados[anio] = self._get(url)
        return self._cache_feriados[anio]

    # ------------------------------------------------------------------
    # Clima historico (Open-Meteo)
    # ------------------------------------------------------------------
    def obtener_clima_historico(
        self,
        fecha_inicio: str,
        fecha_fin: str,
        latitud: float | None = None,
        longitud: float | None = None,
    ) -> pd.DataFrame:
        """Descarga clima diario entre fecha_inicio y fecha_fin (formato
        "YYYY-MM-DD"). Sin coordenadas, usa San Jose como referencia del GAM."""
        params = {
            "latitude": latitud if latitud is not None else self.LATITUD_GAM,
            "longitude": longitud if longitud is not None else self.LONGITUD_GAM,
            "start_date": fecha_inicio,
            "end_date": fecha_fin,
            "daily": ",".join(self.VARIABLES_CLIMA_DIARIAS),
            "timezone": "America/Costa_Rica",
        }
        datos = self._get(self.URL_CLIMA, params=params)

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
    """Prueba rapida manual de ambos metodos."""
    cliente = ClienteAPI()

    print("=== Feriados CR 2021-2022 (muestra corta) ===")
    print(cliente.obtener_feriados_cr(2021, 2022).to_string())

    print("\n=== Clima Enero San Jose: 2026 (muestra corta) ===")
    print(cliente.obtener_clima_historico("2026-07-01", "2026-07-31").to_string())

    print("\n=== Clima Enero Cartago: 2026 (muestra corta) ===")
    clima_cartago = cliente.obtener_clima_historico("2026-07-01", "2026-07-31",9.864, -83.921).to_string()
    print(clima_cartago)

    print("\n=== Clima Enero Heredia: 2026 (muestra corta) ===")
    clima_heredia = cliente.obtener_clima_historico("2026-07-01", "2026-07-31",9.9981, -84.1170).to_string()
    print(clima_heredia)

if __name__ == "__main__":
    main()