"""
src/api/feriados.py

Cliente para API publica de feriados en Costarica (Nager.Date:
https://date.nager.at). Descarga los feriados oficiales por anio y los
devuelve como un DataFrame listo para unir (merge) con el dataset de
viajes por fecha.
 
Uso:
    from src.api.feriados import ClienteFeriadosCR
 
    cliente = ClienteFeriadosCR()
    df_feriados = cliente.obtener_rango(2021, 2026)
"""

import requests
import pandas as pd

class ClienteFeriadosCR:
    """Descarga y guarda en caché los feriados oficiales de Costa Rica por anio,
    usando la API publica de Nager.Date.

    El cacheo es importante por el rango de anios 2021-2026 para no
    golpear la API por cada anio descargado en esta sesión.
    """

    URL_BASE = "https://date.nager.at/api/v3/PublicHolidays"
    CODIGO_PAIS = "CR"

    def __init__(self, timeout_segundos: int = 10):
        self.timeout_segundos = timeout_segundos
        # Cache en memoria: {anio: lista_de_feriados_crudos_del_JSON}.
        # Vive mientras exista el objeto (no persiste entre ejecuciones).
        self._cache: dict[int, list[dict]] = {}

    def obtener_anio(self, anio: int) -> list[dict]:
        """Devuelve lista cruda de feriados (tal como llega el JSON)
        para un anio en especifico. Usa el cache si ya se solicitó el anio
        """
        if anio in self._cache:
            return self._cache[anio]

        url = f"{self.URL_BASE}/{anio}/{self.CODIGO_PAIS}"
        respuesta = requests.get(url, timeout=self.timeout_segundos)

        respuesta.raise_for_status()

        datos = respuesta.json()
        self._cache[anio] = datos
        return datos

    def obtener_rango (self, anio_inicio: int, anio_fin: int) -> pd.DataFrame:
        """Descarga varios anios y los devuelve en un DF
           único tidy, un-feriado-por-fila. """
        filas = []
        for anio in range(anio_inicio, anio_fin + 1):
            for feriado in self.obtener_anio(anio):
                filas.append({
                    "fecha": feriado["date"],
                    "nombre_local": feriado["localName"],
                    "nombre_ingles": feriado["name"],
                    "tipo": ", ".join(feriado.get("types", [])),
                })

        df = pd.DataFrame(filas)
        df["fecha"] = pd.to_datetime(df["fecha"])
        # Crear columna booleana para falicilar merge con
        # viajes diarios (left join + fillna(False) para los dias sin 
        # feriado)
        df["es_feriado"] = True

        return df.sort_values("fecha").reset_index(drop=True)

def main():
    """Prueba rapida manual: descarga feriados 2021-2026 y los imprime.
    Util para correr `uv run python src/api/feriados.py` y confirmar a ojo
    que la API responde y el parseo funciona, sin necesidad de un notebook.
    """
    cliente = ClienteFeriadosCR()
    df = cliente.obtener_rango(2021, 2026)
    print(df.to_string())
    print(f"\nTotal de feriados descargados: {len(df)}")
 
 
if __name__ == "__main__":
    main()
