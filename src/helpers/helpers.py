"""
src/helpers/helpers.py
Inicio Helpers  


Clase Utilidades: funciones auxiliares reutilizables para el proyecto
(validaciones, formateo, y combinacion de datasets). Aqui vive la logica
que une viajes + feriados + clima en un unico DataFrame enriquecido,
porque no es responsabilidad de GestorDatos (que solo conoce el CSV de
ARESEP) ni de ClienteAPI (que solo sabe traer datos crudos de cada API).

Uso:
    from src.helpers.utilidades import Utilidades

    util = Utilidades()
    df_final = util.enriquecer_viajes(df_viajes, df_feriados, df_clima)
    
"""

import pandas as pd


class Utilidades:
    """Funciones auxiliares reutilizables en distintas partes del proyecto."""

    @staticmethod
    def enriquecer_viajes(
        df_viajes: pd.DataFrame,
        df_feriados: pd.DataFrame,
        df_clima: pd.DataFrame,
    ) -> pd.DataFrame:
        """Une viajes_diarios (de GestorDatos) con feriados y clima (de
        ClienteAPI), ambos por la columna 'fecha'.

        Usa left join en los dos merges: queremos conservar TODAS las filas
        de viajes, incluso si un dia no tiene feriado (es la mayoria de los
        casos) o si por algun motivo faltara el dato de clima ese dia.
        """
        df = df_viajes.merge(
            df_feriados[["fecha", "nombre_local", "es_feriado"]],
            on="fecha",
            how="left",
        )
        # Los dias que NO son feriado no aparecen en df_feriados, asi que
        # el merge deja NaN en "es_feriado" para esas filas. Se convierte
        # explicitamente a False (en vez de dejar NaN, que pandas trataria
        # como "verdadero" en un chequeo booleano ingenuo).
        df["es_feriado"] = df["es_feriado"].fillna(False)
        df["nombre_local"] = df["nombre_local"].fillna("")

        df = df.merge(
            df_clima[["fecha", "temp_max_c", "temp_min_c", "precipitacion_mm"]],
            on="fecha",
            how="left",
        )
        return df

    @staticmethod
    def validar_columnas(df: pd.DataFrame, columnas_requeridas: list[str]) -> None:
        """Lanza ValueError si a df le falta alguna columna requerida.
        Util como chequeo de entrada al inicio de un metodo de EDA o de
        entrenamiento de modelo, para fallar rapido con un mensaje claro."""
        faltantes = set(columnas_requeridas) - set(df.columns)
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    @staticmethod
    def formatear_porcentaje(valor: float, decimales: int = 1) -> str:
        """Formatea un valor 0-1 como string de porcentaje, ej. 0.055 -> '5.5%'."""
        return f"{valor * 100:.{decimales}f}%"