"""
src/helpers/utilidades.py
 
Clase Utilidades: funciones auxiliares reutilizables para el proyecto
(validaciones, formateo, y combinacion de datasets). Aqui vive la logica
que une viajes + feriados + clima en un unico DataFrame enriquecido,
porque no es responsabilidad de GestorDatos (que solo conoce el CSV de
ARESEP) ni de ClienteAPI (que solo sabe traer datos crudos de cada API).
 
Uso:
    from src.helpers.utilidades import Utilidades
    from src.api.cliente_api import ClienteAPI
 
    cliente = ClienteAPI()
    df_feriados = cliente.obtener_feriados_cr(2021, 2026)
 
    # Una llamada de clima por ciudad relevante (coordenadas reales del GAM)
    clima_sj = cliente.obtener_clima_historico("2021-01-18", "2026-06-30", latitud=9.9325, longitud=-84.0795)
    clima_heredia = cliente.obtener_clima_historico("2021-01-18", "2026-06-30", latitud=9.9989, longitud=-84.1165)
    clima_cartago = cliente.obtener_clima_historico("2021-01-18", "2026-06-30", latitud=9.8644, longitud=-83.9195)
 
    df_clima = Utilidades.construir_clima_multiciudad({
        "San Jose": clima_sj,
        "Heredia": clima_heredia,
        "Cartago": clima_cartago,
    })
 
    df_final = Utilidades.enriquecer_viajes(df_viajes, df_feriados, df_clima)
"""


import pandas as pd


class Utilidades:
    """Funciones auxiliares reutilizables en distintas partes del proyecto."""
 
    # Mapeo codigo_recorrido -> ciudad climatica de referencia.
    #
    # Se asigna UNA sola ciudad por recorrido (no ciudad_salida +
    # ciudad_entrada por separado): los trayectos de INCOFER en el GAM son
    # cortos y ciudades vecinas comparten microclima (ej. Alajuela ~
    # Heredia, ambas Valle Central oeste), asi que separar origen/destino
    # duplicaria columnas sin aportar señal real al modelo.
    #
    # ADVERTENCIA: para T4464/T4468/T4469/T4470/T4471 se usa la ciudad de
    # DESTINO como proxy climatico (no la de origen, que en la mayoria de
    # los casos es San Jose -- Estacion del Pacifico o del Atlantico). El
    # criterio es que estos son trenes de comuteo hacia/desde esas provincias,
    # por lo que el clima de destino es razonablemente representativo del
    # trayecto tipico. T4465-T4467 confirmados como San Jose puro (no cruzan
    # provincia). Alajuela (T4471) usa el clima de Heredia como proxy, dado
    # que ambas provincias comparten microclima en el Valle Central oeste.
    MAPA_RECORRIDO_CIUDAD = {
        "T4464": "Heredia",   # Estacion del Pacifico - San Antonio de Belen (Belen colinda con Heredia)
        "T4465": "San Jose",  # Curridabat (Indoor Club) - Universidad Latina - Pavas (Metropoli III) -- confirmado San Jose
        "T4466": "San Jose",  # Curridabat (Indoor Club) - Universidad Latina - Estacion del Pacifico -- confirmado San Jose
        "T4467": "San Jose",  # Estacion del Pacifico - Pavas (Metropoli III) -- confirmado San Jose
        "T4468": "Heredia",   # Universidad Latina (sede San Pedro, San Jose) - Heredia: destino Heredia
        "T4469": "Heredia",   # Estacion del Atlantico (San Jose) - Heredia: destino Heredia
        "T4470": "Cartago",   # Estacion del Atlantico (San Jose) - Cartago: destino Cartago
        "T4471": "Heredia",   # Alajuela - Heredia Y Viceversa: se usa clima de Heredia como proxy de Alajuela
    }


    @staticmethod
    def construir_clima_multiciudad(clima_por_ciudad: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Concatenar varios DataFrames de clima (uno por ciudad) en un unico DataFrame
        con una columna 'ciudad' que identifica donde salió cada fila
        """
        partes = []
        for ciudad, df in clima_por_ciudad.items():
            parte = df.copy()
            parte["ciudad"] = ciudad
            partes.append(parte)
        return pd.concat(partes, ignore_index=True)

    @staticmethod
    def enriquecer_viajes(
        df_viajes: pd.DataFrame,
        df_feriados: pd.DataFrame,
        df_clima_por_ciudad: pd.DataFrame,
        mapa_recorrido_ciudad: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """Une viajes_diarios (de GestorDatos) con feriados y clima (de
        ClienteAPI + ciudad de referencia del recorrido), ambos por la columna 'fecha'.
        
        df_clima_por_ciudad debe venir de construir_clima_multiciudad(), con
        columnas: fecha, ciudad, temp_max_c, temp_min_c, precipitacion_mm.

        Usa left join en los dos merges: queremos conservar TODAS las filas
        de viajes, incluso si un dia no tiene feriado (es la mayoria de los
        casos) o si por algun motivo faltara el dato de clima ese dia.
        """
        mapa = mapa_recorrido_ciudad or Utilidades.MAPA_RECORRIDO_CIUDAD

        df = df_viajes.copy()
        df["ciudad_clima"] = df["codigo_recorrido"].map(mapa)

        sin_mapeo = df["ciudad_clima"].isna().sum()
        if sin_mapeo:
            ejemplos = sorted(df.loc[df["ciudad_clima"].isna(), "codigo_recorrido"].unique())
            raise ValueError(
                f"{sin_mapeo} filas sin ciudad climatica asignada. "
                f"Agrega estos codigo_recorrido a MAPA_RECORRIDO_CIUDAD: {ejemplos}"
            )

        df = df.merge(
            df_feriados[["fecha", "nombre_local", "es_feriado"]],
            on="fecha",
            how="left",

        )

        # Los dias NO feriados aparecen como NAN -> se convierten a False.
        df["es_feriado"] = df["es_feriado"].fillna(False)
        df["nombre_local"] = df["nombre_local"].fillna("")

        # Merge clima por fecha, ciudad
        # Cada recorrido se una contra el clima de SU ciudad de referencia/salida.

        df = df.merge(
            df_clima_por_ciudad[["fecha", "ciudad", "temp_max_c", "temp_min_c", "precipitacion_mm"]],
            left_on=["fecha", "ciudad_clima"],
            right_on=["fecha", "ciudad"],
            how="left",
        )
        df = df.drop(columns=["ciudad"])
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