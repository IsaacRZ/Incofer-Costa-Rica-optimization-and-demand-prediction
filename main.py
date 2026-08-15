"""
main.py

Punto de entrada del proyecto. Orquesta el pipeline completo:

    1. GestorDatos       -> limpia el CSV crudo de ARESEP
    2. ClienteAPI         -> descarga feriados CR y clima (3 ciudades)
    3. Utilidades          -> une todo en un DataFrame enriquecido
    4. GestorBaseDatos    -> carga el resultado a la base de datos
    5. ProcesadorEDA      -> (TODO) analisis exploratorio
    6. Visualizador        -> (TODO) graficos
    7. ModeloML            -> (TODO) entrenamiento de modelos

Cada clase se instancia UNA vez aqui, con sus dependencias inyectadas
donde corresponde -- ninguna clase crea instancias de las otras por su
cuenta, para poder probarlas por separado (como hemos hecho hasta ahora
con mocks) sin tener que correr el pipeline completo cada vez.

Uso:
    uv run python main.py
"""

from src.datos.gestor_datos import GestorDatos
from src.api.cliente_api import ClienteAPI
from src.helpers.utilidades import Utilidades
from src.basedatos.gestor_basedatos import GestorBaseDatos

# --- Configuracion del pipeline (ajustar segun el entorno) ---
RUTA_CSV_CRUDO = "data/raw/aresep_tren.csv"
RUTA_PARQUET_LIMPIO = "data/processed/viajes_diarios.parquet"

ANIO_INICIO_FERIADOS = 2021
ANIO_FIN_FERIADOS = 2026

# Coordenadas de referencia por ciudad, usadas para el clima.
COORDENADAS_CIUDADES = {
    "San Jose": (9.9325, -84.0795),
    "Heredia": (9.9989, -84.1165),
    "Cartago": (9.8644, -83.9195),
}

# Cadena de conexion a la base de datos. Cambiar a Postgres/Timescale
# cuando este levantado con `docker compose up -d`:
#   "postgresql+psycopg2://incofer:incofer_dev_password@localhost:5432/incofer"
# Cadena de conexion a la base de datos. Requiere `docker compose up -d`
# levantado (ver docker-compose.yml en la raiz del proyecto) -- Postgres +
# TimescaleDB es el motor elegido para este proyecto por las hypertables.
CONNECTION_STRING_DB = "postgresql+psycopg2://incofer:incofer_dev_password@localhost:5432/incofer"
NOMBRE_TABLA_DB = "viajes_diarios"


def main():
    # --- 1. Limpieza del CSV crudo ---
    gestor_datos = GestorDatos(RUTA_CSV_CRUDO)
    df_viajes = gestor_datos.ejecutar()
    gestor_datos.resumen_calidad()
    gestor_datos.guardar(df_viajes, RUTA_PARQUET_LIMPIO)

    # --- 2. Descarga de feriados y clima ---
    cliente_api = ClienteAPI()
    df_feriados = cliente_api.obtener_feriados_cr(ANIO_INICIO_FERIADOS, ANIO_FIN_FERIADOS)

    fecha_inicio = df_viajes["fecha"].min().strftime("%Y-%m-%d")
    fecha_fin = df_viajes["fecha"].max().strftime("%Y-%m-%d")

    clima_por_ciudad = {
        ciudad: cliente_api.obtener_clima_historico(fecha_inicio, fecha_fin, latitud=lat, longitud=lon)
        for ciudad, (lat, lon) in COORDENADAS_CIUDADES.items()
    }
    df_clima = Utilidades.construir_clima_multiciudad(clima_por_ciudad)

    # --- 3. Enriquecimiento ---
    df_enriquecido = Utilidades.enriquecer_viajes(df_viajes, df_feriados, df_clima)
    print(f"\nDataFrame enriquecido: {df_enriquecido.shape[0]:,} filas, {df_enriquecido.shape[1]} columnas.")

    # --- 4. Carga a base de datos ---
    gestor_bd = GestorBaseDatos(CONNECTION_STRING_DB)
    gestor_bd.cargar_dataframe(df_enriquecido, NOMBRE_TABLA_DB)
    if gestor_bd.es_postgres:
        gestor_bd.crear_hypertable(NOMBRE_TABLA_DB, columna_fecha="fecha")
    gestor_bd.cerrar()

    # --- 5-7. TODO: EDA, visualizacion y modelos ---
    # procesador_eda = ProcesadorEDA(df_enriquecido)
    # ...

    return df_enriquecido


if __name__ == "__main__":
    main()