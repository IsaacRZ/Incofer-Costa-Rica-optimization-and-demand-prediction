"""
main.py

Punto de entrada del proyecto. Orquesta el pipeline completo:

    1. GestorDatos       -> limpia el CSV crudo de ARESEP
    2. ClienteAPI        -> descarga feriados CR y clima (3 ciudades)
    3. Utilidades        -> une todo en un DataFrame enriquecido y guarda Parquet
    4. GestorBaseDatos   -> carga el resultado a la base de datos
    5. ProcesadorEDA     -> análisis exploratorio de datos
    6. Visualizador       -> generación de gráficos y mapas
    7. ModeloML           -> entrenamiento y evaluación de modelos de ML

Cada clase se instancia UNA vez aquí, con sus dependencias inyectadas
donde corresponde -- ninguna clase crea instancias de las otras por su
cuenta, para poder probarlas por separado (como hemos hecho hasta ahora
con mocks) sin tener que correr el pipeline completo cada vez.

Uso:
    uv run python main.py
"""

import os
from src.datos.gestor_datos import GestorDatos
from src.api.cliente_api import ClienteAPI
from src.helpers.utilidades import Utilidades
from src.basedatos.gestor_basedatos import GestorBaseDatos
from src.eda.procesador_eda import ProcesadorEDA
from src.visualizacion.visualizador import Visualizador
from src.modelos.modelo_ml import modelo_ml

# --- Configuración del pipeline ---
RUTA_CSV_CRUDO = "data/raw/aresep_tren.csv"
RUTA_PARQUET_LIMPIO = "data/processed/viajes_diarios.parquet"
CARPETA_OUTPUTS = "data/outputs"

ANIO_INICIO_FERIADOS = 2021
ANIO_FIN_FERIADOS = 2026

# Coordenadas de referencia por ciudad, usadas para el clima.
COORDENADAS_CIUDADES = {
    "San Jose": (9.9325, -84.0795),
    "Heredia": (9.9989, -84.1165),
    "Cartago": (9.8644, -83.9195),
}

CONNECTION_STRING_DB = "postgresql+psycopg2://incofer:incofer_dev_password@localhost:5432/incofer"
NOMBRE_TABLA_DB = "viajes_diarios"


def main():
    os.makedirs(CARPETA_OUTPUTS, exist_ok=True)
    os.makedirs(os.path.dirname(RUTA_PARQUET_LIMPIO), exist_ok=True)

    # --- 1. Limpieza del CSV crudo ---
    gestor_datos = GestorDatos(RUTA_CSV_CRUDO)
    df_viajes = gestor_datos.ejecutar()
    gestor_datos.resumen_calidad()

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

    # --- 3. Enriquecimiento y Guardado del Parquet Completo ---
    df_enriquecido = Utilidades.enriquecer_viajes(df_viajes, df_feriados, df_clima)
    print(f"\nDataFrame enriquecido: {df_enriquecido.shape[0]:,} filas, {df_enriquecido.shape[1]} columnas.")

    # Guardamos el DataFrame ENRIQUECIDO para que Streamlit y EDA lean el archivo correcto
    df_enriquecido.to_parquet(RUTA_PARQUET_LIMPIO, index=False)
    print(f"✅ Guardado Parquet enriquecido en: {RUTA_PARQUET_LIMPIO}")

    # --- 4. Carga a base de datos ---
    try:
        gestor_bd = GestorBaseDatos(CONNECTION_STRING_DB)
        gestor_bd.cargar_dataframe(df_enriquecido, NOMBRE_TABLA_DB)
        if gestor_bd.es_postgres:
            gestor_bd.crear_hypertable(NOMBRE_TABLA_DB, columna_fecha="fecha")
        gestor_bd.cerrar()
    except Exception as e:
        print(f"⚠️ Nota DB: No se pudo conectar a PostgreSQL ({e}). Continuando pipeline...")

    # --- 5. Análisis Exploratorio (EDA) ---
    print("\n--- Ejecutando Análisis Exploratorio (EDA) ---")
    # --- 5. Análisis Exploratorio (EDA) ---
    print("\n--- Ejecutando Análisis Exploratorio (EDA) ---")
    procesador_eda = ProcesadorEDA(df_enriquecido)

    df_semana = procesador_eda.demanda_por_dia_semana()
    df_rutas = procesador_eda.rutas_saturadas()

    print("Demanda Promedio por Día de la Semana:")
    print(df_semana)
    print("\nTop Rutas más Concurridas:")
    print(df_rutas.head())


    # --- 6. Visualización ---
    print("\n--- Generando Visualizaciones ---")
    visualizador = Visualizador()
    df_semana = procesador_eda.demanda_por_dia_semana()
    fig_semana = visualizador.grafico_demanda_semanal(df_semana)
    fig_semana.savefig(f"{CARPETA_OUTPUTS}/demanda_semanal.png")
    print(f"✅ Gráfico guardado en: {CARPETA_OUTPUTS}/demanda_semanal.png")

    # --- 7. Machine Learning ---
    print("\n--- Ejecutando Pipeline de Machine Learning ---")
    ml = modelo_ml(df_enriquecido)
    df_target = ml.crear_variable_ocupacion()

    X_b, y_b = ml.preparar_features_pronostico(df_target)
    res_cls = ml.comparar_modelos_clasificacion(X_b, y_b)

    print("Resultados Clasificación:")
    for mod, metricas in res_cls.items():
        print(f"  • {mod:18s} -> Accuracy CV: {metricas['Accuracy_CV']:.3f}")

    # Optimizamos y guardamos el mejor modelo para usarlo en Streamlit
    ml.optimizar_mejor_modelo(X_b, y_b, nombre_modelo="RandomForest")
    ml.guardar_modelo(f"{CARPETA_OUTPUTS}/modelo_clasificacion.joblib")

    print("\n🎉 Pipeline ejecutado exitosamente de principio a fin.")
    return df_enriquecido


if __name__ == "__main__":
    main()