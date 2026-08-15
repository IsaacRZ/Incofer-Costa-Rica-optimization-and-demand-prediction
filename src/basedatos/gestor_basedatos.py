"""
src/basedatos/gestor_basedatos.py
 
Clase GestorBaseDatos: conecta con SQLite, PostgreSQL (con o sin
TimescaleDB), MySQL o SQL Server, y permite cargar/consultar datos, usando
SQLAlchemy como capa de abstraccion (un mismo codigo Python funciona con
cualquiera de los motores, solo cambia la cadena de conexion).
 
Uso:
    from src.basedatos.gestor_basedatos import GestorBaseDatos
 
    # SQLite (archivo local, sin servidor)
    gestor = GestorBaseDatos("sqlite:///data/incofer.db")
 
    # PostgreSQL/TimescaleDB (requiere docker compose up -d)
    gestor = GestorBaseDatos(
        "postgresql+psycopg2://incofer:incofer_dev_password@localhost:5432/incofer"
    )
 
    gestor.cargar_dataframe(df_viajes, "viajes_diarios", columna_fecha="fecha")
    gestor.crear_hypertable("viajes_diarios", columna_fecha="fecha")  # solo Postgres+Timescale
    df = gestor.consultar("SELECT * FROM viajes_diarios LIMIT 10")
"""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

class GestorBaseDatos:
    """Encapsula la conexion y las operaciones comunes contra la base de
    datos del proyecto. No asume un motor especifico: el `connection_string`
    que recibe en el constructor decide si se conecta a SQLite, PostgreSQL,
    MySQL o SQL Server (SQLAlchemy interpreta el prefijo, ej. "sqlite:///",
    "postgresql+psycopg2://", "mysql+pymysql://", "mssql+pyodbc://").
    """

    def __init__(self, connection_string: str, echo: bool = False):
        self.connection_string = connection_string
        # echo=True imprime cada SQL que ejecuta SQLAlchemy
        # util para depurar
        self.engine: Engine = create_engine(connection_string, echo=echo)

    @property
    def es_postgres(self) -> bool:
        """Para llamar métodos especificos de Postgres/Timescale.
        Fallan usando SQLite/MySQL/SQL Server."""
        return self.engine.dialect.name == "postgresql"

    def cargar_dataframe(
            self,
            df: pd.DataFrame,
            nombre_tabla: str,
            columna_fecha: str | None = None,
            if_exists: str = "replace",
        ) -> None:
        """Escribe un DataFrame completo como tabla. `if_exists='replace'`
        (el default de pandas.to_sql) recrea la tabla si ya existia -- util
        mientras el pipeline de limpieza sigue cambiando; cambia a 'append'
        cuando quieras ir acumulando datos nuevos sin borrar los anteriores.
        """

        df.to_sql(nombre_tabla, self.engine, if_exists=if_exists, index=False)
        print(f"Cargadas {len(df):,} filas en la tabla '{nombre_tabla}'.")

    def crear_hypertable(self, nombre_tabla: str, columna_fecha: str) -> None:

        if not self.es_postgres:
            print(
                f"[aviso] crear_hypertable() ignorado: el motor actual es "
                f"'{self.engine.dialect.name}'. Solamente para el motor de base de datos: PostreSQL 16"
            )
            return

        with self.engine.begin() as conexion:
            conexion.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
            conexion.execute(text(
                f"SELECT create_hypertable("
                f"'{nombre_tabla}', '{columna_fecha}', if_not_exists => TRUE);"
            ))
        print(f"'{nombre_tabla}' convertida en hypertable (particionada por '{columna_fecha}').")

    def consultar (self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """Ejecuta un SELECT y devuelve el resultado como DataFrame.
        `params` permite pasar valores de forma segura (evita SQL injection)
        usando placeholders con nombre, ej: ':anio' en el SQL y
        params={'anio': 2024}."""
        return pd.read_sql(text(sql), self.engine, params=params)

    def cerrar(self) -> None:
        """Libera las conexiones del pool. Buena práctica
            al terminar un script o notebook."""
        self.engine.dispose()

def main():
    """Prueba rapida manual contra SQLite (no requiere Docker), para
    confirmar que la clase funciona de punta a punta antes de probar
    contra el Postgres/Timescale real."""
    import pandas as pd
 
    gestor = GestorBaseDatos("sqlite:///data/raw/prueba_gestor_basedatos.db")
 
    df_prueba = pd.DataFrame({
        "fecha": pd.date_range("2024-01-01", periods=5, freq="D"),
        "recorrido": ["A", "B", "A", "B", "A"],
        "pasajeros": [100, 150, 120, 180, 90],
    })
    gestor.cargar_dataframe(df_prueba, "viajes_prueba")
 
    resultado = gestor.consultar(
        "SELECT recorrido, SUM(pasajeros) AS total FROM viajes_prueba GROUP BY recorrido"
    )
    print(resultado)
 
    gestor.cerrar()
 
 
if __name__ == "__main__":
    main()