"""
src/datos/gestor_datos.py

Clase para cargar y limpiar el CSV crudo de ARESEP (Tren urbano de
pasajeros), produciendo un dataset tidy a nivel de dia x recorrido x
sentido, listo para EDA, feature engineering y carga a base de datos.

Uso como script:
    python src/datos/gestor_datos.py --input data/raw/aresep_tren.csv \
                                  --output data/processed/viajes_diarios.parquet

Uso como modulo:
    from src.datos.gestor_datos import LimpiadorARESEP

    limpiador = LimpiadorARESEP("data/raw/aresep_tren.csv")
    df_limpio = limpiador.ejecutar()
    limpiador.guardar(df_limpio, "data/processed/viajes_diarios.parquet")
"""

import argparse
import unicodedata
from pathlib import Path

import pandas as pd


class GestorDatos:
    """Encapsula todo el flujo de carga y limpieza del dataset de ARESEP.

    Cada paso del pipeline (cargar, normalizar texto, tratar nulos,
    deduplicar, derivar columnas) es un metodo propio para poder probarlo
    o reutilizarlo de forma aislada; `ejecutar()` los encadena en orden.
    """

    COLUMNAS_ESPERADAS = {
        "Código de Ruta": "codigo_ruta",
        "Descripción de la ruta": "ruta",
        "Código recorrido": "codigo_recorrido",
        "Descripción del recorrido": "recorrido",
        "Sentido": "sentido",
        "Dia": "dia",
        "Mes": "mes",
        "Año": "anio",
        "Cantidad de Pasajeros Regulares": "pasajeros_regulares",
        "Cantidad de Adultos Mayores": "adultos_mayores",
        "Ingresos": "ingresos_colones",
    }

    COLUMNAS_FINALES = [
        "fecha", "anio", "mes", "dia", "dia_semana", "nombre_dia", "es_fin_de_semana",
        "codigo_ruta", "ruta", "codigo_recorrido", "recorrido", "recorrido_normalizado",
        "sentido",
        "pasajeros_regulares", "adultos_mayores", "pasajeros_totales",
        "ingresos_colones",
        "pasajeros_regulares_faltante", "ingresos_faltante",
    ]

    def __init__(self, path_csv: str):
        self.path_csv = Path(path_csv)
        self.df_crudo: pd.DataFrame | None = None
        self.df_limpio: pd.DataFrame | None = None
        self._avisos: list[str] = []

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------
    def ejecutar(self) -> pd.DataFrame:
        """Corre el pipeline completo y devuelve el dataframe limpio."""
        self.df_crudo = self._cargar()
        df = self.df_crudo.copy()   # Copia de df crudo
        df = self._normalizar_texto(df)
        df = self._construir_fecha(df)
        df = self._tratar_nulos(df)
        df = self._quitar_duplicados(df)
        df = self._derivar_columnas_calendario(df)
        df = self._derivar_pasajeros_totales(df)
        self.df_limpio = df[self.COLUMNAS_FINALES].sort_values(
            ["fecha", "codigo_recorrido", "sentido"]
        ).reset_index(drop=True)
        return self.df_limpio

    # ------------------------------------------------------------------
    # Pasos individuales
    # ------------------------------------------------------------------
    @staticmethod
    def _reparar_doble_codificacion(texto: str) -> str:
        """Repara texto que fue codificado dos veces por error: UTF-8 leido
        como Latin-1/cp1252 y vuelto a guardar como UTF-8 (ej. "Código" ->
        "CÃ³digo"). Si el texto no tiene ese patron, .encode("latin-1")
        falla (caracteres fuera de rango) o el resultado no es UTF-8 valido;
        en ambos casos se devuelve el texto original sin tocar.
        """
        if not isinstance(texto, str):
            return texto
        try:
            return texto.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return texto

    def _cargar(self) -> pd.DataFrame:
        df = pd.read_csv(self.path_csv, encoding="utf-8-sig")
        # 1) Repara doble codificacion en los NOMBRES de columna...
        df.columns = [
            self._reparar_doble_codificacion(c) for c in df.columns
        ]
        # 2) ...y colapsa espacios internos/extremos, igual que antes.
        df.columns = [" ".join(c.split()) for c in df.columns]

        faltantes = set(self.COLUMNAS_ESPERADAS) - set(df.columns)
        if faltantes:
            raise ValueError(
                f"El CSV no tiene las columnas esperadas: {faltantes}. "
                f"Columnas encontradas: {list(df.columns)}"
            )
        df = df.rename(columns=self.COLUMNAS_ESPERADAS)

        # 3) Repara la misma doble codificacion en los VALORES de texto
        # (ej. "ESTACION DEL PACÃ­FICO" -> "ESTACION DEL PACÍFICO"), no solo
        # en los encabezados.
        for col in ["ruta", "recorrido", "sentido"]:
            df[col] = df[col].apply(self._reparar_doble_codificacion)

        return df

    @staticmethod
    def _quitar_tildes(texto: str) -> str:
        if not isinstance(texto, str):
            return texto
        texto = " ".join(texto.strip().split())
        nfkd = unicodedata.normalize("NFKD", texto)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def _normalizar_texto(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["ruta", "recorrido", "sentido", "codigo_ruta", "codigo_recorrido"]:
            df[col] = df[col].astype(str).str.strip()
        # El CSV fuente mezcla tildes/guiones entre exportaciones (ej. en-dash
        # vs guion normal en "ESTACION DEL ATLANTICO – HEREDIA"). Se crea una
        # columna normalizada para agrupar sin depender de heuristicas fragiles.
        df["recorrido_normalizado"] = df["recorrido"].apply(self._quitar_tildes)
        return df

    def _construir_fecha(self, df: pd.DataFrame) -> pd.DataFrame:
        df["fecha"] = pd.to_datetime(
            dict(year=df["anio"], month=df["mes"], day=df["dia"]), errors="coerce"
        )
        n_invalidas = df["fecha"].isna().sum()
        if n_invalidas:
            self._avisos.append(f"{n_invalidas} filas con fecha invalida, descartadas.")
            df = df.dropna(subset=["fecha"])
        return df

    def _tratar_nulos(self, df: pd.DataFrame) -> pd.DataFrame:
        # pasajeros_regulares / ingresos: no hay evidencia de que nulo == 0
        # (podria ser dia sin dato reportado vs. dia sin servicio). Se deja
        # como NaN, marcado con flag, para que EDA/modelo decidan el trato.
        df["pasajeros_regulares_faltante"] = df["pasajeros_regulares"].isna()
        df["ingresos_faltante"] = df["ingresos_colones"].isna()

        # adultos_mayores nulo es mayoritario (~78%) y consistente con "no
        # hubo adultos mayores" (pasajeros_regulares e ingresos SI existen
        # en esas mismas filas), por lo que aqui si es razonable imputar a 0.
        df["adultos_mayores"] = df["adultos_mayores"].fillna(0)
        return df

    def _quitar_duplicados(self, df: pd.DataFrame) -> pd.DataFrame:
        n_dup = df.duplicated(subset=["codigo_recorrido", "sentido", "fecha"]).sum()
        if n_dup:
            self._avisos.append(
                f"{n_dup} filas duplicadas (recorrido+sentido+fecha), descartadas."
            )
            df = df.drop_duplicates(subset=["codigo_recorrido", "sentido", "fecha"])
        return df

    @staticmethod
    def _derivar_columnas_calendario(df: pd.DataFrame) -> pd.DataFrame:
        df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=lunes ... 6=domingo
        df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6])
        df["nombre_dia"] = df["fecha"].dt.day_name(locale=None)
        return df

    @staticmethod
    def _derivar_pasajeros_totales(df: pd.DataFrame) -> pd.DataFrame:
        df["pasajeros_totales"] = df["pasajeros_regulares"].fillna(0) + df["adultos_mayores"]
        return df

    # ------------------------------------------------------------------
    # Utilidades de salida
    # ------------------------------------------------------------------
    def guardar(self, df: pd.DataFrame, path_salida: str) -> None:
        out_path = Path(path_salida)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        print(f"Guardado: {out_path}")

    def resumen_calidad(self) -> None:
        if self.df_limpio is None:
            raise RuntimeError("Ejecuta ejecutar() antes de pedir el resumen.")
        df = self.df_limpio

        print("\n--- Resumen de calidad del dataset limpio ---")
        for aviso in self._avisos:
            print(f"[aviso] {aviso}")
        print(f"Filas: {len(df):,}")
        print(f"Rango de fechas: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
        print(f"Recorridos distintos: {df['recorrido_normalizado'].nunique()}")
        print(f"% filas con pasajeros_regulares faltante: {df['pasajeros_regulares_faltante'].mean():.1%}")
        print(f"% filas con ingresos faltante: {df['ingresos_faltante'].mean():.1%}")
        rango_completo = pd.date_range(df["fecha"].min(), df["fecha"].max())
        faltantes = rango_completo.difference(df["fecha"].unique())
        print(f"Dias calendario sin ningun registro en el rango: {len(faltantes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Ruta al CSV crudo de ARESEP")
    parser.add_argument("--output", required=True, help="Ruta de salida (.parquet o .csv)")
    args = parser.parse_args()

    gestor = GestorDatos(args.input)
    df_limpio = gestor.ejecutar()
    gestor.resumen_calidad()
    gestor.guardar(df_limpio, args.output)


if __name__ == "__main__":
    main()