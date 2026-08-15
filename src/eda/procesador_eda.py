"""
src/eda/procesador_eda.py
 
Clase ProcesadorEDA: analisis exploratorio del dataset enriquecido
(viajes + feriados + clima). No conoce CSVs, APIs ni bases de datos --
recibe un DataFrame ya listo (de Utilidades.enriquecer_viajes(), o de
GestorBaseDatos.consultar()) y solo calcula estadisticas sobre el.
 
Uso:
    from src.eda.procesador_eda import ProcesadorEDA
 
    eda = ProcesadorEDA(df_enriquecido)
    eda.resumen_general()
    demanda = eda.demanda_por_dia_semana()
    saturadas = eda.rutas_saturadas()
    impacto = eda.impacto_feriados()
    corr = eda.correlacion_clima_pasajeros()
"""

import pandas as pd

class ProcesadorEDA:
    """Encapsula el analisis exploratorio sobre un DataFrame ya enriquecido.
 
    Nota de alcance: el dataset es a nivel DIARIO (no horario), asi que no
    hay "hora pico/valle" -- el equivalente disponible es dia de la semana
    y estacionalidad mensual.
    """
 
    COLUMNAS_REQUERIDAS = [
        "fecha", "dia_semana", "nombre_dia", "es_fin_de_semana",
        "recorrido_normalizado", "pasajeros_totales",
        "es_feriado", "temp_max_c", "precipitacion_mm",
    ]

    def __init__(self, df: pd.DataFrame):
        faltantes = set(self.COLUMNAS_REQUERIDAS) - set(df.columns)
        if faltantes:
            raise ValueError(
                f"ProcesadorEDA espera un DataFrame enriquecido. "
                f"Faltan columnas: {faltantes}"
            )
        self.df = df

    def resumen_general(self) -> dict:
        """Panorama rapido del dataset: tamano, rango de fechas, nulos
        clave. Pensado para imprimir al inicio de un notebook de EDA."""
        resumen = {
            "filas": len(self.df),
            "rango_fechas": (self.df["fecha"].min(), self.df["fecha"].max()),
            "recorridos_distintos": self.df["recorrido_normalizado"].nunique(),
            "pct_pasajeros_nulos": self.df["pasajeros_totales"].isna().mean(),
            "pct_clima_nulo": self.df["temp_max_c"].isna().mean(),
            "dias_feriado": int(self.df.loc[self.df["es_feriado"], "fecha"].nunique()),
        }
        for clave, valor in resumen.items():
            print(f"{clave}: {valor}")
        return resumen

    def estadisticas_descriptivas(self, columnas: list[str] | None = None) -> pd.DataFrame:
        """Wrapper delgado sobre df.describe(), pero limitando a columnas
        numericas relevantes por defecto (evita que describe() se llene de
        columnas booleanas/de codigo que no aportan estadisticos utiles)."""
        columnas = columnas or ["pasajeros_regulares", "adultos_mayores", "pasajeros_totales",
                                 "ingresos_colones", "temp_max_c", "temp_min_c", "precipitacion_mm"]
        return self.df[columnas].describe()
 
    def demanda_por_dia_semana(self) -> pd.DataFrame:
        """Promedio y total de pasajeros por dia de la semana, agregando
        todos los recorridos. Util para identificar patrones de demanda
        semanal (equivalente diario de 'hora pico/valle')."""
        return(
            self.df.groupby(["dia_semana", "nombre_dia"], as_index=False)["pasajeros_totales"]
            .agg(promedio="mean", total="sum", dias_observados="count")
            .sort_values("dia_semana")
            .reset_index(drop=True)
        )

    def rutas_saturadas(self, percentil: float = 0.90) -> pd.DataFrame:
        """Identifica los recorridos cuya demanda promedio diaria supera el
        percentil dado, calculado sobre el promedio por recorrido (no sobre
        filas individuales). Es una aproximacion de "saturacion" basada en
        volumen relativo, no en capacidad real del equipo (esa info no esta
        en este dataset -- ver notas del proyecto sobre capacidad por tipo
        de tren, sacada de los informes PDF de INCOFER)."""
        promedio_por_ruta = (
            self.df.groupby("recorrido_normalizado")["pasajeros_totales"]
            .mean()
            .sort_values(ascending=False)
        )
        umbral = promedio_por_ruta.quantile(percentil)
        saturadas = promedio_por_ruta[promedio_por_ruta >= umbral]
        return saturadas.reset_index(name="promedio_pasajeros_diarios")
 
    def impacto_feriados(self) -> pd.DataFrame:
        """Compara el promedio de pasajeros en dias feriados vs. dias
        regulares, para cuantificar el efecto que pide el proyecto
        ('dia tipo: laboral/feriado')."""
        return (
            self.df.groupby("es_feriado", as_index=False)["pasajeros_totales"]
            .agg(promedio="mean", total="sum", dias="count")
        )
 
    def correlacion_clima_pasajeros(self) -> pd.DataFrame:
        """Matriz de correlacion entre pasajeros_totales y las variables de
        clima, para responder directamente 'impacto de lluvia en cantidad
        de pasajeros' que pide el proyecto."""
        columnas = ["pasajeros_totales", "temp_max_c", "temp_min_c", "precipitacion_mm"]
        return self.df[columnas].corr()
 
    def detectar_outliers(self, columna: str = "pasajeros_totales", factor_iqr: float = 1.5) -> pd.DataFrame:
        """Devuelve las filas cuyo valor en `columna` cae fuera del rango
        [Q1 - factor_iqr*IQR, Q3 + factor_iqr*IQR] (metodo IQR estandar).
        Utiles como candidatos a revisar (ej. dias con eventos especiales,
        suspensiones de servicio, o errores de captura)."""
        q1 = self.df[columna].quantile(0.25)
        q3 = self.df[columna].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - factor_iqr * iqr
        limite_superior = q3 + factor_iqr * iqr
        mascara = (self.df[columna] < limite_inferior) | (self.df[columna] > limite_superior)
        return self.df.loc[mascara].sort_values(columna, ascending=False)
 
 
def main():
    """Prueba rapida manual con datos sinteticos (no requiere BD ni APIs),
    solo para confirmar que los calculos corren sin errores."""
    import numpy as np
 
    rng = np.random.default_rng(42)
    n = 500
    fechas = pd.date_range("2024-01-01", periods=n // 5).repeat(5)[:n]
    df_prueba = pd.DataFrame({
        "fecha": fechas,
        "dia_semana": fechas.dayofweek,
        "nombre_dia": fechas.day_name(),
        "es_fin_de_semana": fechas.dayofweek.isin([5, 6]),
        "recorrido_normalizado": rng.choice(["Ruta A", "Ruta B", "Ruta C"], n),
        "pasajeros_totales": rng.integers(50, 500, n),
        "es_feriado": rng.random(n) < 0.05,
        "temp_max_c": rng.normal(25, 3, n),
        "temp_min_c": rng.normal(15, 2, n),
        "precipitacion_mm": rng.exponential(3, n),
    })
 
    eda = ProcesadorEDA(df_prueba)
    print("=== Resumen general ===")
    eda.resumen_general()
    print("\n=== Demanda por dia de semana ===")
    print(eda.demanda_por_dia_semana())
    print("\n=== Rutas saturadas (percentil 0.90) ===")
    print(eda.rutas_saturadas())
    print("\n=== Impacto feriados ===")
    print(eda.impacto_feriados())
    print("\n=== Correlacion clima-pasajeros ===")
    print(eda.correlacion_clima_pasajeros())
    print("\n=== Outliers detectados ===", len(eda.detectar_outliers()), "filas")
 
 
if __name__ == "__main__":
    main()


