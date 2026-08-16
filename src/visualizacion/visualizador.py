"""
src/visualizacion/visualizador.py

Clase Visualizador: crea graficos (lineas, barras, heatmaps) y mapas a
partir de DataFrames ya calculados. No conoce la base de datos, las APIs
ni hace ningun calculo estadistico propio -- recibe el resultado de
ProcesadorEDA (o cualquier DataFrame con la forma esperada) y solo grafica.

Uso:
    from src.visualizacion.visualizador import Visualizador
    from src.eda.procesador_eda import ProcesadorEDA

    eda = ProcesadorEDA(df_enriquecido)
    viz = Visualizador()

    viz.grafico_serie_temporal(df_enriquecido)
    viz.grafico_demanda_semanal(eda.demanda_por_dia_semana())
    viz.grafico_top_rutas(eda.rutas_saturadas())
    viz.heatmap_correlacion(eda.correlacion_clima_pasajeros())
    viz.mapa_ciudades_clima()
"""

import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import pandas as pd
import folium


class Visualizador:
    """Encapsula la creacion de graficos y mapas. Cada metodo recibe
    exactamente el DataFrame que necesita (ya agregado por ProcesadorEDA
    cuando aplica) y devuelve la figura/mapa, sin guardar estado propio
    entre llamadas."""

    def __init__(self, estilo: str = "darkgrid", figsize: tuple[int, int] = (10, 5)):
        sns.set_style(estilo)
        self.figsize = figsize

    def grafico_serie_temporal(
        self,
        df: pd.DataFrame,
        columna_fecha: str = "fecha",
        columna_valor: str = "pasajeros_totales",
        agregacion: str = "sum",
        titulo: str = "Pasajeros totales por dia",
    ) -> matplotlib.figure.Figure:
        """Linea de tiempo: agrega `columna_valor` por `columna_fecha`
        (sumando todos los recorridos de cada dia) y la grafica."""
        serie = df.groupby(columna_fecha)[columna_valor].agg(agregacion)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(serie.index, serie.values, linewidth=0.8)
        ax.set_title(titulo)
        ax.set_xlabel("Fecha")
        ax.set_ylabel(columna_valor)
        fig.tight_layout()
        return fig

    def grafico_demanda_semanal(
        self,
        df_demanda: pd.DataFrame,
        titulo: str = "Demanda promedio por dia de la semana",
    ) -> matplotlib.figure.Figure:
        """Barras de la salida de ProcesadorEDA.demanda_por_dia_semana()
        (columnas esperadas: nombre_dia, promedio)."""
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.barplot(data=df_demanda, x="nombre_dia", y="promedio", ax=ax)
        ax.set_title(titulo)
        ax.set_xlabel("Dia de la semana")
        ax.set_ylabel("Pasajeros promedio")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def grafico_top_rutas(
        self,
        df_rutas: pd.DataFrame,
        titulo: str = "Rutas con mayor demanda promedio",
    ) -> matplotlib.figure.Figure:
        """Barras horizontales de la salida de
        ProcesadorEDA.rutas_saturadas() (columnas esperadas:
        recorrido_normalizado, promedio_pasajeros_diarios)."""
        fig, ax = plt.subplots(figsize=self.figsize)
        datos = df_rutas.sort_values("promedio_pasajeros_diarios")
        ax.barh(datos["recorrido_normalizado"], datos["promedio_pasajeros_diarios"])
        ax.set_title(titulo)
        ax.set_xlabel("Pasajeros promedio diarios")
        fig.tight_layout()
        return fig

    def heatmap_correlacion(
        self,
        df_corr: pd.DataFrame,
        titulo: str = "Correlacion clima vs. pasajeros",
    ) -> matplotlib.figure.Figure:
        """Heatmap de la salida de
        ProcesadorEDA.correlacion_clima_pasajeros()."""
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(titulo)
        fig.tight_layout()
        return fig

    def mapa_ciudades_clima(
        self,
        coordenadas: dict[str, tuple[float, float]] | None = None,
    ) -> folium.Map:
        """Mapa interactivo (punto extra) con las ciudades usadas como
        referencia climatica en Utilidades.MAPA_RECORRIDO_CIUDAD. Sin
        argumentos, usa las coordenadas por defecto del GAM."""
        coordenadas = coordenadas or {
            "San Jose": (9.9325, -84.0795),
            "Heredia": (9.9989, -84.1165),
            "Cartago": (9.8644, -83.9195),
        }
        lat_centro = sum(lat for lat, _ in coordenadas.values()) / len(coordenadas)
        lon_centro = sum(lon for _, lon in coordenadas.values()) / len(coordenadas)

        mapa = folium.Map(location=[lat_centro, lon_centro], zoom_start=11)
        for ciudad, (lat, lon) in coordenadas.items():
            folium.Marker(
                location=[lat, lon],
                popup=ciudad,
                tooltip=ciudad,
                icon=folium.Icon(color="blue", icon="train", prefix="fa"),
            ).add_to(mapa)
        return mapa

    def guardar(self, figura, ruta_salida: str) -> None:
        """Guarda una figura de matplotlib (.png/.pdf/etc) o un mapa de
        folium (.html) segun el tipo de objeto recibido."""
        if isinstance(figura, folium.Map):
            figura.save(ruta_salida)
        else:
            figura.savefig(ruta_salida, dpi=150, bbox_inches="tight")
        print(f"Guardado: {ruta_salida}")


def main():
    """Prueba rapida manual con datos sinteticos, sin requerir BD ni APIs."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 500
    fechas = pd.date_range("2024-01-01", periods=n // 5).repeat(5)[:n]
    df_prueba = pd.DataFrame({
        "fecha": fechas,
        "pasajeros_totales": rng.integers(50, 500, n),
    })
    df_demanda = pd.DataFrame({
        "nombre_dia": ["Monday", "Tuesday", "Wednesday"],
        "promedio": [300, 280, 310],
    })
    df_rutas = pd.DataFrame({
        "recorrido_normalizado": ["Ruta A", "Ruta B", "Ruta C"],
        "promedio_pasajeros_diarios": [500, 300, 200],
    })
    df_corr = pd.DataFrame(
        [[1.0, -0.2, 0.1], [-0.2, 1.0, 0.05], [0.1, 0.05, 1.0]],
        columns=["pasajeros_totales", "temp_max_c", "precipitacion_mm"],
        index=["pasajeros_totales", "temp_max_c", "precipitacion_mm"],
    )

    viz = Visualizador()
    viz.grafico_serie_temporal(df_prueba)
    viz.grafico_demanda_semanal(df_demanda)
    viz.grafico_top_rutas(df_rutas)
    viz.heatmap_correlacion(df_corr)
    mapa = viz.mapa_ciudades_clima()
    print("Los 4 graficos de matplotlib se generaron sin error.")
    print("Mapa folium generado:", type(mapa))


if __name__ == "__main__":
    main()