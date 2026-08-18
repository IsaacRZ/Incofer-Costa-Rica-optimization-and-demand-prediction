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
    viz.grafico_demanda_por_hora(eda.demanda_por_hora())
    viz.grafico_top_rutas(eda.rutas_saturadas())
    viz.heatmap_correlacion(eda.correlacion_clima_pasajeros())
    viz.mapa_ciudades_clima()
"""

"""
visualizador.py

Modulo encargado de la generacion de graficos estaticos (Matplotlib/Seaborn)
y mapas interactivos (Folium) para el analisis de demanda e infraestructura de INCOFER.
"""

from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import folium


class Visualizador:
    """Clase encargada de construir las visualizaciones estaticas e interactivas del proyecto."""

    def __init__(self, estilo: str = "seaborn-v0_8-whitegrid"):
        """Inicializa los parametros esteticos de Matplotlib y Seaborn.

        Args:
            estilo (str): Tema de estilos para Matplotlib.
        """
        plt.style.use(estilo)
        # Paleta de colores institucional (Azul Ferroviario, Dorado, Rojo, Verde)
        self.paleta_incofer = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        self.color_principal = "#003366"  # Azul INCOFER
        self.color_secundario = "#D9534F"

        # Configuración global de fuentes y legibilidad
        plt.rcParams.update({
            "font.sans-serif": "DejaVu Sans",
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.3,
        })

    def grafico_demanda_semanal(self, df_semana: pd.DataFrame) -> plt.Figure:
        """Crea un grafico de barras para la demanda promedio por dia de la semana."""
        fig, ax = plt.subplots(figsize=(8, 4.5))

        sns.barplot(
            data=df_semana,
            x="nombre_dia",
            y="pasajeros_totales",
            hue="nombre_dia",  # Misma variable que 'x'
            legend=False,  # Desactiva la leyenda innecesaria
            palette="Blues_d",
            ax=ax,
        )

        ax.set_title("Demanda Promedio de Pasajeros por Día de la Semana", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Día de la Semana", fontsize=10, fontweight="bold")
        ax.set_ylabel("Promedio de Pasajeros", fontsize=10, fontweight="bold")

        # Agregar etiquetas de valor en las barras
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height) and height > 0:
                ax.annotate(
                    f"{int(height):,}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points",
                )

        plt.tight_layout()
        return fig

    def grafico_top_rutas(self, df_rutas: pd.DataFrame, top_n: int = 5) -> plt.Figure:
        """Crea un grafico de barras horizontales para las rutas con mayor saturacion."""
        fig, ax = plt.subplots(figsize=(8, 4.5))

        df_top = df_rutas.head(top_n)

        sns.barplot(
            data=df_top,
            y="recorrido_normalizado",
            x="pasajeros_totales",
            hue="recorrido_normalizado",  # 👈 Misma variable del eje categórico
            legend=False,  # 👈 Desactiva leyenda
            palette="Blues_r",
            ax=ax,
        )

        ax.set_title(f"Top {top_n} Rutas/Recorridos con Mayor Demanda", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Pasajeros Acumulados", fontsize=10, fontweight="bold")
        ax.set_ylabel("Recorrido", fontsize=10, fontweight="bold")

        plt.tight_layout()
        return fig

    def grafico_matriz_correlacion(self, df_corr: pd.DataFrame) -> plt.Figure:
        """Genera un mapa de calor (heatmap) para visualizar las correlaciones entre variables."""
        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            df_corr,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            square=True,
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title("Matriz de Correlación: Demanda vs. Clima", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        return fig

    def grafico_impacto_clima(self, df_enriquecido: pd.DataFrame) -> plt.Figure:
        """Grafico de dispersion para evaluar el impacto de la lluvia en la demanda de pasajeros."""
        fig, ax = plt.subplots(figsize=(8, 4.5))

        sns.scatterplot(
            data=df_enriquecido,
            x="precipitacion_mm",
            y="pasajeros_totales",
            hue="es_feriado",
            palette={True: self.color_secundario, False: self.color_principal},
            alpha=0.7,
            ax=ax,
        )

        ax.set_title("Relación entre Precipitación (mm) y Demanda de Pasajeros", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Precipitación (mm)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Pasajeros Totales", fontsize=10, fontweight="bold")
        ax.legend(title="¿Es Feriado?")

        plt.tight_layout()
        return fig

    def mapa_ciudades_clima(
        self,
        coordenadas: Optional[Dict[str, Tuple[float, float]]] = None,
        centro: Tuple[float, float] = (9.9325, -84.0795),
        zoom: int = 11,
    ) -> folium.Map:
        """Genera un mapa interactivo con Folium destacando las estaciones principales y rutas.

        Args:
            coordenadas: Diccionario con nombres de ciudades y sus pares (lat, lon).
            centro: Coordenadas del centro del mapa (default: San Jose).
            zoom: Nivel de zoom inicial.

        Returns:
            folium.Map: Objeto de mapa interactivo de Folium.
        """
        if coordenadas is None:
            coordenadas = {
                "Estación Atlántico (San José)": (9.9325, -84.0795),
                "Estación Heredia": (9.9989, -84.1165),
                "Estación Cartago": (9.8644, -83.9195),
                "Estación Alajuela": (10.0162, -84.2116),
            }

        # Mapa base con tema claro y elegante
        mapa = folium.Map(location=list(centro), zoom_start=zoom, tiles="CartoDB positron")

        # 1. Trazar linea de tren aproximada (Corredor Interurbano)
        puntos_corredor = [
            coordenadas["Estación Alajuela"],
            coordenadas["Estación Heredia"],
            coordenadas["Estación Atlántico (San José)"],
            coordenadas["Estación Cartago"],
        ]

        folium.PolyLine(
            locations=puntos_corredor,
            color="#003366",
            weight=4,
            opacity=0.8,
            tooltip="Línea de Ferrocarril Interurbano (INCOFER)",
        ).add_to(mapa)

        # 2. Marcadores personalizados para cada estacion/nodo
        for nombre, coord in coordenadas.items():
            folium.Marker(
                location=list(coord),
                popup=folium.Popup(f"<b>{nombre}</b><br>Nodo Operativo INCOFER", max_width=200),
                tooltip=nombre,
                icon=folium.Icon(color="blue", icon="train", prefix="fa"),
            ).add_to(mapa)

        return mapa;


if __name__ == "__main__":
    import os
    import numpy as np

    print("---  PRUEBA AISLADA DEL MÓDULO VISUALIZADOR ---")

    # 1. Crear carpeta de salida para los gráficos de prueba
    carpeta_prueba = "data/outputs"
    os.makedirs(carpeta_prueba, exist_ok=True)

    # 2. Generar datos sintéticos simulando el DataFrame enriquecido
    rng = np.random.default_rng(42)
    n_filas = 100
    fechas = pd.date_range("2024-01-01", periods=n_filas, freq="D")

    df_demo = pd.DataFrame({
        "fecha": fechas,
        "nombre_dia": fechas.day_name(),
        "recorrido_normalizado": rng.choice(["San Jose - Heredia", "San Jose - Cartago", "Heredia - Alajuela"], n_filas),
        "pasajeros_totales": rng.integers(300, 1500, n_filas),
        "temp_max_c": rng.uniform(18.0, 30.0, n_filas),
        "precipitacion_mm": rng.uniform(0.0, 35.0, n_filas),
        "es_feriado": rng.choice([True, False], n_filas, p=[0.1, 0.9]),
    })

    # 3. Instanciar el Visualizador
    viz = Visualizador()

    # 4. Probar Gráfico Demanda Semanal
    print(" 📊 Generando gráfico de demanda semanal...")
    df_semana = df_demo.groupby("nombre_dia", as_index=False)["pasajeros_totales"].mean()
    fig1 = viz.grafico_demanda_semanal(df_semana)
    fig1.savefig(f"{carpeta_prueba}/test_demanda_semanal.png")

    # 5. Probar Gráfico Top Rutas
    print("Generando gráfico de top rutas...")
    df_rutas = df_demo.groupby("recorrido_normalizado", as_index=False)["pasajeros_totales"].sum()
    fig2 = viz.grafico_top_rutas(df_rutas)
    fig2.savefig(f"{carpeta_prueba}/test_top_rutas.png")

    # 6. Probar Gráfico Matriz de Correlación
    print("Generando mapa de calor de correlaciones...")
    cols_num = df_demo.select_dtypes(include=[np.number])
    fig3 = viz.grafico_matriz_correlacion(cols_num.corr())
    fig3.savefig(f"{carpeta_prueba}/test_correlaciones.png")

    # 7. Probar Gráfico Impacto Clima
    print("Generando gráfico de impacto del clima...")
    fig4 = viz.grafico_impacto_clima(df_demo)
    fig4.savefig(f"{carpeta_prueba}/test_impacto_clima.png")

    # 8. Probar Mapa Interactivo de Folium
    print("Generando mapa interactivo en HTML...")
    mapa = viz.mapa_ciudades_clima()
    mapa.save(f"{carpeta_prueba}/test_mapa.html")

    print(f"\n¡Prueba completada con éxito! Archivos guardados en '{carpeta_prueba}/':")
    print("   • test_demanda_semanal.png")
    print("   • test_top_rutas.png")
    print("   • test_correlaciones.png")
    print("   • test_impacto_clima.png")
    print("   • test_mapa.html")