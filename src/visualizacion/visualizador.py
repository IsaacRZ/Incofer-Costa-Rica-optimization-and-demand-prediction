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
        ventana_promedio_movil: int = 30,
        titulo: str = "Pasajeros totales por dia",
    ) -> matplotlib.figure.Figure:
        """Linea de tiempo: agrega `columna_valor` por `columna_fecha`
        (sumando todos los recorridos de cada dia) y la grafica.
        """
        serie = df.groupby(columna_fecha)[columna_valor].agg(agregacion)
        promedio_movil = serie.rolling(window=ventana_promedio_movil, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(
            serie.index,
            serie.values,
            linewidth=0.5,
            alpha=0.35,
            color="steelblue",
            label=f"{agregacion} diario",
        )
        ax.plot(
            promedio_movil.index,
            promedio_movil.values,
            linewidth=2.2,
            color="firebrick",
            label=f"promedio movil {ventana_promedio_movil}d",
        )
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel("Fecha")
        ax.set_ylabel(columna_valor)
        ax.legend()
        fig.tight_layout()
        plt.close(fig)
        return fig

    def grafico_demanda_por_hora(
        self,
        df_demanda_hora: pd.DataFrame,
        columna_hora: str = "hora",
        columna_pasajeros: str = "promedio_pasajeros",
        titulo: str = "Flujo de pasajeros por hora del dia (Picos y Valles)",
    ) -> matplotlib.figure.Figure:
        """Grafico de linea/area enfocado en identificar Horas Pico y Horas Valle.
        Columnas esperadas: hora (0-23) y promedio_pasajeros.
        Requerimiento explicito de la rubrica de INCOFER.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Linea principal de flujo
        ax.plot(
            df_demanda_hora[columna_hora],
            df_demanda_hora[columna_pasajeros],
            color="#2b5c8f",
            linewidth=2.5,
            marker="o",
            markersize=5,
        )
        # Sombreado bajo la curva para resaltar volumenes
        ax.fill_between(
            df_demanda_hora[columna_hora],
            df_demanda_hora[columna_pasajeros],
            color="#2b5c8f",
            alpha=0.15,
        )

        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel("Hora del dia (0-23h)")
        ax.set_ylabel("Pasajeros promedio")
        ax.set_xticks(range(0, 24))
        ax.grid(True, linestyle="--", alpha=0.6)
        sns.despine()
        fig.tight_layout()
        plt.close(fig)
        return fig

    def grafico_demanda_semanal(
        self,
        df_demanda: pd.DataFrame,
        titulo: str = "Demanda promedio por dia de la semana",
    ) -> matplotlib.figure.Figure:
        """Barras de la salida de ProcesadorEDA.demanda_por_dia_semana()
        (columnas esperadas: nombre_dia, promedio)."""
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.barplot(
            data=df_demanda,
            x="nombre_dia",
            y="promedio",
            palette="Blues_d",
            hue="nombre_dia",
            legend=False,
            ax=ax,
        )
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel("Dia de la semana")
        ax.set_ylabel("Pasajeros promedio")
        ax.tick_params(axis="x", rotation=30)
        sns.despine()
        fig.tight_layout()
        plt.close(fig)
        return fig

    def grafico_top_rutas(
        self,
        df_rutas: pd.DataFrame,
        titulo: str = "Rutas con mayor demanda promedio",
    ) -> matplotlib.figure.Figure:
        """Barras horizontales de la salida de
        ProcesadorEDA.rutas_saturadas() (columnas esperadas:
        recorrido_normalizado, promedio_pasajeros_diarios)."""
        datos = df_rutas.sort_values("promedio_pasajeros_diarios")

        fig, ax = plt.subplots(figsize=self.figsize)
        paleta = sns.color_palette("Blues_r", n_colors=len(datos))
        barras = ax.barh(
            datos["recorrido_normalizado"],
            datos["promedio_pasajeros_diarios"],
            color=paleta,
        )

        for barra, valor in zip(barras, datos["promedio_pasajeros_diarios"]):
            ax.text(
                barra.get_width() + datos["promedio_pasajeros_diarios"].max() * 0.01,
                barra.get_y() + barra.get_height() / 2,
                f"{valor:,.0f}",
                va="center",
                fontsize=10,
            )
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_xlabel("Pasajeros promedio diarios")
        ax.set_xlim(0, datos["promedio_pasajeros_diarios"].max() * 1.18)
        sns.despine(left=True, bottom=True)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def heatmap_correlacion(
        self,
        df_corr: pd.DataFrame,
        titulo: str = "Correlacion clima vs. pasajeros",
    ) -> matplotlib.figure.Figure:
        """Heatmap de la salida de
        ProcesadorEDA.correlacion_clima_pasajeros()."""
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            df_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        fig.tight_layout()
        plt.close(fig)
        return fig

    def mapa_ciudades_clima(
        self,
        coordenadas: dict[str, tuple[float, float]] | None = None,
        clima_promedio_por_ciudad: dict[str, dict[str, float]] | None = None,
        tiles: str = "CartoDB positron",
        mostrar_corredores: bool = True,
    ) -> folium.Map:
        """Mapa interactivo con las ciudades usadas como referencia climatica
        y la red de corredores interprovinciales.
        """
        coordenadas = coordenadas or {
            "San Jose": (9.9325, -84.0795),
            "Heredia": (9.9989, -84.1165),
            "Cartago": (9.8644, -83.9195),
        }
        lat_centro = sum(lat for lat, _ in coordenadas.values()) / len(coordenadas)
        lon_centro = sum(lon for _, lon in coordenadas.values()) / len(coordenadas)

        mapa = folium.Map(location=[lat_centro, lon_centro], zoom_start=11, tiles=tiles)

        if mostrar_corredores and "San Jose" in coordenadas:
            hub = coordenadas["San Jose"]
            for ciudad, coord in coordenadas.items():
                if ciudad == "San Jose":
                    continue
                folium.PolyLine(
                    locations=[hub, coord],
                    color="#2c3e50",
                    weight=3,
                    opacity=0.6,
                    tooltip=f"Corredor San Jose - {ciudad}",
                ).add_to(mapa)

        if clima_promedio_por_ciudad:
            temperaturas = [v["temp_max_c"] for v in clima_promedio_por_ciudad.values()]
            temp_min_escala, temp_max_escala = min(temperaturas), max(temperaturas)

        for ciudad, (lat, lon) in coordenadas.items():
            if clima_promedio_por_ciudad and ciudad in clima_promedio_por_ciudad:
                datos = clima_promedio_por_ciudad[ciudad]
                temp = datos["temp_max_c"]
                lluvia = datos.get("precipitacion_mm", None)

                if temp_max_escala == temp_min_escala:
                    color = "blue"
                else:
                    posicion = (temp - temp_min_escala) / (temp_max_escala - temp_min_escala)
                    color = "red" if posicion > 0.66 else "orange" if posicion > 0.33 else "green"

                popup_texto = f"<b>{ciudad}</b><br>Temp. max. promedio: {temp:.1f}°C"
                if lluvia is not None:
                    popup_texto += f"<br>Precipitacion promedio: {lluvia:.1f}mm"
            else:
                color = "blue"
                popup_texto = f"<b>{ciudad}</b>"

            # Marcador con icono de tren
            folium.Marker(
                location=[lat, lon],
                popup=popup_texto,
                tooltip=ciudad,
                icon=folium.Icon(color=color, icon="train", prefix="fa"),
            ).add_to(mapa)

            # --- CORRECCION AQUI: El folium.Circle ahora esta DENTRO del ciclo ---
            if clima_promedio_por_ciudad and ciudad in clima_promedio_por_ciudad:
                folium.Circle(
                    location=[lat, lon],
                    radius=3000,  # 3km de radio ajustado
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.2,
                    opacity=0.5,
                    tooltip=popup_texto,
                ).add_to(mapa)

        return mapa

    def guardar(self, figura, ruta_salida: str) -> None:
        """Guarda una figura de matplotlib (.png/.pdf) o un mapa de folium (.html)."""
        if isinstance(figura, folium.Map):
            figura.save(ruta_salida)
        else:
            figura.savefig(ruta_salida, dpi=200, bbox_inches="tight")
        print(f"Guardado exitoso: {ruta_salida}")


def main():
    """Prueba rapida de integracion."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 500
    fechas = pd.date_range("2024-01-01", periods=n // 5).repeat(5)[:n]

    df_prueba = pd.DataFrame({
        "fecha": fechas,
        "pasajeros_totales": rng.integers(50, 500, n),
    })
    df_hora = pd.DataFrame({
        "hora": list(range(24)),
        "promedio_pasajeros": [20, 10, 5, 5, 15, 80, 250, 450, 300, 150, 120, 140, 160, 150, 140, 180, 400, 500, 320, 180, 100, 60, 40, 25],
    })
    df_demanda = pd.DataFrame({
        "nombre_dia": ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"],
        "promedio": [300, 280, 310, 295, 330, 120, 80],
    })
    df_rutas = pd.DataFrame({
        "recorrido_normalizado": ["San Jose - Cartago", "San Jose - Heredia", "San Jose - Alajuela"],
        "promedio_pasajeros_diarios": [5200, 4300, 2100],
    })
    df_corr = pd.DataFrame(
        [[1.0, -0.2, -0.45], [-0.2, 1.0, 0.05], [-0.45, 0.05, 1.0]],
        columns=["pasajeros_totales", "temp_max_c", "precipitacion_mm"],
        index=["pasajeros_totales", "temp_max_c", "precipitacion_mm"],
    )

    clima_ciudades = {
        "San Jose": {"temp_max_c": 24.5, "precipitacion_mm": 5.2},
        "Heredia": {"temp_max_c": 23.0, "precipitacion_mm": 6.1},
        "Cartago": {"temp_max_c": 21.0, "precipitacion_mm": 4.8},
    }

    viz = Visualizador()
    fig1 = viz.grafico_serie_temporal(df_prueba)
    fig2 = viz.grafico_demanda_por_hora(df_hora)
    fig3 = viz.grafico_demanda_semanal(df_demanda)
    fig4 = viz.grafico_top_rutas(df_rutas)
    fig5 = viz.heatmap_correlacion(df_corr)
    mapa = viz.mapa_ciudades_clima(clima_promedio_por_ciudad=clima_ciudades)

    print("Todos los graficos y el mapa folium se generaron sin errores.")



if __name__ == "__main__":
    main()