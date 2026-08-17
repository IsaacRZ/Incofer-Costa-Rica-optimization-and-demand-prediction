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
        ventana_promedio_movil: int = 30,
        titulo: str = "Pasajeros totales por dia",
    ) -> matplotlib.figure.Figure:
        """Linea de tiempo: agrega `columna_valor` por `columna_fecha`
        (sumando todos los recorridos de cada dia) y la grafica.

        La serie diaria cruda es muy ruidosa (huecos por feriados sin
        servicio, fines de semana con pocos datos, picos de eventos como
        la Romeria), lo que hace dificil ver la tendencia real a simple
        vista. Se agrega una linea de promedio movil (30 dias por defecto)
        encima, mas gruesa, para que la tendencia de fondo sea legible sin
        perder la serie original como referencia de fondo.
        """
        serie = df.groupby(columna_fecha)[columna_valor].agg(agregacion)
        promedio_movil = serie.rolling(window=ventana_promedio_movil, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(serie.index, serie.values, linewidth=0.5, alpha=0.35,
                 color="steelblue", label=f"{agregacion} diario")
        ax.plot(promedio_movil.index, promedio_movil.values, linewidth=2.2,
                 color="firebrick", label=f"promedio movil {ventana_promedio_movil}d")
        ax.set_title(titulo)
        ax.set_xlabel("Fecha")
        ax.set_ylabel(columna_valor)
        ax.legend()
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
        sns.barplot(data=df_demanda, x="nombre_dia", y="promedio", ax=ax)
        ax.set_title(titulo)
        ax.set_xlabel("Dia de la semana")
        ax.set_ylabel("Pasajeros promedio")
        ax.tick_params(axis="x", rotation=30)
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
        # Etiqueta con el valor exacto al final de cada barra: mas facil
        # de leer que solo confiar en el eje X.
        for barra, valor in zip(barras, datos["promedio_pasajeros_diarios"]):
            ax.text(
                barra.get_width() + datos["promedio_pasajeros_diarios"].max() * 0.01,
                barra.get_y() + barra.get_height() / 2,
                f"{valor:,.0f}",
                va="center",
                fontsize=10,
            )
        ax.set_title(titulo)
        ax.set_xlabel("Pasajeros promedio diarios")
        ax.set_xlim(0, datos["promedio_pasajeros_diarios"].max() * 1.15)  # espacio para las etiquetas
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
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title(titulo)
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
        """Mapa interactivo (punto extra) con las ciudades usadas como
        referencia climatica en Utilidades.MAPA_RECORRIDO_CIUDAD.

        `tiles`: estilo de mapa base. "CartoDB positron" (default) es un
        estilo minimalista sin la sobrecarga de carreteras/POI de
        OpenStreetMap estandar, mas apropiado para un mapa tematico de
        una red ferroviaria especifica. Otras opciones validas de folium:
        "OpenStreetMap", "CartoDB dark_matter".

        `mostrar_corredores`: dibuja lineas entre San Jose y cada
        provincia conectada (Heredia, Cartago), representando los
        corredores interprovinciales reales de INCOFER -- para que el
        mapa comunique "red ferroviaria", no solo puntos sueltos.
        NOTA: las lineas conectan los centros de ciudad (aproximacion),
        no el trazado exacto de la via ferrea, que no tenemos geocodificado.

        Sin `clima_promedio_por_ciudad`, solo marca las ciudades. Si se
        pasa (ej. df_clima.groupby('ciudad')[['temp_max_c',
        'precipitacion_mm']].mean().to_dict('index')), colorea cada
        marcador segun su temperatura promedio (verde=mas fresco,
        rojo=mas calido) y muestra las cifras en el popup.
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
                # Escala simple de color: mas caliente relativo -> rojo/naranja,
                # mas fresco relativo -> verde/azul.
                if temp_max_escala == temp_min_escala:
                    color = "blue"
                else:
                    posicion = (temp - temp_min_escala) / (temp_max_escala - temp_min_escala)
                    color = "red" if posicion > 0.66 else "orange" if posicion > 0.33 else "green"
                popup_texto = f"{ciudad}<br>Temp. max. promedio: {temp:.1f}°C"
                if lluvia is not None:
                    popup_texto += f"<br>Precipitacion promedio: {lluvia:.1f}mm"
            else:
                color = "blue"
                popup_texto = ciudad

            folium.Marker(
                location=[lat, lon],
                popup=popup_texto,
                tooltip=ciudad,
                icon=folium.Icon(color=color, icon="train", prefix="fa"),
            ).add_to(mapa)

        if clima_promedio_por_ciudad and ciudad in clima_promedio_por_ciudad:
                        folium.Circle(
                            location=[lat, lon],
                            radius=6000,  # metros
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.25,
                            opacity=0.5,
                            tooltip=popup_texto,
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