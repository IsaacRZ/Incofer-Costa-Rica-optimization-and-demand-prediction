"""
app.py

Dashboard Interactivo en Streamlit para el Analisis de Demanda y Prediccion
del Transporte Publico (INCOFER - Costa Rica).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_folium import st_folium

from src.eda.procesador_eda import ProcesadorEDA
from src.visualizacion.visualizador import Visualizador
from src.modelos.modelo_ml import modelo_ml

# Configuración de la página
st.set_page_config(
    page_title="INCOFER - Demand & Analytics",
    page_icon="🚆",
    layout="wide",
)


# --- Carga de Datos con Caché ---
@st.cache_data
def cargar_datos():
    try:
        return pd.read_parquet("data/processed/viajes_diarios.parquet")
    except Exception:
        # Datos de respaldo en caso de no encontrar el archivo procesado
        fechas = pd.date_range("2024-01-01", periods=100)
        return pd.DataFrame({
            "fecha": fechas,
            "recorrido_normalizado": np.random.choice(["San Jose - Heredia", "San Jose - Cartago"], 100),
            "pasajeros_totales": np.random.randint(200, 1000, 100),
            "temp_max_c": np.random.uniform(20, 28, 100),
            "precipitacion_mm": np.random.uniform(0, 10, 100),
            "nombre_dia": fechas.day_name(),
            "es_feriado": False,
        })


df = cargar_datos()
eda = ProcesadorEDA(df)
viz = Visualizador()

# --- Encabezado ---
st.title("🚆 Panel de Control e Inteligencia de Demanda - INCOFER")
st.markdown("Plataforma analítica y predictiva para la red ferroviaria nacional.")

# --- Métricas Principales (KPIs) ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Pasajeros Totales", f"{df['pasajeros_totales'].sum():,}")
kpi2.metric("Promedio Diario", f"{int(df['pasajeros_totales'].mean()):,} pax")
kpi3.metric("Ruta Más Concurrida", df.groupby("recorrido_normalizado")["pasajeros_totales"].sum().idxmax())
kpi4.metric("Días Analizados", f"{df['fecha'].nunique()} días")

st.divider()

# --- Pestañas del Dashboard ---
tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio", "🗺️ Red Ferroviaria y Clima", "🤖 Predicción ML"])

# TAB 1: EDA
with tab1:
    st.subheader("Comportamiento Semanal y Tendencias de Demanda")
    col1, col2 = st.columns(2)

    with col1:
        df_semana = eda.demanda_por_dia_semana()
        fig_semana = viz.grafico_demanda_semanal(df_semana)
        st.pyplot(fig_semana)

    with col2:
        df_rutas = eda.rutas_saturadas()
        fig_rutas = viz.grafico_top_rutas(df_rutas)
        st.pyplot(fig_rutas)

# TAB 2: Mapa Interactivo
with tab2:
    st.subheader("Cobertura Geográfica y Promedio Climático")
    mapa = viz.mapa_ciudades_clima()
    st_folium(mapa, width="100%", height=500)

# TAB 3: Predicción con Machine Learning
with tab3:
    st.subheader("Calculadora de Demanda y Nivel de Ocupación")
    st.markdown("Ingrese los parámetros del viaje para estimar la afluencia esperada.")

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        ruta_sel = st.selectbox("Recorrido", df["recorrido_normalizado"].unique())
        dia_sel = st.selectbox("Día de la Semana",
                               ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    with col_input2:
        temp_input = st.slider("Temperatura Máxima Esperada (°C)", 15.0, 35.0, 24.0)
        precip_input = st.slider("Precipitación Estimada (mm)", 0.0, 50.0, 2.0)

    with col_input3:
        es_feriado_input = st.checkbox("¿Es día feriado?")

    if st.button("🔮 Calcular Predicción", type="primary"):
        # Crear DataFrame para el modelo
        input_df = pd.DataFrame({
            "recorrido_normalizado": [ruta_sel],
            "nombre_dia": [dia_sel],
            "temp_max_c": [temp_input],
            "precipitacion_mm": [precip_input],
            "es_feriado": [es_feriado_input],
        })

        # Cargar modelo entrenado o entrenar al vuelo
        try:
            modelo_pipe = joblib.load("data/outputs/modelo_clasificacion.joblib")
            pred_ocupacion = modelo_pipe.predict(input_df)[0]

            st.success(f"**Nivel de Ocupación Estimado:** {pred_ocupacion}")

        except Exception:
            st.warning("Efectuando predicción simulada (entrene el modelo para guardar el artefacto final).")
            st.info("Nivel de Ocupación Estimado: **Alta**")