"""
src/modelos/modelo_ml1.py

Clase ModeloML: Entrena y evalúa modelos supervisados (Regresión y Clasificación).
Proyecto 1A (Regresión de pasajeros)
Proyecto 1B (Clasificación de ocupación)
Siguiendo mejores prácticas de ML (validación cruzada, GridSearchCV y prevenir data leakage "fuga de datos"). 
"""

from typing import Dict, Tuple, Any, List, Optional
import pandas as pd
import numpy as np
import joblib

# Selección y Preprocesamiento
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline   

# Modelos de Regresión
from sklearn.linear_model import LinearRegression       # Regresion Lineal simple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor   # 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Modelos de Clasificación: Predecir Baja/Media/Alta/Saturada
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
                           #precisión/recall/F1, Tabla:acierto/error,

class modelo_ml1:
    """Entrena, compara y optimiza modelos supervisados para INCOFER."""

    CAPACIDAD_REFERENCIA_FLOTA = {
        "apollo_pax" : 180,
        "crrc_sencillo_pax": 372,
        "crrc_doble_pax": 700,
    }

    NIVELES_OCUPACION = ["Baja", "Media", "Alta", "Saturada"]

    def __init__(self, df: pd.Dataframe, random_state: int = 42):
        columnas_requeridas = ["fecha", "recorrido_normalizado", "pasajeros_totales"]
        faltantes = set(columnas_requeridas) - set(df.columns)
        if faltantes:
            raise ValueError(f"ModeloML espera un DataFrame enriquecido. Faltan columnas {faltantes}")
        
        self.df = df.copy()
        self.random_state = random_state
        self.mejor_modelo_clasificacion: Pipeline | None = None
        self.mejor_modelo_regresion: Pipeline | None = None

    # Preparación Target (Nivel de Ocupación)
    def crear_variable_ocupacion(
            self,
            percentil_capacidad: float = 0.97,
            umbrales: tuple[float,float,float] = (0.25, 0.60, 0.85),
            capacidad_por_anio: bool = True,
    ) -> pd.DataFrame:
        """
        Calcula 'capacidad_diaria_estimada', 'ocupacion_pct' y 'nivel_ocupacion'.

        capacidad_por_anio (default True): calcula el percentil de capacidad
        POR RECORRIDO Y POR ANIO en vez de un solo numero para
        historico. Un percentil global mezcla los anios de recuperacion
        pos-pandemia (2021-2022, capacidad/demanda mucho mas baja) con el
        sistema ya estabilizado (2023+), aplanando el crecimiento real
        (confirmado con datos reales: capacidad empirica de Cartago paso de
        ~1,023 en 2021 a ~4,090 en 2023, luego se estabilizo). Usar False
        para reproducir el comportamiento de percentil global unico.        
        """
        df = self.df.copy()

        if capacidad_por_anio:
            if "anio" not in df.columns:
                df["anio"] = pd.to_datetime(df["fecha"]).dt.year
            columnas_grupo = ["recorrido_normalizado", "anio"]
        else:
            columnas_grupo = ["recorrido_normalizado"]

        df["capacidad_diaria_estimada"] = df.groupby(columnas_grupo)