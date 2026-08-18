"""
src/modelos/modelo_ml.py

Clase ModeloML: Entrena y evalúa modelos supervisados (Regresión y Clasificación).
Cubre el Proyecto 1A (Regresión de pasajeros) y Proyecto 1B (Clasificación de ocupación)
siguiendo las mejores prácticas de ML (validación cruzada, GridSearchCV y prevención de data leakage).
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Modelos de Clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


class ModeloML:
    """Entrena, compara y optimiza modelos supervisados para INCOFER."""

    CAPACIDAD_REFERENCIA_FLOTA = {
        "apolo_pax": 180,
        "crrc_sencillo_pax": 372,
        "crrc_doble_pax": 700,
    }

    NIVELES_OCUPACION = ["Baja", "Media", "Alta", "Saturada"]

    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        columnas_requeridas = ["fecha", "recorrido_normalizado", "pasajeros_totales"]
        faltantes = set(columnas_requeridas) - set(df.columns)
        if faltantes:
            raise ValueError(f"ModeloML espera un DataFrame enriquecido. Faltan columnas: {faltantes}")

        self.df = df.copy()
        self.random_state = random_state
        self.mejor_modelo_clasificacion: Optional[Pipeline] = None
        self.mejor_modelo_regresion: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    # 1. Preparación del Target de Clasificación (Nivel de Ocupación)
    # ------------------------------------------------------------------
    def crear_variable_ocupacion(
            self,
            percentil_capacidad: float = 0.97,
            umbrales: tuple[float, float, float] = (0.25, 0.60, 0.85),
    ) -> pd.DataFrame:
        """Calcula 'capacidad_diaria_estimada', 'ocupacion_pct' y 'nivel_ocupacion'."""
        df = self.df.copy()

        # Techo empírico de capacidad por recorrido
        capacidad_ref = (
            df.groupby("recorrido_normalizado")["pasajeros_totales"]
            .quantile(percentil_capacidad)
            .to_dict()
        )

        df["capacidad_diaria_estimada"] = df["recorrido_normalizado"].map(capacidad_ref)
        df["ocupacion_pct"] = df["pasajeros_totales"] / df["capacidad_diaria_estimada"]

        u1, u2, u3 = umbrales
        condiciones = [
            df["ocupacion_pct"] <= u1,
            (df["ocupacion_pct"] > u1) & (df["ocupacion_pct"] <= u2),
            (df["ocupacion_pct"] > u2) & (df["ocupacion_pct"] <= u3),
            df["ocupacion_pct"] > u3,
        ]

        df["nivel_ocupacion"] = np.select(condiciones, self.NIVELES_OCUPACION, default="Media")
        self.df = df
        return df

    # ------------------------------------------------------------------
    # 2. PROYECTO 1A: REGRESIÓN (Predicción de Demanda Continua)
    # ------------------------------------------------------------------
    def comparar_modelos_regresion(
            self,
            df: pd.DataFrame,
            col_target: str = "pasajeros_totales",
            cols_num: Optional[List[str]] = None,
            cols_cat: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Entrena y evalúa modelos de regresión (LinearRegression, RandomForest, GradientBoosting)."""
        cols_num = cols_num or ["temp_max_c", "precipitacion_mm"]
        cols_cat = cols_cat or ["recorrido_normalizado", "nombre_dia", "es_feriado"]

        cols_num = [c for c in cols_num if c in df.columns]
        cols_cat = [c for c in cols_cat if c in df.columns]

        X = df[cols_num + cols_cat]
        y = df[col_target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), cols_num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
            ]
        )

        modelos = {
            "RegresionLineal": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            "GradientBoosting": GradientBoostingRegressor(random_state=self.random_state),
        }

        resultados = {}
        mejor_r2 = -float("inf")

        for nombre, model in modelos.items():
            pipeline = Pipeline([("prep", preprocessor), ("mod", model)])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            resultados[nombre] = {"RMSE": rmse, "MAE": mae, "R2": r2}

            if r2 > mejor_r2:
                mejor_r2 = r2
                self.mejor_modelo_regresion = pipeline

        return resultados

    # ------------------------------------------------------------------
    # 3. PROYECTO 1B: CLASIFICACIÓN (Modelo A con fuga vs Modelo B pronóstico)
    # ------------------------------------------------------------------
    def preparar_features_pronostico(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Modelo B: Pronóstico real usando solo variables exógenas (sin fuga)."""
        cols_cat = ["recorrido_normalizado", "nombre_dia", "es_feriado"]
        cols_num = ["temp_max_c", "precipitacion_mm"]

        cols_existentes = [c for c in cols_cat + cols_num if c in df.columns]
        X = df[cols_existentes].copy()
        y = df["nivel_ocupacion"]
        return X, y

    def comparar_modelos_clasificacion(
            self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Benchmarking con StratifiedKFold CV entre varios clasificadores."""
        cols_num = X.select_dtypes(include=[np.number]).columns.tolist()
        cols_cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), cols_num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
            ]
        )

        modelos = {
            "RegresionLogistica": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "ArbolDecision": DecisionTreeClassifier(random_state=self.random_state),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
        }

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        resultados = {}

        for nombre, model in modelos.items():
            pipeline = Pipeline([("prep", preprocessor), ("mod", model)])
            cv_results = cross_validate(
                pipeline, X, y, cv=skf, scoring=["accuracy", "f1_weighted"]
            )

            acc_mean = float(cv_results["test_accuracy"].mean())
            f1_mean = float(cv_results["test_f1_weighted"].mean())

            resultados[nombre] = {"Accuracy_CV": acc_mean, "F1_Weighted_CV": f1_mean}

        return resultados

    def optimizar_mejor_modelo(
            self, X: pd.DataFrame, y: pd.Series, nombre_modelo: str = "RandomForest"
    ) -> Pipeline:
        """GridSearchCV para optimizar los hiperparámetros del modelo ganador."""
        cols_num = X.select_dtypes(include=[np.number]).columns.tolist()
        cols_cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), cols_num),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
            ]
        )

        if nombre_modelo == "RandomForest":
            pipeline = Pipeline(
                [("prep", preprocessor), ("mod", RandomForestClassifier(random_state=self.random_state))])
            param_grid = {
                "mod__n_estimators": [50, 100, 200],
                "mod__max_depth": [None, 10, 20],
            }
        else:
            pipeline = Pipeline(
                [("prep", preprocessor), ("mod", LogisticRegression(max_iter=1000, random_state=self.random_state))])
            param_grid = {"mod__C": [0.1, 1.0, 10.0]}

        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
        grid.fit(X, y)

        self.mejor_modelo_clasificacion = grid.best_estimator_
        print(f"Mejores hiperparámetros ({nombre_modelo}): {grid.best_params_}")
        return grid.best_estimator_

    # ------------------------------------------------------------------
    # 4. Guardar Artefactos para el Dashboard (Streamlit)
    # ------------------------------------------------------------------
    def guardar_modelo(self, ruta_salida: str = "data/outputs/modelo_clasificacion.joblib") -> None:
        """Exporta el modelo entrenado a un archivo .joblib para usarlo en la app."""
        if self.mejor_modelo_clasificacion is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        joblib.dump(self.mejor_modelo_clasificacion, ruta_salida)
        print(f"Modelo guardado exitosamente en: {ruta_salida}")