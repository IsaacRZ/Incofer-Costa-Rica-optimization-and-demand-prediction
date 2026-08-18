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


class modelo_ml:
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
            capacidad_por_anio: bool = True,
    ) -> pd.DataFrame:
        """Calcula 'capacidad_diaria_estimada', 'ocupacion_pct' y 'nivel_ocupacion'.

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

        df["capacidad_diaria_estimada"] = df.groupby(columnas_grupo)["pasajeros_totales"].transform(
            lambda s: s.quantile(percentil_capacidad)
        )
        df["ocupacion_pct"] = df["pasajeros_totales"] / df["capacidad_diaria_estimada"]

        u1, u2, u3 = umbrales
        condiciones = [
            df["ocupacion_pct"] <= u1,
            (df["ocupacion_pct"] > u1) & (df["ocupacion_pct"] <= u2),
            (df["ocupacion_pct"] > u2) & (df["ocupacion_pct"] <= u3),
            df["ocupacion_pct"] > u3,
        ]

        df["nivel_ocupacion"] = np.select(condiciones, self.NIVELES_OCUPACION, default="Media")
        self._capacidad_por_anio = capacidad_por_anio
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
        cols_cat = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

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
        cols_cat = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

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

if __name__ == "__main__":
    import os

    print("--- PRUEBA AISLADA DEL MÓDULO DE MACHINE LEARNING ---")

    # 1. Generar datos sintéticos de prueba con la estructura exacta esperada
    rng = np.random.default_rng(42)
    n_samples = 300

    fechas = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")
    recorridos = ["San Jose - Heredia", "San Jose - Cartago", "Heredia - Alajuela"]

    df_demo = pd.DataFrame({
        "fecha": fechas,
        "recorrido_normalizado": rng.choice(recorridos, n_samples),
        "pasajeros_totales": rng.integers(100, 1200, n_samples),
        "temp_max_c": rng.uniform(18.0, 29.0, n_samples),
        "precipitacion_mm": rng.uniform(0.0, 25.0, n_samples),
        "nombre_dia": fechas.day_name(),
        "dia_semana": fechas.dayofweek,
        "es_feriado": rng.choice([True, False], n_samples, p=[0.1, 0.9]),
    })

    # 2. Instanciar la clase y preparar variables
    ml = modelo_ml(df_demo)
    df_target = ml.crear_variable_ocupacion()

    # 3. Probar Regresión
    print("\n--- 1. Evaluando Modelos de Regresión (Demanda Continua) ---")
    res_reg = ml.comparar_modelos_regresion(df_target)
    for modelo_nombre, metricas in res_reg.items():
        print(
            f"  • {modelo_nombre:18s} -> R²: {metricas['R2']:.3f} | RMSE: {metricas['RMSE']:.2f} | MAE: {metricas['MAE']:.2f}")

    # 4. Probar Clasificación (Pronóstico sin fuga)
    print("\n--- 2. Evaluando Modelos de Clasificación (Nivel Ocupación) ---")
    X_b, y_b = ml.preparar_features_pronostico(df_target)
    res_cls = ml.comparar_modelos_clasificacion(X_b, y_b)
    for modelo_nombre, metricas in res_cls.items():
        print(
            f"  • {modelo_nombre:18s} -> Accuracy CV: {metricas['Accuracy_CV']:.3f} | F1 Weighted CV: {metricas['F1_Weighted_CV']:.3f}")

    # 5. Probar optimización y guardado de modelo
    print("\n--- 3. Optimizando Hiperparámetros y Exportando Artefacto ---")
    os.makedirs("data/outputs", exist_ok=True)
    mejor_modelo = ml.optimizar_mejor_modelo(X_b, y_b, nombre_modelo="RandomForest")
    ml.guardar_modelo("data/outputs/modelo_clasificacion.joblib")

    print("\n¡Prueba aislada completada con éxito!")