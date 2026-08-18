# INCOFER Costa Rica — Demand Optimization & Prediction

Data analysis and machine learning project on INCOFER's urban train
service in Costa Rica's Greater Metropolitan Area (GAM). Combines ARESEP
open data (daily ridership), official holidays (Nager.Date), and
historical weather (Open-Meteo) to explore demand patterns and predict
them.

## Team

| Role | Name |
|---|---|
| **Tech Lead** | Isaac Rodriguez |
| Contributor | Byron Perez |
| Contributor | Jaf420710 |

## Architecture

Each module under `src/` implements **a single class** with a single
responsibility (OOP pattern):

| Folder | Class | Responsibility |
|---|---|---|
| `src/datos/` | `GestorDatos` | Loads and cleans the raw ARESEP CSV |
| `src/api/` | `ClienteAPI` | Fetches CR holidays and historical weather (Nager.Date, Open-Meteo) |
| `src/helpers/` | `Utilidades` | Merges trips + holidays + weather; validation and formatting |
| `src/basedatos/` | `GestorBaseDatos` | Connects to and operates SQLite or PostgreSQL/TimescaleDB (SQLAlchemy) |
| `src/eda/` | `ProcesadorEDA` | Descriptive statistics, correlations, outlier detection |
| `src/visualizacion/` | `Visualizador` | Charts (lines, bars, heatmap) and interactive map |
| `src/modelos/` | `ModeloML` | Regression (demand forecasting) and classification (occupancy level) |

`main.py` is the single orchestrator: it instantiates each class and
injects dependencies between them — no class creates instances of
another on its own, which keeps every class independently testable.

`app.py` is an interactive Streamlit dashboard consuming the same
classes (EDA, visualizations, and the trained classification model) for
exploration and live prediction.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python environment/package manager)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for PostgreSQL + TimescaleDB)
- Internet connection (for the holidays/weather APIs)

## 1. Installation

```bash
git clone <repo-url>
cd Incofer-Costa-Rica-optimization-and-demand-prediction
uv sync
```

`uv sync` reads `pyproject.toml` / `uv.lock` and creates the virtual
environment (`.venv/`) with all dependencies (`pandas`, `sqlalchemy`,
`psycopg2-binary`, `requests`, `matplotlib`, `seaborn`, `folium`,
`scikit-learn`, `joblib`, `streamlit`, `streamlit-folium`, etc.).

## 2. Start the database (PostgreSQL + TimescaleDB)

The project uses TimescaleDB (a PostgreSQL extension) for hypertables,
suited to the daily time-series volume of this dataset.

```bash
docker compose up -d
docker compose ps      # confirms "incofer_timescaledb" is "Up"
```

Credentials (defined in `docker-compose.yml`):

| Field | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Database | `incofer` |
| User | `incofer` |
| Password | `incofer_dev_password` |

To stop the container without losing data: `docker compose stop`.
To remove it completely (including the data volume): `docker compose down -v`.

## 3. Input data

Download the ARESEP CSV ("Urban Train Passengers") from
[aresep.go.cr/datos-abiertos/tren-urbano-pasajeros](https://aresep.go.cr/datos-abiertos/tren-urbano-pasajeros/)
and place it at:

```
data/raw/aresep_tren.csv
```

> ⚠️ The source CSV commonly has encoding issues (UTF-8 misread and
> re-saved, e.g. "Código" → "CÃ³digo") if it was opened/edited in Excel.
> `GestorDatos` repairs this automatically — no pre-processing needed,
> but if you edited the file manually, prefer the original download from
> the portal.

## 4. Run the full pipeline

```bash
uv run python main.py
```

This runs, in order:

1. **`GestorDatos`** cleans the CSV → `data/processed/viajes_diarios.parquet`
2. **`ClienteAPI`** fetches holidays (2021-2026) and daily weather for San
   José, Heredia, and Cartago
3. **`Utilidades.enriquecer_viajes`** merges everything into a single
   enriched DataFrame (saved as the final Parquet)
4. **`GestorBaseDatos`** loads the result into PostgreSQL and converts it
   into a hypertable (`create_hypertable`, partitioned by `fecha`)
5. **`ProcesadorEDA`** prints exploratory statistics (weekly demand, top
   routes)
6. **`Visualizador`** generates charts to `data/outputs/`
7. **`ModeloML`** builds the occupancy target, benchmarks classifiers
   (Logistic Regression, Decision Tree, Random Forest) with cross-
   validation, optimizes the winner with `GridSearchCV`, and exports the
   trained model to `data/outputs/modelo_clasificacion.joblib`

On success you should see something like:

```
--- Resumen de calidad del dataset limpio ---
Filas: 21,874
...
Cargadas 21,874 filas en la tabla 'viajes_diarios'.
'viajes_diarios' convertida en hypertable (particionada por 'fecha').
...
Mejores hiperparámetros (RandomForest): {...}
Modelo guardado exitosamente en: data/outputs/modelo_clasificacion.joblib
🎉 Pipeline ejecutado exitosamente de principio a fin.
```

## 5. Verify the data (DBeaver)

1. New connection → **PostgreSQL** (not SQLite) with the credentials from
   section 2.
2. `Test Connection` → `Finish`.
3. Quick verification SQL:

```sql
SELECT COUNT(*) FROM viajes_diarios;

SELECT ciudad_clima, COUNT(*) AS filas,
       SUM(CASE WHEN temp_max_c IS NULL THEN 1 ELSE 0 END) AS sin_clima
FROM viajes_diarios
GROUP BY ciudad_clima;
```

## 6. Notebooks (EDA and visualization)

The notebooks in `notebooks/` consume the `src/` classes — they don't
recompute anything on their own:

```python
from src.basedatos.gestor_basedatos import GestorBaseDatos
from src.eda.procesador_eda import ProcesadorEDA
from src.visualizacion.visualizador import Visualizador

gestor = GestorBaseDatos("postgresql+psycopg2://incofer:incofer_dev_password@localhost:5432/incofer")
df = gestor.consultar("SELECT * FROM viajes_diarios")
gestor.cerrar()

eda = ProcesadorEDA(df)
viz = Visualizador()
```

> 💡 If `Visualizador.mapa_ciudades_clima()` renders blank inside VS
> Code, that's a notebook webview limitation (it blocks loading external
> map tiles), not a code bug. Use `viz.guardar(mapa, "mapa_clima.html")`
> and open it directly in your browser.

## 7. Interactive dashboard (Streamlit)

```bash
docker compose up -d          # database must be running
uv run streamlit run app.py
```

Opens a browser tab with three tabs: exploratory analysis, the
geographic/weather map, and an occupancy-level prediction form powered
by the trained model in `data/outputs/modelo_clasificacion.joblib`.

## Project structure

```
.
├── docker-compose.yml       # PostgreSQL + TimescaleDB
├── main.py                  # Pipeline orchestrator
├── app.py                   # Streamlit dashboard
├── pyproject.toml
├── data/
│   ├── raw/                 # aresep_tren.csv (not versioned)
│   ├── processed/           # viajes_diarios.parquet (enriched)
│   └── outputs/             # charts, exported model
├── notebooks/                # EDA and exploratory visualization
└── src/
    ├── datos/gestor_datos.py
    ├── api/cliente_api.py
    ├── helpers/utilidades.py
    ├── basedatos/gestor_basedatos.py
    ├── eda/procesador_eda.py
    ├── visualizacion/visualizador.py
    └── modelos/modelo_ml.py
```

## Known data quality notes

Documented in detail in `notebooks/03_EDA.ipynb`, quick summary:

- **11 of 51 weekday holidays (2021-2026) have no recorded service** — 4
  of them in 2021 (possible pandemic effect, consistent with INCOFER's
  own official reports for that year); Christmas/New Year's are missing
  only from 2023 onward (possible recent operational policy change).
- **Weekends have a very small sample** (194 Saturdays, 97 Sundays vs.
  ~4,300 observations per weekday) — any Saturday/Sunday average is
  statistically less reliable.
- **August 1-2 are real outliers, not errors**: they coincide with the
  Romería a la Basílica de los Ángeles pilgrimage (Cartago route), which
  spikes demand well above normal every year.
- **`adultos_mayores_faltante`** flags which rows originally had a null
  value before it was imputed to 0 — useful for not confusing "genuinely
  zero senior riders" with "data not reported".
- **Empirical capacity ceiling is computed per route AND per year**
  (`ModeloML.crear_variable_ocupacion(capacidad_por_anio=True)`), not as
  a single number across the whole 2021-2026 range — a global percentile
  would blend the pandemic-recovery years (lower ridership/fleet) with
  the stabilized post-2023 system, flattening real growth (confirmed
  with real data: Cartago's empirical ceiling rose from ~1,023 in 2021
  to ~4,090 in 2023, then stabilized).
- **Classification accuracy is reported for two variants**: Model A
  (per the project's literal input spec, including `pasajeros_totales`
  and `capacidad_diaria_estimada` as features) shows near-100% accuracy
  because the target is a deterministic function of those same inputs —
  this is expected and documented, not a modeling error. Model B (no
  leakage, forecast-only features: calendar, weather, route) gives the
  honest predictive accuracy and is the one relevant to real-world
  frequency planning.

## Quick troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'psycopg2'` | Missing driver: `uv add psycopg2-binary` |
| `ModuleNotFoundError: No module named 'sklearn'` | `uv add scikit-learn joblib` |
| Text with `Ã³`, `â€“` in columns/values | CSV edited in Excel (double encoding); `GestorDatos` already repairs it — use the original CSV if it persists |
| `sqlalchemy.exc.NotSupportedError: table ... is not empty` in `crear_hypertable` | Missing `migrate_data => true` in `SELECT create_hypertable(...)` |
| `ConnectionResetError` when fetching weather | Transient network drop; `ClienteAPI` retries automatically (4 attempts, exponential backoff) |
| Folium map blank in VS Code | Webview limitation; export to `.html` and open in a browser |
| `missing ScriptRunContext` warning | You ran `python app.py` instead of `streamlit run app.py` |
| Streamlit app shows random/fake data | `app.py` must load data via `GestorBaseDatos` from Postgres — it should **fail loudly** if the DB isn't reachable, never silently fall back to synthetic data |