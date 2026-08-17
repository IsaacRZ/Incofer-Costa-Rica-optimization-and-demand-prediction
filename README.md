# INCOFER Costa Rica — Optimización y Predicción de Demanda

Proyecto de análisis de datos y machine learning sobre el servicio de tren
urbano de INCOFER en el Gran Área Metropolitana (GAM) de Costa Rica.
Combina datos abiertos de ARESEP (pasajeros diarios), feriados oficiales
(Nager.Date) y clima histórico (Open-Meteo) para explorar patrones de
demanda y, más adelante, predecirla.

## Arquitectura

Cada módulo de `src/` implementa **una sola clase**, con responsabilidad
única (patrón POO):

| Carpeta | Clase | Responsabilidad |
|---|---|---|
| `src/datos/` | `GestorDatos` | Carga y limpia el CSV crudo de ARESEP |
| `src/api/` | `ClienteAPI` | Descarga feriados CR y clima histórico (Nager.Date, Open-Meteo) |
| `src/helpers/` | `Utilidades` | Une viajes + feriados + clima; validaciones y formateo |
| `src/basedatos/` | `GestorBaseDatos` | Conecta y opera contra SQLite o PostgreSQL/TimescaleDB (SQLAlchemy) |
| `src/eda/` | `ProcesadorEDA` | Estadísticas descriptivas, correlaciones, detección de outliers |
| `src/visualizacion/` | `Visualizador` | Gráficos (líneas, barras, heatmap) y mapa interactivo |
| `src/modelos/` | `ModeloML` | *(en construcción)* Regresión y clasificación de demanda |

`main.py` es el único orquestador: instancia cada clase e inyecta las
dependencias entre ellas — ninguna clase crea instancias de otra por su
cuenta, lo que permite probarlas por separado.

## Prerrequisitos

- [uv](https://docs.astral.sh/uv/) (gestor de entornos/paquetes de Python)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (para PostgreSQL + TimescaleDB)
- Conexión a internet (para las APIs de feriados/clima)

## 1. Instalación

```bash
git clone <url-del-repo>
cd Incofer-Costa-Rica-optimization-and-demand-prediction
uv sync
```

`uv sync` lee `pyproject.toml` / `uv.lock` y crea el entorno virtual
(`.venv/`) con todas las dependencias (`pandas`, `sqlalchemy`,
`psycopg2-binary`, `requests`, `matplotlib`, `seaborn`, `folium`, etc.).

## 2. Levantar la base de datos (PostgreSQL + TimescaleDB)

El proyecto usa TimescaleDB (extensión de PostgreSQL) por las hypertables,
apropiadas para el volumen de datos de series de tiempo diarias.

```bash
docker compose up -d
docker compose ps      # confirma que "incofer_timescaledb" esta "Up"
```

Credenciales (definidas en `docker-compose.yml`):

| Campo | Valor |
|---|---|
| Host | `localhost` |
| Puerto | `5432` |
| Base de datos | `incofer` |
| Usuario | `incofer` |
| Contraseña | `incofer_dev_password` |

Para detener el contenedor sin perder los datos: `docker compose stop`.
Para eliminarlo completamente (incluye el volumen de datos): `docker compose down -v`.

## 3. Datos de entrada

Descarga el CSV de ARESEP ("Tren urbano de pasajeros") desde
[aresep.go.cr/datos-abiertos/tren-urbano-pasajeros](https://aresep.go.cr/datos-abiertos/tren-urbano-pasajeros/)
y colócalo en:

```
data/raw/aresep_tren.csv
```

> ⚠️ El CSV fuente suele traer problemas de codificación (UTF-8 mal
> reinterpretado, ej. "Código" → "CÃ³digo") si se edita en Excel antes de
> guardarlo. `GestorDatos` repara esto automáticamente — no hace falta
> pre-procesar el archivo, pero si lo editaste manualmente, mejor usa el
> CSV original descargado directo del portal.

## 4. Correr el pipeline completo

```bash
uv run python main.py
```

Esto ejecuta, en orden:

1. **`GestorDatos`** limpia el CSV → `data/processed/viajes_diarios.parquet`
2. **`ClienteAPI`** descarga feriados (2021-2026) y clima diario de San
   José, Heredia y Cartago
3. **`Utilidades.enriquecer_viajes`** une todo en un único DataFrame
4. **`GestorBaseDatos`** carga el resultado a PostgreSQL y lo convierte en
   hypertable (`create_hypertable`, particionada por `fecha`)

Al terminar deberías ver algo como:

```
--- Resumen de calidad del dataset limpio ---
Filas: 21,874
...
Cargadas 21,874 filas en la tabla 'viajes_diarios'.
'viajes_diarios' convertida en hypertable (particionada por 'fecha').
```

## 5. Verificar los datos (DBeaver)

1. Nueva conexión → **PostgreSQL** (no SQLite) con las credenciales de la
   sección 2.
2. `Test Connection` → `Finish`.
3. SQL de verificación rápida:

```sql
SELECT COUNT(*) FROM viajes_diarios;

SELECT ciudad_clima, COUNT(*) AS filas,
       SUM(CASE WHEN temp_max_c IS NULL THEN 1 ELSE 0 END) AS sin_clima
FROM viajes_diarios
GROUP BY ciudad_clima;
```

## 6. Notebooks (EDA y visualización)

Los notebooks en `notebooks/` consumen las clases de `src/` — no
recalculan nada por su cuenta:

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

> 💡 Si el mapa de `Visualizador.mapa_ciudades_clima()` sale en blanco
> dentro de VS Code, es una limitación del webview de notebooks (bloquea
> la carga de tiles externas), no un bug del código. Usa
> `viz.guardar(mapa, "mapa_clima.html")` y ábrelo directo en el navegador.

## 7. Siguiente paso: Modelos de Machine Learning

`src/modelos/` (clase `ModeloML`) está pendiente de construir. Cubrirá:

- **Regresión** — predicción del número de pasajeros por día/recorrido
  (variables: día tipo, recorrido, clima, feriado)
- **Clasificación** — nivel de ocupación (Baja/Media/Alta/Saturada/Sin
  servicio), usando `pasajeros_totales` y capacidad estimada por tipo de
  equipo

## Estructura del proyecto

```
.
├── docker-compose.yml       # PostgreSQL + TimescaleDB
├── main.py                  # Orquestador del pipeline
├── pyproject.toml
├── data/
│   ├── raw/                 # aresep_tren.csv (no versionado)
│   └── processed/           # viajes_diarios.parquet
├── notebooks/                # EDA y visualización exploratoria
└── src/
    ├── datos/gestor_datos.py
    ├── api/cliente_api.py
    ├── helpers/utilidades.py
    ├── basedatos/gestor_basedatos.py
    ├── eda/procesador_eda.py
    ├── visualizacion/visualizador.py
    └── modelos/               # (en construcción)
```

## Notas de calidad de datos conocidas

Documentadas en detalle en `notebooks/03_EDA.ipynb`, resumen rápido:

- **11 de 51 feriados en día hábil (2021-2026) no tienen servicio
  registrado** — 4 de ellos en 2021 (posible efecto pandemia, consistente
  con los informes oficiales de INCOFER de ese año); Navidad/Año Nuevo
  faltan solo desde 2023 (posible cambio de política operativa reciente).
- **Fin de semana tiene muestra muy chica** (194 sábados, 97 domingos vs.
  ~4,300 observaciones por día laboral) — cualquier promedio de
  sábado/domingo es estadísticamente menos confiable.
- **1-2 de agosto son outliers reales, no errores**: coinciden con la
  Romería a la Basílica de los Ángeles (ruta Cartago), que dispara la
  demanda muy por encima de lo normal cada año.
- **`adultos_mayores_faltante`** marca qué filas tenían el valor
  originalmente nulo antes de imputarlo a 0 — útil para no confundir
  "cero adultos mayores real" con "dato no reportado".

## Troubleshooting rápido

| Síntoma | Causa / solución |
|---|---|
| `ModuleNotFoundError: No module named 'psycopg2'` | Falta el driver: `uv add psycopg2-binary` |
| Texto con `Ã³`, `â€“` en columnas/valores | CSV editado en Excel (doble codificación); `GestorDatos` ya lo repara, usa el CSV original si persiste |
| `sqlalchemy.exc.NotSupportedError: table ... is not empty` en `crear_hypertable` | Falta `migrate_data => true` en el `SELECT create_hypertable(...)` |
| `ConnectionResetError` al pedir clima | Caída transitoria de red; `ClienteAPI` reintenta automático (4 intentos, backoff exponencial) |
| Mapa de folium en blanco en VS Code | Limitación del webview; exporta a `.html` y ábrelo en el navegador |