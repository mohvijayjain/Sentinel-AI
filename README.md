# 🛡️ Sentinel-AI

**A self-healing ML pipeline.** Sentinel-AI serves a LightGBM model that predicts NYC taxi trip duration. It watches production data for drift, decides on its own when the model needs retraining, retrains it, and promotes the new model only if it passes strict validation gates. An AI assistant (RAG) explains every decision in plain English.

**🚀 Live demo:** [mohvijayjain-sentinel-ai-frontendapp-1pecn3.streamlit.app](https://mohvijayjain-sentinel-ai-frontendapp-1pecn3.streamlit.app/)

> **Note:** the UI is always online, but the backend runs on AWS EC2 and is kept stopped to save cost. If the UI shows *Backend: Offline*, email **[mohvijayjain12@gmail.com](mailto:mohvijayjain12@gmail.com)** and I'll start it for you.

---

## What it does

| Step | What happens |
|---|---|
| **Serve** | FastAPI serves the current **Champion** model from the MLflow Model Registry and logs every prediction to PostgreSQL. |
| **Monitor** | Upload a month of production data. Three detectors compare it with the reference data: **statistical** (PSI / KS / JS), **SHAP** (feature-importance shift) and **prediction** drift. |
| **Decide** | A weighted drift score (0.4 · statistical + 0.3 · SHAP + 0.3 · prediction) maps to an action: `WAIT` < 0.25 ≤ `MONITOR` < 0.5 ≤ `ALERT` < 0.75 ≤ `RETRAIN`. |
| **Heal** | On `RETRAIN`, a **Challenger** model is trained and must pass the metric, bootstrap, segment and recent-data gates before it replaces the Champion. Otherwise it is rejected and the Champion stays. |
| **Explain** | Every monitoring run and retraining event is indexed in ChromaDB, so the AI Assistant can answer questions like *"Why was the challenger rejected?"* with cited sources. |

---

## Full project explanation

### 1. The problem

ML models degrade silently. The world changes (new routes, new pricing, holidays, an airport closure), the input data starts to look different from the training data, and predictions get worse, often without anyone noticing. In most teams, someone has to spot this, retrain by hand, and decide whether the new model is really better.

Sentinel-AI automates that loop: **detect drift → decide → retrain → validate → promote or reject → explain**, with a human able to see and question every step.

### 2. The model

- **Task:** regression. Predict NYC yellow-taxi **trip duration (seconds)** from the trip details known at pickup.
- **Data:** public NYC TLC trip records. Raw files go through one shared preprocessing step (`src/ingestion`) that cleans invalid rows, removes outliers and engineers time features. Training and monitoring use the **same** code, so features always match.
- **Features (11):** `trip_distance`, `pickup_hour`, `pickup_day_of_week`, `pickup_month`, `is_weekend`, `is_rush_hour`, `PULocationID`, `DOLocationID`, `payment_type`, `VendorID`, `RatecodeID`.
- **Algorithm:** LightGBM. The baseline reaches R² ≈ 0.86 (RMSE ≈ 336 s, MAE ≈ 203 s).
- **Registry:** every model is logged to MLflow. The production model is whichever version holds the **`Champion` alias**; the API loads it at startup.

### 3. Serving

FastAPI loads the Champion once (in the app lifespan) and serves `POST /predict`. After each response, a background task writes the input, prediction and model version to `prediction_logs` in PostgreSQL, so logging never slows down or breaks a prediction.

### 4. Drift detection: three independent views

A new batch of production data (for example, one month) is compared with the **reference data** the model was trained on:

| Detector | Question it answers | How |
|---|---|---|
| **Statistical** | Do the inputs look different? | Per feature: **PSI** (Population Stability Index), **Kolmogorov–Smirnov** test, **Jensen–Shannon** distance |
| **SHAP** | Is the model *reasoning* differently? | Compares mean \|SHAP\| feature importance on reference vs current data |
| **Prediction** | Are the *outputs* shifting? | Compares the distributions of the model's predictions on both datasets |

Using all three avoids false alarms. A feature can drift without hurting the model, and the model can change behaviour even when no single feature looks very different.

### 5. Scoring and decision

The three results are combined into one score from 0 to 1:

```
overall = 0.4 × statistical + 0.3 × SHAP + 0.3 × prediction
```

| Score | Action | Meaning |
|---|---|---|
| < 0.25 | `WAIT` | Nothing to do |
| 0.25 – 0.5 | `MONITOR` | Keep an eye on it |
| 0.5 – 0.75 | `ALERT` | Significant drift, a human should look |
| ≥ 0.75 | `RETRAIN` | Retrain automatically |

Every run is stored in `monitoring_runs` (scores, action, drifted features) and `drift_scores` (per-feature details).

### 6. Retraining and the validation gates

On `RETRAIN`, the orchestrator trains a **Challenger** on fresh data and compares it with the Champion on the **same frozen test set** (fixed rows, never used for training, so the comparison is fair). The Challenger must pass **every** gate:

1. **Metric gate:** RMSE at least 1% better than the Champion, with MAE no worse and R² no worse.
2. **Bootstrap gate:** the RMSE improvement must hold up under bootstrap resampling, not just be luck on one test sample.
3. **Segment gate:** no important segment (top pickup zones, JFK, LaGuardia, Newark) may get more than 10% worse, so a better average can't hide a broken airport segment.
4. **Recent-data gate:** on the most recent month, the Challenger may not be more than 10% worse than the Champion.

If all gates pass, the MLflow `Champion` alias moves to the new version (**promote**). If any gate fails, the Challenger is **rejected** and production stays untouched. Either way, the full result and the reasons are written to `retraining_events`.

### 7. The AI Assistant (RAG)

Each monitoring run and retraining event is turned into a text document, embedded with NVIDIA embeddings and stored in **ChromaDB**. When you ask a question:

1. The question is embedded and the closest documents are retrieved.
2. Documents that aren't relevant enough (cosine distance above `RAG_MAX_DISTANCE`, 0.85 by default) are dropped, so the LLM is never fed unrelated context.
3. The LLM answers **only from those documents** and cites them as `[Source n]`.

So *"Why was the challenger rejected?"* is answered from the real retraining record (which gate failed and by how much), not invented. If the vector index is lost, it can be rebuilt from PostgreSQL with `python -m src.rag.rebuild_index`.

### 8. Upload monitoring and the Stop button

From the UI you can upload any month of raw TLC data. The API saves it to a private temporary folder, runs the full pipeline (validate → preprocess → three detectors → score → save → index), and deletes the temporary files afterwards. Only one run can execute at a time, since detection loads the full reference dataset.

**Stop:** the run checks a stop flag between stages. When you press Stop, it ends at the next stage boundary and **nothing is saved**, because the final check happens before any database write.

### 9. Security

- API key (`X-API-Key`) on every endpoint except `/health`, including `/metrics`.
- CORS allowlist (no `*`), upload size limit, and file-type and column validation.
- Error messages to clients are generic; server logs are redacted (no keys or data).
- In production, only Caddy (ports 80/443, HTTPS) is public. PostgreSQL, MLflow and the API are reachable only inside the Docker network.
- Secrets live in `.env` / Streamlit secrets, never in git.

### 10. Deployment

- **Backend:** one AWS EC2 instance running Docker Compose (API, PostgreSQL, MLflow, Caddy). Caddy gets a free HTTPS certificate automatically through an `sslip.io` hostname.
- **UI:** Streamlit Community Cloud, which calls the API over HTTPS with the key stored in Streamlit secrets.
- **Cost control:** the instance is stopped when not in use and started on demand for demos.

---

## Architecture

```mermaid
flowchart LR
    User([User / Interviewer])

    subgraph UI["Streamlit Community Cloud"]
        ST[Streamlit UI<br/>Dashboard - Monitoring - Run - Retraining<br/>Predictions - AI Assistant]
    end

    subgraph EC2["AWS EC2 - Docker Compose"]
        CADDY[Caddy<br/>auto-HTTPS :443]
        API[FastAPI<br/>sentinel-api :8000]
        PG[(PostgreSQL<br/>runs - scores - events - logs)]
        ML[MLflow<br/>tracking + model registry]
        CH[(ChromaDB<br/>RAG index)]
    end

    NV[[NVIDIA API<br/>embeddings + LLM]]

    User --> ST
    ST -- "HTTPS + X-API-Key" --> CADDY
    CADDY -- reverse proxy --> API
    API --> PG
    API -- "load Champion / register Challenger" --> ML
    API --> CH
    API -- "embed + generate" --> NV
```

Only ports **80/443** are public. The API, PostgreSQL and MLflow have no public ports; they talk to each other on a private Docker network.

---

## API routing

Everything reaches the API through Caddy (`https://<host>/...` → `sentinel-api:8000`). Protected routes need the `X-API-Key` header.

```mermaid
flowchart TD
    C[Caddy :443] --> APP[FastAPI app]

    APP --> R1["Serving router - /"]
    APP --> R2["Monitoring router - /monitoring (key)"]
    APP --> R3["RAG router - /chat (key)"]
    APP --> R4["/metrics (key) (Prometheus)"]

    R1 --> H["GET /health (public)"]
    R1 --> P["POST /predict (key)"]
    R1 --> MI["GET /model/info (key)"]

    R2 --> L["GET /latest - /history"]
    R2 --> F["GET /drifted-features - /feature-scores"]
    R2 --> E["GET /retraining-events - /prediction-logs"]
    R2 --> RUN["POST /run (upload dataset)"]
    R2 --> STOP["POST /run/stop"]

    R3 --> CHAT["POST /chat"]
```

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | – | Service info |
| GET | `/health` | – | Liveness + whether the model is loaded |
| POST | `/predict` | 🔒 | Predict trip duration; logged to `prediction_logs` in the background |
| GET | `/model/info` | 🔒 | Champion version, metrics, load time |
| GET | `/monitoring/latest` | 🔒 | Latest monitoring run |
| GET | `/monitoring/history` | 🔒 | Past monitoring runs |
| GET | `/monitoring/drifted-features` | 🔒 | Features flagged in the latest run |
| GET | `/monitoring/feature-scores` | 🔒 | Per-feature PSI / KS / JS scores |
| GET | `/monitoring/retraining-events` | 🔒 | Retraining history (promoted / rejected, with reasons) |
| GET | `/monitoring/prediction-logs` | 🔒 | Recent predictions |
| POST | `/monitoring/run` | 🔒 | Upload a `.csv` / `.parquet` and run one monitoring cycle |
| POST | `/monitoring/run/stop` | 🔒 | Stop the run in progress (nothing is saved) |
| POST | `/chat` | 🔒 | Ask the AI Assistant (RAG over runs and events) |
| GET | `/metrics` | 🔒 | Prometheus metrics |

The API also enforces a CORS allowlist (`CORS_ALLOWED_ORIGINS`), a 300 MB upload limit, and one monitoring run at a time.

---

## Monitoring run flow

```mermaid
sequenceDiagram
    actor U as User
    participant UI as Streamlit
    participant API as FastAPI
    participant PL as Pipeline
    participant DB as PostgreSQL + ChromaDB

    U->>UI: Upload dataset, click Run Monitoring
    UI->>API: POST /monitoring/run (file, period)
    API->>PL: validate columns -> preprocess
    PL->>PL: statistical -> SHAP -> prediction drift
    opt User clicks Stop
        UI->>API: POST /monitoring/run/stop
        API-->>PL: stop flag (checked between stages)
        PL-->>UI: {"status": "stopped"} - nothing saved
    end
    PL->>PL: weighted drift score -> action
    PL->>DB: save run + feature scores, index for RAG
    API-->>UI: scores, action, drifted features
    alt action = RETRAIN
        API->>PL: start retraining in background
    end
```

## Self-healing loop

```mermaid
flowchart LR
    D[Drift score >= 0.75<br/>RETRAIN] --> T[Train Challenger<br/>LightGBM]
    T --> G{Validation gates<br/>RMSE - MAE - R2 - bootstrap<br/>segment - recent data}
    G -- all pass --> P[Promote: move MLflow<br/>Champion alias]
    G -- any fail --> R[Reject: keep Champion]
    P --> LOG[(retraining_events)]
    R --> LOG
    LOG --> RAG[Indexed for AI Assistant]
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Model | LightGBM, SHAP, scikit-learn, pandas, PyArrow |
| Drift detection | PSI, Kolmogorov–Smirnov, Jensen–Shannon, SHAP drift, prediction drift |
| MLOps | MLflow (tracking + registry with Champion alias), Prometheus metrics |
| API | FastAPI, Uvicorn, SQLAlchemy |
| Storage | PostgreSQL 15, ChromaDB |
| GenAI | RAG with NVIDIA embeddings + LLM, relevance threshold, cited sources |
| UI | Streamlit |
| Infra | Docker Compose, Caddy (HTTPS), AWS EC2, Streamlit Community Cloud |
| Testing | pytest (800+ tests, no external services needed) |

---

## Project layout

```
src/
  serving/      FastAPI app, /predict /health /model/info, auth, CORS, prediction logging
  api/routes/   /monitoring and /chat routers
  ingestion/    raw TLC data reading + preprocessing (shared by training and monitoring)
  monitoring/   drift detectors, drift runner, scorer, upload runs (with Stop)
  training/     retraining, promotion gates, MLflow logging, orchestrator
  evaluation/   bootstrap, segment and recent-data gates
  rag/          ChromaDB store, embeddings, retriever, LLM, index rebuild
  database/     PostgreSQL access and migrations
frontend/       Streamlit UI (views/ = pages, api_client.py = the only API caller)
deploy/         AWS EC2 deployment: compose override, Caddyfile, setup + data migration scripts
docker/mlflow/  MLflow server image
tests/          pytest suite
```

---

## Run locally

1. Copy `.env.example` to `.env` and set `API_KEY`, the PostgreSQL settings and `NVIDIA_API_KEY`.
2. Start the backend (API, PostgreSQL, MLflow):

   ```bash
   docker compose up -d --build
   ```

3. Start the UI (set `SENTINEL_API_URL` and `SENTINEL_API_KEY` in `frontend/.env`):

   ```bash
   cd frontend && streamlit run app.py
   ```

4. Run the tests:

   ```bash
   pytest
   ```

| Service | Local URL |
|---|---|
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Streamlit | http://localhost:8501 |

## Deploy

See [deploy/README.md](deploy/README.md) for the AWS EC2 setup (Caddy HTTPS, data migration) and [frontend/README.md](frontend/README.md) for the Streamlit UI.

## License

See [LICENSE](LICENSE).
