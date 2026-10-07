# Sentinel-AI Monitoring UI

A small Streamlit dashboard over the **existing** Sentinel-AI FastAPI service:

**Monitor → detect drift → decide → retrain → train a Challenger → validate → promote or reject → explain with AI**

The UI is HTTP-only: it never touches PostgreSQL, MLflow or ChromaDB. Every value it shows comes from the API. When the API has no data, the UI says so instead of making something up.

| Page | Shows | API endpoints used |
|---|---|---|
| Dashboard | Model health (version, R², MAE, RMSE), current decision, the three detector scores, recent activity | `/model/info`, `/monitoring/latest`, `/monitoring/history`, `/monitoring/retraining-events` |
| Monitoring | Overall score and action, detector scores, latest PSI per feature, drift over time | `/monitoring/latest`, `/monitoring/feature-scores`, `/monitoring/history` |
| Retraining | Latest Champion vs Challenger result, the final decision, validation gates, event history | `/monitoring/retraining-events` |
| Predictions | Recent served predictions | `/monitoring/prediction-logs` |
| AI Assistant | Questions answered from Sentinel's own records (RAG) | `POST /chat` |
| Sidebar | Backend connected or offline | `/health` (public) |

## Setup

1. Start the backend (from the repository root):

   ```bash
   docker compose up -d
   ```

   The API is published on `http://localhost:8000`.

2. Configure the UI. Copy the template and set the key to the API's `API_KEY` from the root `.env`:

   ```bash
   cp frontend/.env.example frontend/.env
   ```

   | Variable | Meaning |
   |---|---|
   | `SENTINEL_API_URL` | API base URL (default `http://localhost:8000`) |
   | `SENTINEL_API_KEY` | Sent as `X-API-Key` on every data call. Required, because every endpoint except `/health` and `/` needs it. It stays inside the Streamlit server process and is never displayed. |

   You can also export the two variables in your shell instead of using `frontend/.env`. That file is gitignored, so the key never gets committed.

3. Install the UI dependencies (they're already in the project venv) and run the app from the `frontend/` directory, so its dark theme in `frontend/.streamlit/config.toml` is applied:

   ```bash
   pip install -r frontend/requirements.txt
   cd frontend && streamlit run app.py
   ```

   Running `streamlit run frontend/app.py` from the repository root works too, but uses Streamlit's default theme.

Streamlit calls the API server-side, so no browser CORS configuration is needed.

## States you may see

| Message | Meaning |
|---|---|
| Set `SENTINEL_API_KEY` to connect | The key isn't configured. |
| Not authenticated, check `SENTINEL_API_KEY` | The API returned 401: the key is wrong. |
| Backend offline | The API can't be reached at `SENTINEL_API_URL`. |
| … is not available on the running API | The running API image predates that endpoint. Rebuild it with `docker compose build sentinel-api && docker compose up -d --no-deps sentinel-api`. |
| The assistant timed out, try again | `/chat` depends on the NVIDIA LLM and can take 20–50 s, occasionally longer. |

Validation gate results (Frozen, Bootstrap, Segment, Recent) are computed during promotion but only written to the promotion log, not to the database. The Retraining page therefore shows them as "not recorded" and doesn't guess.
