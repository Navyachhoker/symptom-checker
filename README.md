# AI Medical Triage Agent

### Multi-agent LangGraph · FastAPI · PostgreSQL · Groq · React

A conversational triage assistant. A user describes their symptoms; a LangGraph workflow extracts structured clinical details, runs rule-based safety checks and a clinical scorer, optionally consults a domain specialist, and returns an urgency level (`low`, `moderate`, `high`, `emergency`) with next-step advice. A separate safety agent reviews every decision and can only **raise** urgency, never lower it.

**Live demo:** https://medtriage-frontend.onrender.com/ (Render free tier, so the first request after idle can take 30–50 seconds)

> **Disclaimer:** Educational and demonstration project only. It is not a medical device and not a substitute for professional medical advice, diagnosis, or treatment. If you think you are having an emergency, call your local emergency number.

---

## Highlights

- **Orchestrator + specialists + safety reviewer** built on LangGraph with typed shared state.
- **Rule-based safety first:** a keyword red-flag check can bypass the LLM entirely and return an emergency response; a deterministic clinical scorer anchors the urgency estimate.
- **Upgrade-only safety layer:** hard-coded emergency overrides, then an LLM review; urgency can only move up.
- **Five specialist prompts:** cardiac, respiratory, mental health, pediatric, general. Cardiac and respiratory consultation is enforced in code before concluding.
- **Deterministic-first orchestration:** routing decisions are plain Python; the LLM is used for extraction, question wording, specialist assessment, final advice and safety review. Profiling showed LLM routing calls were ~85% of latency; replacing them cut high-urgency latency from ~30 s to ~4 s in development measurements.
- **Stateless API:** conversation history is rebuilt from PostgreSQL on every request.
- **Observability:** per-request ID in every log line, per-step timing, and a full agent reasoning trace returned to the UI.
- **Evaluation suite** with a single-agent baseline, classification, response-quality (rules + LLM-as-judge), consistency and latency scripts.

---

## How It Works

```
POST /api/chat
   │
   ▼
Validate + rate limit → load session & history from PostgreSQL
   │
   ▼
┌──────────────────────── orchestrator node ────────────────────────┐
│ 1. extract_symptoms     LLM → symptoms, duration, severity, age,   │
│                         existing conditions                        │
│ 2. check_red_flags      RULES → emergency bypass (skips the LLM)   │
│ 3. planning loop (≤ 3 steps):                                      │
│      ask_question  │ run clinical scorer │ call specialist │       │
│      conclude      │ escalate                                      │
└──────────────┬───────────────────────────────────────────┬─────────┘
               │ awaiting user input                       │ triage complete
               ▼                                           ▼
              END                                   safety agent
                                         hard overrides → LLM review
                                         (upgrade-only)   │
                                                          ▼
                                                         END
   │
   ▼
Save assistant reply + triage outcome → return reply, outcome, agent trace
```

**Typical paths**

| Path | LLM calls | Notes |
|---|---|---|
| Emergency red-flag bypass | 1–2 | Extraction, then (usually) a safety review |
| High / emergency scorer band | 3 | Extraction, final triage, safety review |
| Low / moderate band, or cardiac / respiratory | 4 | Adds one specialist consultation |

**Decision rules (in order)**

1. Max steps reached with confidence below 75 → escalate.
2. Cardiac or respiratory domain, scorer already run, no specialist yet → call that specialist.
3. No symptoms extracted and fewer than 2 questions asked → LLM phrases a clarifying question.
4. Scorer not yet run → run the clinical scorer.
5. Confidence ≥ 75, or a specialist has weighed in → conclude.
6. Otherwise → call a specialist.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph, LangChain (`langchain-groq`) |
| LLM | Groq-hosted `openai/gpt-oss-120b` (temperature 0.0 for orchestrator and safety, 0.1 for specialists) |
| API | FastAPI, Uvicorn, Pydantic v2, slowapi (rate limiting) |
| Database | PostgreSQL, SQLAlchemy 2.0 (async), asyncpg, Alembic |
| Frontend | React 18, Vite, axios |
| Deployment | Render (`render.yaml` blueprint), Dockerfiles for backend and frontend |
| Language | Python 3.11, JavaScript |

---

## Project Structure

```
symptom-checker/
├── render.yaml                      # Render blueprint: backend, frontend, Postgres
├── backend/
│   ├── Dockerfile                   # runs `alembic upgrade head` then Uvicorn
│   ├── requirements.txt
│   ├── alembic/                     # async migrations (3 revisions)
│   ├── app/
│   │   ├── main.py                  # app, lifespan, CORS, request-ID middleware, error handler
│   │   ├── config.py                # settings from environment / .env
│   │   ├── api/routes.py            # /chat, /history, /sessions, /token
│   │   ├── models/schemas.py        # Pydantic request/response models + input sanitising
│   │   ├── db/                      # async engine/session, ORM models
│   │   ├── core/logging_config.py   # request-ID logging, rotating file handler
│   │   └── agent/
│   │       ├── graph.py             # StateGraph assembly and routing
│   │       ├── state.py             # TriageState, AgentTrace
│   │       ├── orchestrator.py      # extraction, planning loop, final triage
│   │       ├── tools.py             # clinical scorer, red-flag checker, specialist router, escalation
│   │       ├── specialists.py       # cardiac / respiratory / mental / pediatric / general
│   │       ├── safety.py            # second-pass reviewer (upgrade-only)
│   │       └── nodes.py             # legacy single-agent nodes, kept for the baseline eval
│   └── eval/                        # evaluation scripts (see below)
└── frontend/                        # React chat UI with reasoning-trace panel
```

---

## API

Base path: `/api`

| Method | Endpoint | Rate limit | Description |
|---|---|---|---|
| `POST` | `/chat` | 20/min | Send a message; creates a session if `session_id` is omitted |
| `GET` | `/history/{session_id}` | 30/min | Messages and latest triage outcome for a session |
| `GET` | `/sessions` | 30/min | Paginated list of sessions (`limit`, `offset`) |
| `POST` | `/token` | 10/min | Issues an anonymous UUID token |
| `GET` | `/health` | – | Liveness check (served at `/health`, outside `/api`) |

Clients send the token in an `X-User-Token` header. Interactive docs are served at `/docs`.

**Request**

```json
{ "message": "Fever 38.5 and sore throat for 3 days. Severity 5/10. Age 32.", "session_id": null }
```

`message` must be 1–2000 characters; HTML-like tags and null bytes are stripped.

**Response (abridged)**

```json
{
  "session_id": "b3f1…",
  "reply": "Triage Assessment — Moderate urgency\n…",
  "is_complete": true,
  "triage_outcome": {
    "urgency": "moderate",
    "confidence": 72,
    "advice_text": "…",
    "symptoms_summary": "…",
    "specialist_called": "general"
  },
  "agent_trace": [
    { "agent": "orchestrator", "action": "extract_symptoms", "…": "…" },
    { "agent": "orchestrator", "action": "run_tool", "output": "Clinical score 3 → MODERATE risk …" },
    { "agent": "safety_agent", "action": "approve", "…": "…" }
  ]
}
```

---

## Data Model

- **sessions**: `id` (UUID), `created_at`, `updated_at`, `user_identifier`
- **messages**: `id`, `session_id` → sessions (cascade), `role` (user / assistant / system), `content`, `order`, `created_at`
- **triage_outcomes**: `id`, `session_id` → sessions (cascade), `urgency` (enum), `confidence`, `advice_text`, `symptoms_summary`, `specialist_called`, `created_at`

Schema changes are managed by Alembic. `create_all` runs only when `APP_ENV=development`.

---

## Setup & Run (local)

**Prerequisites:** Python 3.11, Node 20, PostgreSQL, a [Groq API key](https://console.groq.com).

1. **Database**
   ```sql
   CREATE DATABASE symptom_checker;
   CREATE USER app_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE symptom_checker TO app_user;
   ```

2. **Environment**: create a `.env` file in the repository root:
   ```env
   DATABASE_URL=postgresql://app_user:your_password@localhost/symptom_checker
   GROQ_API_KEY=your_groq_api_key
   APP_ENV=development
   FRONTEND_URL=http://localhost:5173
   ```
   `postgresql://` URLs are automatically converted to `postgresql+asyncpg://`.

3. **Backend**
   ```bash
   cd backend
   python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   alembic upgrade head
   uvicorn app.main:app --reload
   ```
   API docs: http://127.0.0.1:8000/docs

4. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Set `VITE_API_URL` if the backend is not at `http://localhost:8000/api`.

**Environment variables**

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes | PostgreSQL connection string |
| `GROQ_API_KEY` | yes | Groq API key |
| `APP_ENV` | no (default `development`) | `development` enables `create_all` and SQL echo |
| `FRONTEND_URL` | no | Allowed CORS origin(s), comma-separated; unset allows all origins |
| `VITE_API_URL` | frontend build | Backend base URL |

---

## Evaluation

Scripts live in `backend/eval/` and call the live LLM, so they need a valid `GROQ_API_KEY`. Run them from `backend/`:

```bash
python -m eval.baseline_single_agent_test   # single-agent baseline (legacy nodes)
python -m eval.test_classification          # 15 cases across all four urgency levels; emergency false-negative rate
python -m eval.test_llm_quality             # rule-based checks + LLM-as-judge tone (--runs N, --strict-urgency)
python -m eval.test_consistency             # repeat-run consistency, edge cases, regressions, specialist routing
python -m eval.test_latency                 # tool latency and per-node pipeline latency (PIPELINE_RUNS=N)
```

These are evaluation scripts, not a unit-test suite. They measure agreement with expected labels written by the author, **not clinical accuracy**.

---

## Deployment

`render.yaml` provisions a backend web service, a frontend web service and a free PostgreSQL database. The backend start command runs `alembic upgrade head` before Uvicorn. Set `GROQ_API_KEY` and `FRONTEND_URL` manually in the Render dashboard. Dockerfiles are provided for both services.

---

## Known Limitations

**Clinical**
- Not validated against clinical guidelines (e.g. Manchester Triage System, ESI) or reviewed by medical professionals. The clinical scorer is a simplified, NEWS2-style point system with author-chosen weights, not NEWS2.
- The LLM can produce plausible but wrong assessments; temperature 0 reduces variance but does not remove it.
- The evaluation set is small (15 classification cases) and self-labelled. A passing score shows consistency with expected outputs, not safety.
- Red-flag and safety-override rules are English keyword matches over the conversation text. They do not handle negation ("no seizures"), paraphrases or other languages.
- The agent asks follow-up questions only when it cannot extract any symptoms; otherwise it concludes with the information it has.
- Escalation returns a fixed "contact a medical professional" message; there is no human review workflow.
- No memory across sessions.

**Security and reliability**
- The anonymous UUID token is session scoping, not authentication. Ownership checks run only when the `X-User-Token` header is sent, and `/api/sessions` without a token lists all sessions.
- No prompt-injection defences.
- Rate limiting is in-memory and keyed on client IP.
- Single LLM provider with no retry, fallback or circuit breaker; a Groq outage fails the request.
- The database connection is held for the duration of the LLM-bound request, which limits concurrency.

**Hosting**
- Render free tier: the backend sleeps after inactivity and the free database has size and retention limits.

### Roadmap

- Mandatory, server-validated authentication and owner-scoped queries
- Negation-aware red-flag detection restricted to user messages
- Gate conclusions on severity and duration so follow-up questions actually happen
- Schema-validated structured output with retry
- LLM retries, timeouts and a fallback provider
- Unit tests with a mocked LLM, API tests and CI
- Retrieval over triage guidelines to ground specialist advice
- Clinician-labelled evaluation set

---

## Author

**Navya Chhoker**
