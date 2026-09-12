# AI Medical Triage Agent

### LangGraph · FastAPI · PostgreSQL · Groq LLM

> A conversational AI agent that triages patient symptoms through multi-turn dialogue, built with LangGraph, FastAPI, and PostgreSQL.

---

##  Overview

This project implements a lightweight medical triage assistant that engages users in a structured conversation to assess symptom severity and recommend appropriate next steps. The agent intelligently decides when it has gathered enough context to issue a triage decision — and when it needs to ask one more targeted question first.

**Key design principle:** The agent asks a single clarification per interaction and then reassesses whether enough information has been gathered to make a triage recommendation.

---

##  Highlights

- Built a stateful AI agent using LangGraph
- Designed a multi-turn symptom assessment workflow
- Integrated Groq-hosted LLMs for real-time reasoning
- Implemented conditional routing and graph cycles
- Persisted conversations using PostgreSQL and SQLAlchemy
- Exposed functionality through FastAPI REST endpoints

---

## Disclaimer

This project is intended for educational and demonstration purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional for medical concerns.

---

##  How It Works

```
1. User submits symptoms via POST /chat
2. LangGraph agent enters the assess_node
   ├── Not enough info? → Returns ONE follow-up question (stage: "assess")
   └── Enough info?    → Routes to recommend_node
3. recommend_node generates urgency level + actionable advice (stage: "done")
4. Full conversation (messages + outcome) is persisted to PostgreSQL
```

**Urgency levels returned:** `LOW` · `MEDIUM` · `HIGH`

---

##  Architecture

```
User
  │
  ▼
POST /chat  (FastAPI + Uvicorn)
  │
  ▼
┌─────────────────────────────┐
│        assess_node          │  ← LangGraph node
│                             │
│  Enough info?               │
│  ├── NO  → FOLLOWUP (ask)  │
│  └── YES → READY (proceed) │
└──────────────┬──────────────┘
               │ READY
               ▼
┌─────────────────────────────┐
│       recommend_node        │  ← LangGraph node
│  Urgency level + advice     │
└──────────────┬──────────────┘
               │
               ▼
         PostgreSQL
  (sessions · messages · triage_outcomes)
```

---

##  LangGraph Concepts Demonstrated

This project showcases several core LangGraph concepts:

- Stateful workflow management
- Conditional routing between nodes
- Multi-turn conversational memory
- Cyclic graph execution for follow-up questions
- LLM-powered decision making
- Structured state propagation across nodes

---
## Known Limitations

This section documents the current limitations of the system honestly.
Understanding the boundaries of an AI system is as important as understanding its capabilities.

### Clinical reliability

The triage classifications produced by this system have not been validated
against clinical guidelines or reviewed by medical professionals. The underlying
language model (Llama 3.3 70B via Groq) can produce plausible-sounding but
incorrect urgency assessments, particularly for edge cases, rare conditions,
or symptoms that require physical examination to evaluate.

The evaluation suite scores 100% on 10 fixed scenarios, but this measures
consistency against pre-defined expected outputs — not clinical accuracy.
A score of 100% on the eval suite does not mean the system is safe for
real medical use.

### Hallucination risk

Large language models can hallucinate. The triage decision node may:
- Assign an incorrect urgency level with high stated confidence
- Generate advice that sounds authoritative but is clinically wrong
- Miss critical symptoms that were mentioned but not weighted correctly
- Produce different urgency classifications for the same symptoms across runs
  (mitigated by setting temperature to 0.1, but not eliminated)

### No memory across sessions

Each session is independent. The system has no knowledge of a user's
medical history from previous sessions. A user who disclosed a heart
condition in one session will need to mention it again in a new session.

### No authentication beyond token scoping

The current auth implementation uses a UUID token stored in localStorage.
This is not cryptographically secured authentication. The token can be
copied between browsers, lost on browser data clear, or accessed by
other scripts running on the same origin. It provides session isolation
for demonstration purposes only — not production-grade user authentication.

### Single point of failure on Groq

The system has no fallback LLM provider. If the Groq API is unavailable,
all triage functionality fails. There is no retry logic, circuit breaker,
or degraded-mode response. For production use, a fallback provider
(e.g. OpenAI or Anthropic) should be configured.

### Free tier constraints

Both the backend and the PostgreSQL database are hosted on Render's free tier.
The backend service spins down after 15 minutes of inactivity, causing the
first request after idle to take 30-50 seconds. The free PostgreSQL instance
has a 1GB storage limit and will be deleted after 90 days of inactivity
on Render's free plan.

### What would be required for production use

- Clinical validation of triage outputs against established triage protocols
  such as the Manchester Triage System or ESI
- Review and sign-off by licensed medical professionals
- Robust authentication with server-side session management
- Fallback LLM provider with retry logic and circuit breaking
- Monitoring and alerting on triage outcome distributions
- A feedback mechanism for users to flag incorrect assessments
- Compliance review for any jurisdiction where medical software is regulated
  (e.g. CDSCO in India, FDA in the US, CE marking in Europe)

##  Project Structure

```
symptom-checker/
│
├── main.py                   # FastAPI entry point & app initialization
│
├── .env                      # Environment secrets (excluded from version control)
│
├── agent/
│   ├── state.py              # LangGraph shared state definition
│   ├── nodes.py              # assess_node and recommend_node logic
│   └── graph.py              # StateGraph assembly and compilation
│
├── routes/
│   ├── chat.py               # POST /chat  — submit a symptom message
│   └── history.py            # GET  /history/{session_id} — retrieve session log
│
├── db/
│   ├── database.py           # SQLAlchemy engine & session factory
│   ├── models.py             # ORM table definitions
│   └── crud.py               # Save and fetch operations
│
└── schemas/
    └── models.py             # Pydantic request/response models
```

---

##  Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| AI Agent | LangGraph + LangChain | Multi-node stateful graph |
| LLM | Groq · `llama-3.1-8b-instant` | Free-tier inference |
| API | FastAPI + Uvicorn | Async, auto-documented |
| Database | PostgreSQL + SQLAlchemy | Persistent session storage |
| Language | Python 3.11 | |

---

##  Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/symptom-checker.git
cd symptom-checker
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv \
            langgraph langchain langchain-groq
```

### 4. Set up PostgreSQL

```sql
CREATE DATABASE symptom_checker;
CREATE USER app_user WITH PASSWORD 'ur_password';
GRANT ALL PRIVILEGES ON DATABASE symptom_checker TO app_user;
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://username:password@localhost/symptom_checker
GROQ_API_KEY=your_groq_api_key
```

### 6. Start the server

```bash
uvicorn main:app --reload
```

### 7. Explore the API

Open the auto-generated interactive docs at:
**http://127.0.0.1:8000/docs**

---

##  Sample Conversation

The agent maintains session state across turns, enabling a coherent multi-message triage flow.

**Turn 1 — User reports symptoms**

```json
POST /chat
{
  "session_id": "session1",
  "message": "I have a headache and fever"
}
```

```json
{
  "reply": "How long have you had these symptoms and what is your temperature?",
  "stage": "assess"
}
```

**Turn 2 — User provides details → triage issued**

```json
POST /chat
{
  "session_id": "session1",
  "message": "Since yesterday, fever is 102F"
}
```

```json
{
  "reply": "Urgency: MEDIUM\nRest and stay hydrated. Take paracetamol for fever. If fever exceeds 103F or persists beyond 48hrs, see a doctor.",
  "stage": "done"
}
```

---



##  API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Submit a symptom message for a given session |
| `GET` | `/history/{session_id}` | Retrieve full conversation history for a session |

Request and response schemas are defined in `schemas/models.py` and browsable at `/docs`.

---

## 👩 Author

**Navya Chhoker**  
