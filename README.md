# 🩺 AI Medical Triage Agent

### LangGraph · FastAPI · PostgreSQL · Groq LLM

> A conversational AI agent that triages patient symptoms through multi-turn dialogue, built with LangGraph, FastAPI, and PostgreSQL.

---

## 📌 Overview

This project implements a lightweight medical triage assistant that engages users in a structured conversation to assess symptom severity and recommend appropriate next steps. The agent intelligently decides when it has gathered enough context to issue a triage decision — and when it needs to ask one more targeted question first.

**Key design principle:** The agent asks a single clarification per interaction and then reassesses whether enough information has been gathered to make a triage recommendation.

---

## ✨ Highlights

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

## 🧠 How It Works

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

## 🏗️ Architecture

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

## 🔄 LangGraph Concepts Demonstrated

This project showcases several core LangGraph concepts:

- Stateful workflow management
- Conditional routing between nodes
- Multi-turn conversational memory
- Cyclic graph execution for follow-up questions
- LLM-powered decision making
- Structured state propagation across nodes

---

## 🗂️ Project Structure

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

## ⚙️ Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| AI Agent | LangGraph + LangChain | Multi-node stateful graph |
| LLM | Groq · `llama-3.1-8b-instant` | Free-tier inference |
| API | FastAPI + Uvicorn | Async, auto-documented |
| Database | PostgreSQL + SQLAlchemy | Persistent session storage |
| Language | Python 3.11 | |

---

## 🚀 Setup & Run

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

## 📬 Sample Conversation

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



## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Submit a symptom message for a given session |
| `GET` | `/history/{session_id}` | Retrieve full conversation history for a session |

Request and response schemas are defined in `schemas/models.py` and browsable at `/docs`.

---

## 👩‍💻 Author

**Navya Chhoker**  
B.Tech CSE (Data Science) · Gautam Buddha University  
Research interests: AI for healthcare applications