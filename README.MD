# 🤖 CRM Lead Scoring OpenEnv

> **Production-grade AI environment for CRM lead management and sales decision-making.**
> Built on OpenEnv standards. Deployable via Docker. HuggingFace Spaces ready.

---

## 📁 Project Structure

```
crm_lead_env/
│
├── api/
│   └── server.py              # FastAPI server — /reset, /step, /state, /actions, /health
│
├── environment/
│   ├── __init__.py
│   ├── environment.py         # Core OpenEnv: reset() · step() · state()
│   ├── nlp_analyzer.py        # NLP: intent · key_points · entities · complexity
│   ├── scam_detector.py       # Scam detection (medium + hard tasks)
│   └── reward_engine.py       # Shaped reward function (-1.0 to +1.0)
│
├── models/
│   └── schemas.py             # Pydantic typed models for all I/O
│
├── openenv.yaml               # OpenEnv environment specification
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🎯 Problem Statement

Modern CRM systems accumulate vast interaction histories — emails, calls, notes, meeting logs — making it difficult for sales teams to:

- Quickly judge lead quality and intent
- Identify optimal next actions
- Detect fraudulent or low-quality leads
- Prioritize follow-ups efficiently

This OpenEnv environment simulates that reality, enabling AI agents to **learn optimal CRM strategies** through reward-based feedback.

---

## 🏗️ Architecture

```
Agent (Client)
     │
     ▼ POST /reset  ─────────────────────────────────────────────
     │                   Generates synthetic lead
     │                   Initializes session, stage = "new"
     │◄──────────── { session_id, observation, reward=0 } ──────
     │
     ▼ POST /step   ─────────────────────────────────────────────
     │  { session_id, action, additional_input }
     │                   NLPAnalyzer  → intent, key_points, entities
     │                   ScamDetector → scam_detected, scam_reason
     │                   Lead score updated
     │                   Stage advanced
     │                   RewardEngine → shaped reward
     │◄──────────── { observation, reward, reward_score, analysis, NBA } ──
     │
     ▼ GET /state/{id} ──────────────────────────────────────────
                         Returns current session snapshot
```

---

## 🎮 Observation Space

| Field              | Type       | Values / Range                              |
|--------------------|------------|---------------------------------------------|
| `lead_id`          | string     | `LEAD-XXXXXXXX`                             |
| `score`            | float      | 0.0 – 1.0                                   |
| `priority`         | categorical| `low` · `medium` · `high` · `urgent`        |
| `stage`            | categorical| `new` → `contacted` → `engaged` → `hot` → `converted` / `dropped` |
| `interaction_count`| int        | 0 – max_steps                               |
| `history_summary`  | string     | Rolling interaction log                     |

---

## 🎯 Action Space (6 actions)

| Action             | Score Δ | Description                         |
|--------------------|---------|--------------------------------------|
| `call_lead`        | +0.12   | Direct phone call to the lead        |
| `send_email`       | +0.05   | Send personalized follow-up email    |
| `schedule_meeting` | +0.18   | Book discovery / demo meeting        |
| `ignore_lead`      | -0.05   | Take no action this step             |
| `mark_as_hot`      | +0.10   | Flag lead as urgent priority         |
| `mark_as_cold`     | -0.30   | Downgrade / drop the lead            |

---

## 💰 Reward System (-1.0 to +1.0)

```
reward =
  + correct_action_bonus  (up to +0.50, scaled by intent)
  + conversion_bonus      (+1.00 when stage = converted)
  + great_action_bonus    (+1.00 for mark_as_hot on hot lead)
  - wrong_action_penalty  (-0.50 for counterproductive actions)
  - ignore_active_penalty (-0.30 for ignoring engaged/hot leads)
  - scam_penalty          (-1.00 when scam is detected)
  - spam_repeat_penalty   (-0.30 when same action > 3×)
```

`reward_score` = cumulative episode reward across all steps.

---

## 🧪 Tasks

### 🟢 EASY — Cold Lead Classification
- Initial score: `0.05 – 0.20`
- Goal: Correctly classify and initiate outreach
- Extra fields: *none (base observation only)*

### 🟡 MEDIUM — Lead Engagement + Scam Detection
- Initial score: `0.20 – 0.40`
- Goal: Choose optimal actions while filtering fraudulent leads
- Extra fields: `scam_detected` · `scam_keywords` · `scam_reason`

### 🔴 HARD — Full Lifecycle Conversion Optimization
- Initial score: `0.50 – 0.70`
- Goal: Full sales cycle — qualify → engage → close
- Extra fields: All medium fields + `next_best_action` · `action_reason` · `recommended_schedule` · `risk_flag`

---

## 🔄 API Reference

### `POST /reset`
```json
Request:
{
  "task": "easy" | "medium" | "hard",
  "seed": 42
}

Response (hard example):
{
  "session_id": "3f7a9c12-...",
  "lead_id": "LEAD-9BA81C63",
  "task": "hard",
  "observation": {
    "lead_id": "LEAD-9BA81C63",
    "score": 0.5808,
    "priority": "high",
    "stage": "new",
    "interaction_count": 0,
    "history_summary": "Multiple demo requests, discussed budget..."
  },
  "reward": 0.0,
  "scam_detected": false,
  "scam_reason": "No suspicious patterns detected.",
  "next_best_action": "send_email",
  "recommended_schedule": "Within 48 hours",
  "risk_flag": "🟡 New lead — qualification needed."
}
```

---

### `POST /step`
```json
Request:
{
  "session_id": "3f7a9c12-...",
  "action": "schedule_meeting",
  "additional_input": "We are ready to sign the contract. Budget approved."
}

Response (hard example):
{
  "observation": { "score": 0.8148, "stage": "new", "priority": "urgent", ... },
  "lead_score": 0.8148,
  "reward": 0.10,
  "reward_score": 0.10,
  "step": 1,
  "done": false,
  "message": "Reasonable action but not optimal for stage 'new'.",
  "input_summary": "Lead provided 14-word message. Intent: High.",
  "task_complexity": "hard",
  "complexity_reason": "Keyword 'contract' signals hard-complexity interaction.",
  "analysis": {
    "key_points": ["We are ready to sign the contract this week", "Budget approved by the board"],
    "intent": "High",
    "entities": ["No structured entities detected."]
  },
  "prediction": "High probability of conversion. Recommend immediate close action.",
  "confidence_score": 0.92,
  "status": "active",
  "scam_detected": false,
  "scam_keywords": [],
  "scam_reason": "No suspicious patterns detected.",
  "next_best_action": "schedule_meeting",
  "action_reason": "Strong buying signal — lock in a meeting immediately.",
  "recommended_schedule": "Today or within 24 hours",
  "risk_flag": "🟡 New lead — qualification needed."
}
```

---

### `GET /state/{session_id}`
```json
{
  "session_id": "3f7a9c12-...",
  "task": "hard",
  "step": 2,
  "observation": { ... },
  "reward_score": 0.60,
  "done": false
}
```

---

## 🚨 Scam Detection (Medium + Hard)

Triggered by keywords: `urgent`, `free`, `money`, `lottery`, `win`, `prize`, `congratulations`, `wire transfer`, `bitcoin`, `inheritance`, `claim now`, etc.

When detected:
- `scam_detected: true`
- `scam_keywords: ["urgent", "free", "lottery", ...]`
- `scam_reason: "⚠️ Potential scam detected. High-risk keywords found: ..."`
- Stage → `dropped`
- Reward: `-1.0` (maximum penalty)

---

## 🤖 Next Best Action (Hard Only)

| Intent  | Action             | Reason                              |
|---------|--------------------|--------------------------------------|
| `Low`   | `send_email`       | Nurturing approach — lead not ready |
| `Medium`| `call_lead`        | Personal call can move them forward |
| `High`  | `schedule_meeting` | Strong buying signal — lock in now  |

---

## ⚙️ Setup & Running

### Local
```bash
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

### Docker
```bash
docker build -t crm-lead-scoring-openenv .
docker run -p 8000:8000 crm-lead-scoring-openenv
# Docs: http://localhost:8000/docs
```

### HuggingFace Spaces (Docker SDK)
1. Create a new Space → SDK: Docker
2. Push this repo
3. Space auto-deploys the FastAPI server
4. Live Swagger UI at your Space URL `/docs`

---

## 🧠 NLP Analysis Fields

Every `/step` response includes:

| Field               | Description                                      |
|---------------------|--------------------------------------------------|
| `input_summary`     | Auto-generated summary of `additional_input`     |
| `task_complexity`   | `easy` / `medium` / `hard` (inferred from text)  |
| `complexity_reason` | Explanation of complexity classification         |
| `analysis.key_points` | Up to 4 key sentences extracted                |
| `analysis.intent`   | `Low` / `Medium` / `High` buying intent          |
| `analysis.entities` | Detected emails, phones, companies, amounts      |
| `prediction`        | Conversion probability text                      |
| `confidence_score`  | 0.0 – 1.0 confidence in prediction               |

---

## 🔥 Key Innovations

- **Full OpenEnv compliance** — `reset()`, `step()`, `state()` with typed Pydantic models
- **Scam detection** — keyword-pattern system with risk scoring
- **NLP analysis** — intent classification + entity extraction (no external ML libs)
- **Shaped rewards** — `-1.0` to `+1.0` with 7 distinct reward components
- **3 task levels** — progressive complexity with different observation fields
- **Docker + HuggingFace ready** — zero-config deployment
- **Swagger UI** — fully documented interactive API at `/docs`