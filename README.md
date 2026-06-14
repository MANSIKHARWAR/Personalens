# PersonaLens 🔍

**AI-Powered Personality Analysis from Text**

Paste any text — tweets, chats, essays — and get a deep personality profile powered by a 3-layer NLP pipeline.

`Python` `FastAPI` `HuggingFace` `Groq`

---

## 🧠 What It Does

PersonaLens analyzes text and outputs:

- **Big Five (OCEAN)** personality traits with evidence
- **Communication style** — analytical, expressive, driver, amiable
- **Behavioral insights** — thinking style, decision making, stress signals
- **Strengths & blind spots**
- **Career fit suggestions**

---

## ⚙️ Architecture — 3-Layer NLP Pipeline

```
Raw Text Input
     │
     ▼
Layer 1 — Classical NLP (fast, free)
  ├── VADER Sentiment Analysis
  ├── RoBERTa Sentiment (cardiffnlp)
  ├── KeyBERT Keyword Extraction
  └── spaCy Linguistic Features
     │
     ▼
Layer 2 — Deep NLP (HuggingFace)
  ├── Emotion Detection (j-hartmann/emotion-english)
  └── Sentence Embeddings (all-MiniLM-L6-v2)
     │
     ▼
Layer 3 — LLM Reasoning (Groq)
  └── Llama 3.3 70B → OCEAN Profile + Insights
     │
     ▼
Structured JSON Response → Beautiful Frontend UI
```

---

## 🛠️ Tech Stack

| Layer        | Technology |
|--------------|------------|
| Backend      | FastAPI + Python |
| Sentiment    | VADER + cardiffnlp/twitter-roberta |
| Keywords     | KeyBERT |
| Linguistic   | spaCy |
| Emotions     | j-hartmann/emotion-english-distilroberta |
| Embeddings   | sentence-transformers/all-MiniLM-L6-v2 |
| LLM          | Groq API (Llama 3.3 70B) |
| Frontend     | HTML + CSS + Vanilla JS |

---

## 🚀 Getting Started

### 1. Clone / unzip the project

```bash
cd personalens
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> The first run will also download several HuggingFace models
> (~1–2 GB total). This happens automatically on first request.

### 4. Set up environment variables

Copy `.env.example` to `.env` inside the `backend/` folder (or project root)
and add your free Groq API key (get one at https://console.groq.com):

```bash
cp .env.example backend/.env
```

```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the backend

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`
(interactive docs at `http://localhost:8000/docs`).

### 6. Open the frontend

Open `frontend/index.html` directly in your browser
(or serve it with `python -m http.server` from the `frontend/` folder).

---

## 📊 API

### `POST /analyze`

**Request body:**

```json
{ "text": "Your text here, at least a few sentences..." }
```

**Response (abridged):**

```json
{
  "ocean": {
    "openness": {"score": 8, "label": "High", "evidence": "..."},
    "conscientiousness": {"score": 7, "label": "Above Average", "evidence": "..."},
    "extraversion": {"score": 2, "label": "Low", "evidence": "..."},
    "agreeableness": {"score": 5, "label": "Average", "evidence": "..."},
    "neuroticism": {"score": 1, "label": "Low", "evidence": "..."}
  },
  "communication_style": {
    "primary_style": "analytical",
    "secondary_style": "driver",
    "tone": "reserved",
    "description": "..."
  },
  "behavioral_insights": {
    "thinking_style": "...",
    "decision_making": "...",
    "stress_signals": "..."
  },
  "strengths": ["problem-solving", "technical expertise", "self-motivation"],
  "blind_spots": ["..."],
  "career_fit": ["software engineer", "data analyst", "researcher"],
  "summary": "...",
  "nlp_signals": { "...": "raw Layer 1 & 2 outputs" },
  "meta": { "word_count": 142, "timing_seconds": { "...": "..." } }
}
```

### `GET /health`

Simple health check.

---

## 📁 Project Structure

```
personalens/
├── backend/
│   ├── main.py              # FastAPI app & /analyze endpoint
│   ├── layer1_classical.py  # VADER, RoBERTa, KeyBERT, spaCy
│   ├── layer2_deep.py       # Emotion detection, embeddings
│   └── layer3_llm.py        # Groq Llama 3.3 70B reasoning
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── .env.example
```

---

## ⚠️ Notes

- The first analysis request will be slow as HuggingFace models download
  and load into memory. Subsequent requests are much faster.
- Minimum input: ~5 words. For reliable OCEAN scores, aim for at least
  a few sentences (50+ words).
- All model loading is lazy and cached as module-level singletons.
