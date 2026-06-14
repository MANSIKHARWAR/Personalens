"""
PersonaLens — FastAPI Backend
3-Layer NLP Pipeline:
  Layer 1: Classical NLP (VADER, RoBERTa sentiment, KeyBERT, spaCy)
  Layer 2: Deep NLP (Emotion detection, Sentence embeddings)
  Layer 3: LLM Reasoning (Groq Llama 3.3 70B) -> OCEAN profile + insights
"""

import os
import time
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import layer1_classical as l1
import layer2_deep as l2
import layer3_llm as l3

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("personalens")

app = FastAPI(
    title="PersonaLens API",
    description="AI-Powered Personality Analysis from Text — 3-Layer NLP Pipeline",
    version="1.0.0",
)

# Allow the static frontend (opened via file:// or any local server) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Text to analyze (min 20 characters)")


class AnalyzeResponse(BaseModel):
    ocean: dict
    communication_style: dict
    behavioral_insights: dict
    strengths: list
    blind_spots: list
    career_fit: list
    summary: str
    nlp_signals: dict
    meta: dict


@app.get("/")
def root():
    return {
        "name": "PersonaLens API",
        "status": "running",
        "endpoints": {
            "analyze": "POST /analyze",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    text = req.text.strip()

    if len(text.split()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Please provide more text (at least a few sentences) for accurate analysis.",
        )

    start = time.time()

    # --- Layer 1: Classical NLP ---
    try:
        t1 = time.time()
        layer1_result = l1.run_layer1(text)
        layer1_time = round(time.time() - t1, 3)
        logger.info(f"Layer 1 complete in {layer1_time}s")
    except Exception as e:
        logger.exception("Layer 1 failed")
        raise HTTPException(status_code=500, detail=f"Layer 1 (classical NLP) failed: {e}")

    # --- Layer 2: Deep NLP ---
    try:
        t2 = time.time()
        layer2_result = l2.run_layer2(text)
        layer2_time = round(time.time() - t2, 3)
        logger.info(f"Layer 2 complete in {layer2_time}s")
    except Exception as e:
        logger.exception("Layer 2 failed")
        raise HTTPException(status_code=500, detail=f"Layer 2 (deep NLP) failed: {e}")

    # --- Layer 3: LLM Reasoning ---
    try:
        t3 = time.time()
        layer3_result = l3.run_layer3(text, layer1_result, layer2_result)
        layer3_time = round(time.time() - t3, 3)
        logger.info(f"Layer 3 complete in {layer3_time}s")
    except RuntimeError as e:
        logger.exception("Layer 3 failed")
        raise HTTPException(status_code=502, detail=f"Layer 3 (LLM reasoning) failed: {e}")
    except Exception as e:
        logger.exception("Layer 3 failed")
        raise HTTPException(status_code=500, detail=f"Layer 3 (LLM reasoning) failed: {e}")

    total_time = round(time.time() - start, 3)

    return {
        "ocean": layer3_result["ocean"],
        "communication_style": layer3_result["communication_style"],
        "behavioral_insights": layer3_result["behavioral_insights"],
        "strengths": layer3_result["strengths"],
        "blind_spots": layer3_result["blind_spots"],
        "career_fit": layer3_result["career_fit"],
        "summary": layer3_result["summary"],
        "nlp_signals": {
            "vader_sentiment": layer1_result["vader_sentiment"],
            "roberta_sentiment": layer1_result["roberta_sentiment"],
            "keywords": layer1_result["keywords"],
            "linguistic_features": layer1_result["linguistic_features"],
            "emotions": layer2_result["emotions"],
            "embedding_summary": layer2_result["embedding_summary"],
        },
        "meta": {
            "word_count": layer1_result["linguistic_features"]["word_count"],
            "timing_seconds": {
                "layer1_classical_nlp": layer1_time,
                "layer2_deep_nlp": layer2_time,
                "layer3_llm_reasoning": layer3_time,
                "total": total_time,
            },
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
