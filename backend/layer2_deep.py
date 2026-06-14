"""
Layer 2 — Deep NLP (HuggingFace)
- Emotion Detection (j-hartmann/emotion-english-distilroberta-base)
- Sentence Embeddings (sentence-transformers/all-MiniLM-L6-v2)
"""

from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

_emotion_clf = None
_embedder = None


def get_emotion_clf():
    global _emotion_clf
    if _emotion_clf is None:
        _emotion_clf = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )
    return _emotion_clf


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def analyze_emotions(text: str) -> dict:
    clf = get_emotion_clf()
    truncated = text[:512]
    results = clf(truncated)[0]
    # Sort by score descending
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        "dominant_emotion": sorted_results[0]["label"],
        "emotion_distribution": {
            r["label"]: round(r["score"], 4) for r in sorted_results
        },
    }


def generate_embedding_summary(text: str) -> dict:
    """
    Generate sentence embedding and return summary stats.
    Full embedding vector is kept out of the API response (too large),
    but a compact fingerprint is returned for downstream use/debugging.
    """
    embedder = get_embedder()
    embedding = embedder.encode(text)
    return {
        "embedding_dim": int(embedding.shape[0]),
        "embedding_norm": round(float(np.linalg.norm(embedding)), 4),
        "embedding_preview": [round(float(x), 4) for x in embedding[:8]],
    }


def run_layer2(text: str) -> dict:
    """Run all Layer 2 deep NLP analyses."""
    return {
        "emotions": analyze_emotions(text),
        "embedding_summary": generate_embedding_summary(text),
    }
