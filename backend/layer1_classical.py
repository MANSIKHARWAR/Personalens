"""
Layer 1 — Classical NLP (fast, free)
- VADER Sentiment Analysis
- RoBERTa Sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest)
- KeyBERT Keyword Extraction
- spaCy Linguistic Features
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from keybert import KeyBERT
import spacy

# --- Lazy-loaded singletons (loaded once at startup) ---
_vader = None
_roberta = None
_keybert = None
_nlp = None


def get_vader():
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def get_roberta():
    global _roberta
    if _roberta is None:
        _roberta = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
    return _roberta


def get_keybert():
    global _keybert
    if _keybert is None:
        # Reuse a lightweight sentence-transformer backend
        _keybert = KeyBERT(model="all-MiniLM-L6-v2")
    return _keybert


def get_spacy():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to blank English pipeline if model isn't downloaded
            _nlp = spacy.blank("en")
    return _nlp


def analyze_vader(text: str) -> dict:
    scores = get_vader().polarity_scores(text)
    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "neutral": scores["neu"],
        "negative": scores["neg"],
    }


def analyze_roberta(text: str) -> dict:
    clf = get_roberta()
    # Truncate to avoid token-limit errors on long inputs
    truncated = text[:512]
    result = clf(truncated)[0]
    return {"label": result["label"], "score": round(result["score"], 4)}


def extract_keywords(text: str, top_n: int = 10) -> list:
    kw_model = get_keybert()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n,
    )
    return [{"keyword": k, "relevance": round(score, 4)} for k, score in keywords]


def analyze_linguistic_features(text: str) -> dict:
    nlp = get_spacy()
    doc = nlp(text)

    sentences = list(doc.sents) if doc.has_annotation("SENT_START") else [doc]
    words = [t for t in doc if not t.is_space]
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    avg_sentence_length = round(word_count / sentence_count, 2)
    avg_word_length = (
        round(sum(len(t.text) for t in words) / word_count, 2) if word_count else 0
    )

    pos_counts = {}
    pronoun_counts = {"first_person": 0, "second_person": 0, "third_person": 0}

    first_person = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
    second_person = {"you", "your", "yours"}
    third_person = {"he", "she", "they", "him", "her", "them", "his", "their", "theirs"}

    for token in words:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        lower = token.text.lower()
        if lower in first_person:
            pronoun_counts["first_person"] += 1
        elif lower in second_person:
            pronoun_counts["second_person"] += 1
        elif lower in third_person:
            pronoun_counts["third_person"] += 1

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "pos_distribution": pos_counts,
        "pronoun_usage": pronoun_counts,
    }


def run_layer1(text: str) -> dict:
    """Run all Layer 1 classical NLP analyses."""
    return {
        "vader_sentiment": analyze_vader(text),
        "roberta_sentiment": analyze_roberta(text),
        "keywords": extract_keywords(text),
        "linguistic_features": analyze_linguistic_features(text),
    }
