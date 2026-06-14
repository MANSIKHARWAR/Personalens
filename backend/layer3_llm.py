"""
Layer 3 — LLM Reasoning (Groq)
Takes the outputs of Layer 1 (classical NLP) and Layer 2 (deep NLP)
plus the raw text, and asks Llama 3.3 70B (via Groq) to produce a
structured OCEAN personality profile and behavioral insights.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are PersonaLens, an expert AI psychologist specializing in
personality analysis from text (computational linguistics + psychometrics).

You will be given:
1. The original text written by a person.
2. Pre-computed NLP signals (sentiment, emotion, keywords, linguistic stats).

Your job is to synthesize ALL of this evidence into a personality profile.

Respond with ONLY valid JSON (no markdown fences, no preamble, no commentary)
matching EXACTLY this schema:

{
  "ocean": {
    "openness": {"score": <1-10 integer>, "label": "<Low|Below Average|Average|Above Average|High>", "evidence": "<1-2 sentence justification citing specific text/signals>"},
    "conscientiousness": {"score": <1-10>, "label": "<...>", "evidence": "<...>"},
    "extraversion": {"score": <1-10>, "label": "<...>", "evidence": "<...>"},
    "agreeableness": {"score": <1-10>, "label": "<...>", "evidence": "<...>"},
    "neuroticism": {"score": <1-10>, "label": "<...>", "evidence": "<...>"}
  },
  "communication_style": {
    "primary_style": "<analytical|expressive|driver|amiable>",
    "secondary_style": "<analytical|expressive|driver|amiable>",
    "tone": "<e.g. reserved, warm, assertive, formal, casual>",
    "description": "<2-3 sentence description of how this person communicates>"
  },
  "behavioral_insights": {
    "thinking_style": "<2-3 sentences>",
    "decision_making": "<2-3 sentences>",
    "stress_signals": "<2-3 sentences describing signs of stress or calm, or 'No significant stress signals detected.'>"
  },
  "strengths": ["<3-5 short strength phrases>"],
  "blind_spots": ["<2-4 short blind spot / growth area phrases>"],
  "career_fit": ["<3-5 job titles/roles that fit this profile>"],
  "summary": "<3-4 sentence overall personality summary written in second person ('You tend to...')>"
}

Rules:
- Scores must be integers 1-10 where 1 = very low trait, 10 = very high trait.
- Ground every claim in the provided text and NLP signals; avoid generic filler.
- If the text is very short, acknowledge lower confidence in the "summary" field but still provide your best estimate.
- Output ONLY the JSON object, nothing else.
"""


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


def build_user_prompt(text: str, layer1: dict, layer2: dict) -> str:
    return f"""=== ORIGINAL TEXT ===
{text}

=== LAYER 1: CLASSICAL NLP SIGNALS ===
VADER Sentiment: {json.dumps(layer1['vader_sentiment'])}
RoBERTa Sentiment: {json.dumps(layer1['roberta_sentiment'])}
Top Keywords: {json.dumps(layer1['keywords'])}
Linguistic Features: {json.dumps(layer1['linguistic_features'])}

=== LAYER 2: DEEP NLP SIGNALS ===
Emotion Analysis: {json.dumps(layer2['emotions'])}

Based on all of the above, generate the personality profile JSON as instructed."""


def run_layer3(text: str, layer1: dict, layer2: dict) -> dict:
    client = get_client()
    user_prompt = build_user_prompt(text, layer1, layer2)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content.strip()

    # Defensive cleanup in case the model wraps output in markdown fences
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM JSON response: {e}\nRaw: {content}")
