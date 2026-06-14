// ============================================
// PersonaLens — Frontend Logic
// ============================================

const API_BASE = "http://localhost:8000";

const textInput = document.getElementById("text-input");
const wordCountEl = document.getElementById("word-count");
const analyzeBtn = document.getElementById("analyze-btn");
const statusLine = document.getElementById("status-line");
const statusText = document.getElementById("status-text");
const errorBox = document.getElementById("error-box");
const resultsSection = document.getElementById("results");

const OCEAN_LABELS = {
  openness: "Openness",
  conscientiousness: "Conscientiousness",
  extraversion: "Extraversion",
  agreeableness: "Agreeableness",
  neuroticism: "Neuroticism",
};

// --- Word count ---
textInput.addEventListener("input", () => {
  const words = textInput.value.trim().split(/\s+/).filter(Boolean).length;
  wordCountEl.textContent = `${words} word${words === 1 ? "" : "s"}`;
});

// --- Status message cycling while waiting ---
const STATUS_MESSAGES = [
  "Running layer 1 — classical NLP (sentiment, keywords, linguistics)…",
  "Running layer 2 — deep NLP (emotion detection, embeddings)…",
  "Running layer 3 — LLM reasoning (OCEAN profile + insights)…",
  "Almost there — compiling your personality profile…",
];

let statusInterval = null;

function startStatusCycle() {
  let i = 0;
  statusText.textContent = STATUS_MESSAGES[0];
  statusInterval = setInterval(() => {
    i = (i + 1) % STATUS_MESSAGES.length;
    statusText.textContent = STATUS_MESSAGES[i];
  }, 2200);
}

function stopStatusCycle() {
  clearInterval(statusInterval);
  statusInterval = null;
}

// --- Main analyze action ---
analyzeBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();

  errorBox.hidden = true;
  resultsSection.hidden = true;

  if (text.split(/\s+/).filter(Boolean).length < 5) {
    showError("Please enter at least a few sentences for an accurate analysis.");
    return;
  }

  analyzeBtn.disabled = true;
  statusLine.hidden = false;
  startStatusCycle();

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Something went wrong during analysis.");
    }

    renderResults(data);
  } catch (err) {
    showError(
      `${err.message} — make sure the backend is running at ${API_BASE} and GROQ_API_KEY is set.`
    );
  } finally {
    analyzeBtn.disabled = false;
    statusLine.hidden = true;
    stopStatusCycle();
  }
});

function showError(message) {
  errorBox.hidden = false;
  errorBox.textContent = message;
}

// --- Render ---
function renderResults(data) {
  // Summary
  document.getElementById("summary-text").textContent = data.summary;

  // OCEAN
  const oceanGrid = document.getElementById("ocean-grid");
  oceanGrid.innerHTML = "";
  Object.entries(data.ocean).forEach(([key, val]) => {
    const row = document.createElement("div");
    row.className = "trait-row";
    row.innerHTML = `
      <div class="trait-name-block">
        <span class="trait-name">${OCEAN_LABELS[key] || key}</span>
        <span class="trait-label">${val.label || ""}</span>
      </div>
      <div class="trait-track">
        <div class="trait-fill" data-width="${val.score * 10}%"></div>
      </div>
      <div class="trait-score">${val.score}/10</div>
    `;
    oceanGrid.appendChild(row);

    const evidence = document.createElement("div");
    evidence.className = "trait-evidence";
    evidence.textContent = val.evidence || "";
    oceanGrid.appendChild(evidence);
  });

  // Animate bars after insertion
  requestAnimationFrame(() => {
    document.querySelectorAll(".trait-fill").forEach((el) => {
      el.style.width = el.dataset.width;
    });
  });

  // Communication style
  const comm = data.communication_style;
  const commStyles = document.getElementById("comm-styles");
  commStyles.innerHTML = `
    <span class="comm-chip primary">${comm.primary_style}</span>
    ${comm.secondary_style ? `<span class="comm-chip">${comm.secondary_style}</span>` : ""}
    ${comm.tone ? `<span class="comm-chip tone">${comm.tone}</span>` : ""}
  `;
  document.getElementById("comm-desc").textContent = comm.description || "";

  // Behavioral insights
  const behavior = data.behavioral_insights;
  const behaviorList = document.getElementById("behavior-list");
  behaviorList.innerHTML = `
    <dt>Thinking Style</dt>
    <dd>${behavior.thinking_style}</dd>
    <dt>Decision Making</dt>
    <dd>${behavior.decision_making}</dd>
    <dt>Stress Signals</dt>
    <dd>${behavior.stress_signals}</dd>
  `;

  // Strengths / blind spots
  const strengthsList = document.getElementById("strengths-list");
  strengthsList.innerHTML = data.strengths.map((s) => `<li>${s}</li>`).join("");

  const blindSpotsList = document.getElementById("blindspots-list");
  blindSpotsList.innerHTML = data.blind_spots.map((s) => `<li>${s}</li>`).join("");

  // Career fit
  const careerTags = document.getElementById("career-tags");
  careerTags.innerHTML = data.career_fit
    .map((c) => `<span class="career-tag">${c}</span>`)
    .join("");

  // Raw signals
  const signals = data.nlp_signals;
  document.getElementById("signal-vader").textContent = JSON.stringify(
    signals.vader_sentiment,
    null,
    2
  );
  document.getElementById("signal-roberta").textContent = JSON.stringify(
    signals.roberta_sentiment,
    null,
    2
  );
  document.getElementById("signal-emotion").textContent = JSON.stringify(
    signals.emotions,
    null,
    2
  );
  document.getElementById("signal-linguistic").textContent = JSON.stringify(
    signals.linguistic_features,
    null,
    2
  );

  const keywordTags = document.getElementById("signal-keywords");
  keywordTags.innerHTML = signals.keywords
    .map((k) => `<span class="keyword-tag">${k.keyword} · ${k.relevance}</span>`)
    .join("");

  // Timing
  const t = data.meta.timing_seconds;
  document.getElementById("timing-note").textContent =
    `Layer 1: ${t.layer1_classical_nlp}s · Layer 2: ${t.layer2_deep_nlp}s · ` +
    `Layer 3: ${t.layer3_llm_reasoning}s · Total: ${t.total}s · ` +
    `${data.meta.word_count} words analyzed`;

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// --- Toggle raw signals panel ---
document.getElementById("toggle-signals").addEventListener("click", () => {
  const body = document.getElementById("signals-body");
  const icon = document.querySelector(".toggle-icon");
  const isHidden = body.hidden;
  body.hidden = !isHidden;
  icon.textContent = isHidden ? "−" : "+";
});
