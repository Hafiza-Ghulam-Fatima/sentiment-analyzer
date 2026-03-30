import { useState } from "react";

// ── Replace this with your actual Hugging Face Space URL after deploying ──
const API_URL = "https://hafizaghulamfatima55-sentiment-api.hf.space";

const EMOJIS = { positive: "😊", negative: "😠", neutral: "😐" };
const COLORS = {
  positive: { bg: "#00e5a015", border: "#00e5a040", text: "#00e5a0" },
  negative: { bg: "#ff4f6d15", border: "#ff4f6d40", text: "#ff4f6d" },
  neutral:  { bg: "#7b8cff15", border: "#7b8cff40", text: "#7b8cff" },
};

// Count real words in a string (same logic as the API guard)
function wordCount(str) {
  return str.trim().split(/\s+/).filter(Boolean).length;
}

export default function App() {
  const [tab, setTab] = useState("single");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Batch
  const [batchText, setBatchText] = useState("");
  const [batchResult, setBatchResult] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // Button is disabled until the user has typed at least 3 words
  const MIN_WORDS = 3;
  const enoughWords = wordCount(text) >= MIN_WORDS;

  async function analyze() {
    if (!enoughWords) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error((await res.json()).error || "Error");
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function batchAnalyze() {
    const lines = batchText.split("\n").map(t => t.trim()).filter(Boolean);
    if (!lines.length) return;
    setBatchLoading(true); setBatchResult(null);
    try {
      const res = await fetch(`${API_URL}/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines }),
      });
      if (!res.ok) throw new Error((await res.json()).error || "Error");
      setBatchResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setBatchLoading(false);
    }
  }

  const pred = result?.results?.lr;
  const wc   = wordCount(text);

  return (
    <div style={styles.body}>
      <div style={styles.grid} />
      <div style={styles.wrap}>

        {/* Header */}
        <div style={styles.header}>
          <div style={styles.badge}>✦ ML PROJECT</div>
          <h1 style={styles.h1}>Sentiment<span style={{ color: "#c8ff00" }}>AI</span></h1>
          <p style={styles.sub}>Detect emotion in text — Logistic Regression & Naive Bayes</p>
        </div>

        {/* Tabs */}
        <div style={styles.tabs}>
          {["single", "batch"].map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ ...styles.tabBtn, ...(tab === t ? styles.tabActive : {}) }}>
              {t === "single" ? "Single Analyze" : "Batch Analyze"}
            </button>
          ))}
        </div>

        {/* ── Single Tab ── */}
        {tab === "single" && (
          <div>
            <div style={styles.card}>
              <div style={styles.cardLabel}>// Enter Text</div>
              <textarea
                value={text}
                onChange={e => { setText(e.target.value); setResult(null); setError(""); }}
                placeholder="Enter at least 3 words — e.g. 'great product quality'..."
                maxLength={1000}
                style={styles.textarea}
                onKeyDown={e => e.ctrlKey && e.key === "Enter" && enoughWords && analyze()}
              />

              {/* Word count hint — shows live feedback */}
              <div style={styles.charRow}>
                <span style={{
                  ...styles.wordHint,
                  color: wc === 0 ? "#6b6b80" : wc < MIN_WORDS ? "#ff4f6d" : "#00e5a0"
                }}>
                  {wc === 0
                    ? "type at least 3 words"
                    : wc < MIN_WORDS
                      ? `${wc} word${wc !== 1 ? "s" : ""} — need ${MIN_WORDS - wc} more`
                      : `${wc} word${wc !== 1 ? "s" : ""} ✓`}
                </span>
                <span style={styles.charCount}>{text.length} / 1000</span>
              </div>

              <button
                onClick={analyze}
                disabled={loading || !enoughWords}
                style={{
                  ...styles.btn,
                  opacity: enoughWords ? 1 : 0.4,
                  cursor: enoughWords ? "pointer" : "not-allowed",
                }}
              >
                {loading ? "Analyzing..." : "Analyze Sentiment →"}
              </button>

              {error && <div style={styles.error}>⚠ {error}</div>}
            </div>

            {result && pred && (
              <div>
                {/* Low-confidence warning banner */}
                {result.uncertain && (
                  <div style={styles.uncertainBanner}>
                    ⚠ Low confidence — the model is unsure. Try adding more context
                    or detail for a more reliable result.
                  </div>
                )}

                {/* Verdict */}
                <div style={{
                  ...styles.verdict,
                  background: COLORS[pred.prediction].bg,
                  border: `1px solid ${COLORS[pred.prediction].border}`
                }}>
                  <span style={{ fontSize: 36 }}>{EMOJIS[pred.prediction]}</span>
                  <div>
                    <div style={styles.verdictLabel}>Primary Prediction</div>
                    <div style={{ ...styles.verdictText, color: COLORS[pred.prediction].text }}>
                      {pred.prediction.toUpperCase()}
                    </div>
                  </div>
                  <div style={styles.confBlock}>
                    <div style={{ ...styles.confNum, color: COLORS[pred.prediction].text }}>
                      {pred.confidence}%
                    </div>
                    <div style={styles.confLabel}>CONFIDENCE</div>
                  </div>
                </div>

                {/* Stats */}
                <div style={styles.statsRow}>
                  {[["Words", result.word_count], ["Characters", result.char_count], ["Models", 2]].map(([k, v]) => (
                    <div key={k} style={styles.statPill}>{k}: <span style={{ color: "#e8e8f0" }}>{v}</span></div>
                  ))}
                </div>

                {/* Model cards */}
                <div style={styles.modelLabel}>// Model Comparison</div>
                <div style={styles.modelsGrid}>
                  {Object.entries(result.results).map(([key, m]) => (
                    <div key={key} style={styles.modelCard}>
                      <div style={styles.modelName}>{m.model_name}</div>
                      <div style={{ ...styles.modelPred, color: COLORS[m.prediction].text }}>
                        {EMOJIS[m.prediction]} {m.prediction.toUpperCase()} · {m.confidence}%
                      </div>
                      {["positive", "negative", "neutral"].map(cls => (
                        <div key={cls} style={styles.probRow}>
                          <div style={styles.probLbl}>{cls.substring(0, 3).toUpperCase()}</div>
                          <div style={styles.probBarWrap}>
                            <div style={{
                              ...styles.probBar,
                              width: `${m.probabilities[cls] || 0}%`,
                              background: COLORS[cls].text
                            }} />
                          </div>
                          <div style={styles.probPct}>{m.probabilities[cls] || 0}%</div>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Batch Tab ── */}
        {tab === "batch" && (
          <div>
            <div style={styles.card}>
              <div style={styles.cardLabel}>// Enter Multiple Texts (one per line, max 20)</div>
              <textarea
                value={batchText}
                onChange={e => setBatchText(e.target.value)}
                placeholder={"Great product, loved it!\nThis is terrible, avoid.\nIt was okay I guess."}
                style={{ ...styles.textarea, minHeight: 180 }}
              />
              <button onClick={batchAnalyze} disabled={batchLoading} style={styles.btn}>
                {batchLoading ? "Processing..." : "Analyze Batch →"}
              </button>
            </div>

            {batchResult && (
              <div style={styles.card}>
                <div style={styles.cardLabel}>// Results</div>
                {batchResult.results.map((r, i) => {
                  const isUncertain = r.prediction === "uncertain";
                  const color = isUncertain
                    ? { bg: "#ffffff10", border: "#ffffff20", text: "#6b6b80" }
                    : COLORS[r.prediction];
                  return (
                    <div key={i} style={styles.batchItem}>
                      <div style={{
                        ...styles.batchDot,
                        background: isUncertain ? "#6b6b80" : color.text
                      }} />
                      <div style={styles.batchText}>{r.text}</div>
                      <div style={{
                        ...styles.batchPred,
                        background: color.bg,
                        color: color.text
                      }}>
                        {isUncertain ? "TOO SHORT" : r.prediction.toUpperCase()}
                      </div>
                      <div style={styles.batchConf}>
                        {isUncertain ? "—" : `${r.confidence}%`}
                      </div>
                    </div>
                  );
                })}
                <div style={styles.summaryGrid}>
                  {["positive", "negative", "neutral"].map(s => (
                    <div key={s} style={styles.summaryBox}>
                      <div style={{ ...styles.summaryNum, color: COLORS[s].text }}>
                        {batchResult.summary[s]}
                      </div>
                      <div style={styles.summaryLbl}>{s.toUpperCase()}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <footer style={styles.footer}>
          SENTIMENTAI · ML PROJECT · HOSTED ON HUGGING FACE
        </footer>
      </div>
    </div>
  );
}

// ── Styles ───────────────────────────────────────────────────────────────────
const styles = {
  body:         { background: "#0a0a0f", minHeight: "100vh", color: "#e8e8f0", fontFamily: "'Segoe UI', sans-serif", position: "relative", overflowX: "hidden" },
  grid:         { position: "fixed", inset: 0, backgroundImage: "linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px)", backgroundSize: "48px 48px", pointerEvents: "none", zIndex: 0 },
  wrap:         { position: "relative", zIndex: 1, maxWidth: 860, margin: "0 auto", padding: "48px 20px 80px" },
  header:       { textAlign: "center", marginBottom: 48 },
  badge:        { display: "inline-block", background: "rgba(200,255,0,0.1)", border: "1px solid rgba(200,255,0,0.2)", color: "#c8ff00", fontSize: 11, letterSpacing: 2, padding: "6px 14px", borderRadius: 100, marginBottom: 16 },
  h1:           { fontSize: "clamp(2.2rem, 6vw, 3.6rem)", fontWeight: 800, letterSpacing: -1, margin: 0 },
  sub:          { color: "#6b6b80", fontSize: "0.95rem", marginTop: 10 },
  tabs:         { display: "flex", gap: 4, background: "#111118", border: "1px solid #2a2a3a", borderRadius: 12, padding: 4, marginBottom: 24 },
  tabBtn:       { flex: 1, background: "none", border: "none", color: "#6b6b80", fontFamily: "inherit", fontSize: "0.85rem", fontWeight: 600, padding: "10px 16px", borderRadius: 8, cursor: "pointer" },
  tabActive:    { background: "#1a1a24", color: "#e8e8f0" },
  card:         { background: "#111118", border: "1px solid #2a2a3a", borderRadius: 16, padding: 28, marginBottom: 20 },
  cardLabel:    { fontFamily: "monospace", fontSize: 10, letterSpacing: 2, color: "#6b6b80", textTransform: "uppercase", marginBottom: 12 },
  textarea:     { width: "100%", background: "#1a1a24", border: "1px solid #2a2a3a", borderRadius: 10, color: "#e8e8f0", fontFamily: "monospace", fontSize: "0.88rem", lineHeight: 1.7, padding: "14px 16px", resize: "vertical", minHeight: 120, outline: "none", boxSizing: "border-box" },
  charRow:      { display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 6 },
  wordHint:     { fontFamily: "monospace", fontSize: 11, transition: "color 0.2s" },
  charCount:    { fontFamily: "monospace", fontSize: 11, color: "#6b6b80" },
  btn:          { width: "100%", background: "#c8ff00", color: "#0a0a0f", border: "none", borderRadius: 12, fontFamily: "inherit", fontSize: "1rem", fontWeight: 700, padding: "15px", marginTop: 14, transition: "opacity 0.2s" },
  error:        { background: "rgba(255,79,109,0.12)", border: "1px solid rgba(255,79,109,0.25)", borderRadius: 10, padding: "12px 16px", color: "#ff4f6d", fontSize: "0.85rem", marginTop: 12 },
  uncertainBanner: { background: "rgba(200,180,0,0.1)", border: "1px solid rgba(200,180,0,0.3)", borderRadius: 10, padding: "12px 16px", color: "#c8b800", fontSize: "0.85rem", marginBottom: 14 },
  verdict:      { display: "flex", alignItems: "center", gap: 16, borderRadius: 12, padding: "20px 24px", marginBottom: 18 },
  verdictLabel: { fontFamily: "monospace", fontSize: 10, letterSpacing: 2, opacity: 0.6, marginBottom: 4, textTransform: "uppercase" },
  verdictText:  { fontSize: "1.4rem", fontWeight: 800 },
  confBlock:    { marginLeft: "auto", textAlign: "right" },
  confNum:      { fontSize: "1.8rem", fontWeight: 800, fontFamily: "monospace" },
  confLabel:    { fontFamily: "monospace", fontSize: 10, letterSpacing: 1, color: "#6b6b80", textTransform: "uppercase" },
  statsRow:     { display: "flex", gap: 10, marginBottom: 18, flexWrap: "wrap" },
  statPill:     { background: "#1a1a24", border: "1px solid #2a2a3a", borderRadius: 8, padding: "8px 14px", fontFamily: "monospace", fontSize: "0.78rem", color: "#6b6b80" },
  modelLabel:   { fontFamily: "monospace", fontSize: 10, letterSpacing: 2, color: "#6b6b80", textTransform: "uppercase", marginBottom: 12 },
  modelsGrid:   { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 20 },
  modelCard:    { background: "#111118", border: "1px solid #2a2a3a", borderRadius: 12, padding: 18 },
  modelName:    { fontSize: "0.72rem", fontWeight: 700, letterSpacing: "0.08em", color: "#6b6b80", textTransform: "uppercase", marginBottom: 8 },
  modelPred:    { fontSize: "1rem", fontWeight: 700, marginBottom: 12 },
  probRow:      { display: "flex", alignItems: "center", gap: 8, marginBottom: 6 },
  probLbl:      { fontFamily: "monospace", fontSize: 10, color: "#6b6b80", width: 30, flexShrink: 0 },
  probBarWrap:  { flex: 1, height: 5, background: "#1a1a24", borderRadius: 100, overflow: "hidden" },
  probBar:      { height: "100%", borderRadius: 100, transition: "width 0.6s ease" },
  probPct:      { fontFamily: "monospace", fontSize: 10, color: "#6b6b80", width: 36, textAlign: "right" },
  batchItem:    { display: "flex", alignItems: "center", gap: 12, padding: "12px 0", borderBottom: "1px solid #2a2a3a" },
  batchDot:     { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },
  batchText:    { flex: 1, fontSize: "0.83rem", color: "#6b6b80", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
  batchPred:    { fontFamily: "monospace", fontSize: 10, fontWeight: 600, padding: "3px 10px", borderRadius: 100 },
  batchConf:    { fontFamily: "monospace", fontSize: 10, color: "#6b6b80", width: 40, textAlign: "right" },
  summaryGrid:  { display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginTop: 18 },
  summaryBox:   { background: "#1a1a24", borderRadius: 10, padding: 14, textAlign: "center" },
  summaryNum:   { fontSize: "1.8rem", fontWeight: 800 },
  summaryLbl:   { fontFamily: "monospace", fontSize: 10, letterSpacing: 1, color: "#6b6b80", textTransform: "uppercase", marginTop: 4 },
  footer:       { textAlign: "center", marginTop: 60, color: "#6b6b80", fontFamily: "monospace", fontSize: 11, letterSpacing: 1 },
};
// cache bust
