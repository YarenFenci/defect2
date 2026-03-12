import csv
import io
import re
from itertools import combinations
from typing import List, Tuple, Dict
 
import pandas as pd
import streamlit as st
 
# ─────────────────────────────────────────────────────────────
# Similarity engine
# ─────────────────────────────────────────────────────────────
 
STOP = {
    'a','an','the','in','on','at','to','of','for','and','or','is','are','was','were',
    'with','from','by','that','this','it','be','as','not','but','have','has','when',
    'i','ios','android','clone','via','bip','its','so','than','we','our',
}
 
def clean_summary(text: str) -> str:
    """Strip platform prefixes, CLONE tag, punctuation, lowercase."""
    t = text.lower().strip()
    # Remove CLONE - prefix
    t = re.sub(r'clone\s*[-–]?\s*', '', t, flags=re.IGNORECASE)
    # Remove leading platform prefix:  "iOS | ...",  "Android - ..."
    t = re.sub(
        r'^\s*(ios|iphone|android|samsung|huawei|redmi|xiaomi|oppo|realme|vivo)\s*[|–\-]\s*',
        '', t, flags=re.IGNORECASE
    )
    # Normalize separators
    t = re.sub(r'[|–\-/]', ' ', t)
    # Remove non-word chars
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t
 
def meaningful_tokens(text: str, n: int = None) -> List[str]:
    toks = [w for w in clean_summary(text).split() if w not in STOP and len(w) > 2]
    return toks[:n] if n else toks
 
def jaccard_sim(a: str, b: str) -> float:
    ta, tb = set(meaningful_tokens(a)), set(meaningful_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
 
def prefix_sim(a: str, b: str, n: int = 4) -> float:
    """How much do the first N meaningful tokens overlap?"""
    ta = meaningful_tokens(a, n)
    tb = meaningful_tokens(b, n)
    if not ta or not tb:
        return 0.0
    return len(set(ta) & set(tb)) / max(len(ta), len(tb))
 
def combined_score(a: str, b: str) -> Tuple[float, str]:
    """
    Returns (score, match_type).
    match_type: 'exact' | 'prefix' | 'similar'
 
    Logic:
    - Jaccard = 1.0                    → exact duplicate
    - prefix_sim(n=4) >= 0.75          → same bug area, different symptom description
    - jaccard >= threshold             → generally similar
    """
    j = jaccard_sim(a, b)
    p = prefix_sim(a, b, n=4)
 
    if j >= 0.99:
        return 1.0, "exact"
    if p >= 0.75:
        # Boost the score so it ranks high even if tail words differ
        score = max(j, p * 0.90)
        return score, "prefix"
    score = j
    return score, "similar"
 
 
# ─────────────────────────────────────────────────────────────
# CSV loader
# ─────────────────────────────────────────────────────────────
 
def load_csv(file) -> Tuple[List[Dict], str]:
    """Try semicolon then comma delimiter."""
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
 
    for delim in (";", ",", "\t"):
        reader = csv.DictReader(io.StringIO(content), delimiter=delim)
        rows = list(reader)
        if rows and len(rows[0]) > 2:
            return rows, delim
    return [], ";"
 
 
def find_duplicates(rows: List[Dict], threshold: float) -> List[Dict]:
    """
    Compare all pairs. Return list of duplicate candidates sorted by score desc.
    """
    results = []
    for i, j in combinations(range(len(rows)), 2):
        r1, r2 = rows[i], rows[j]
        s1 = r1.get("Summary", "").strip()
        s2 = r2.get("Summary", "").strip()
        if not s1 or not s2:
            continue
 
        score, match_type = combined_score(s1, s2)
        if score >= threshold:
            results.append({
                "score":      round(score, 2),
                "match_type": match_type,
                "key_1":      r1.get("Issue key", ""),
                "key_2":      r2.get("Issue key", ""),
                "summary_1":  s1,
                "summary_2":  s2,
                "priority_1": r1.get("Priority", ""),
                "priority_2": r2.get("Priority", ""),
                "repo_1":     r1.get("Custom field (Test Repository Path)", ""),
                "repo_2":     r2.get("Custom field (Test Repository Path)", ""),
            })
 
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
 
 
# ─────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────
 
MATCH_META = {
    "exact":   {"color": "#E53935", "label": "Exact Duplicate",     "icon": "🔴"},
    "prefix":  {"color": "#FB8C00", "label": "Same Bug Area",        "icon": "🟠"},
    "similar": {"color": "#1E88E5", "label": "Similar",              "icon": "🔵"},
}
 
PRIORITY_COLORS = {
    "gating":  "#E53935",
    "high":    "#FB8C00",
    "medium":  "#1E88E5",
    "low":     "#43A047",
}
 
def priority_badge(p: str) -> str:
    col = PRIORITY_COLORS.get(p.lower(), "#78909C")
    return (
        f'<span style="background:{col};color:#fff;padding:1px 7px;'
        f'border-radius:4px;font-size:0.7rem;font-family:monospace;'
        f'font-weight:700">{p.upper()}</span>'
    )
 
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: #0A0F1E !important;
        color: #E8EEFF !important;
    }
    .block-container { padding: 1.5rem 2rem !important; max-width: 1100px !important; }
 
    .dd-header {
        text-align: center; padding: 1.2rem 0 0.6rem;
    }
    .dd-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.6rem; font-weight: 700; letter-spacing: 0.04em;
        color: #E8EEFF;
    }
    .dd-title span { color: #4FC3F7; }
    .dd-subtitle {
        font-size: 0.8rem; color: #4A5A7A; margin-top: 0.3rem;
        font-family: 'JetBrains Mono', monospace; letter-spacing: 0.06em;
    }
    .dd-divider {
        height: 1px; background: linear-gradient(90deg,transparent,#1E2761,transparent);
        margin: 0.8rem 0 1.2rem;
    }
 
    .pair-card {
        background: #0D1321;
        border: 1px solid #1E2761;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .pair-header {
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 0.6rem; flex-wrap: wrap;
    }
    .pair-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem; font-weight: 700;
    }
    .match-badge {
        padding: 2px 10px; border-radius: 10px;
        font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; font-weight: 600;
    }
    .issue-row {
        display: flex; align-items: flex-start; gap: 10px;
        padding: 0.5rem 0.7rem; border-radius: 6px;
        background: #111827; border: 1px solid #1A2340;
        margin-bottom: 5px;
    }
    .issue-key {
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        color: #4FC3F7; min-width: 100px; font-weight: 600; flex-shrink: 0;
    }
    .issue-summary {
        font-size: 0.82rem; color: #A0B0CC; flex: 1; line-height: 1.4;
    }
    .issue-summary em {
        background: #2A1A00; color: #FFB74D;
        padding: 0 3px; border-radius: 3px; font-style: normal;
    }
    .diff-reason {
        font-size: 0.72rem; color: #4A6A8A; margin-top: 0.4rem;
        font-style: italic; padding-left: 0.5rem;
        border-left: 2px solid #1E2761;
    }
 
    .stat-box {
        background: #0D1321; border: 1px solid #1E2761; border-radius: 8px;
        padding: 0.8rem 1rem; text-align: center;
    }
    .stat-num {
        font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
        font-weight: 700; color: #4FC3F7;
    }
    .stat-label { font-size: 0.72rem; color: #4A5A7A; margin-top: 2px; }
 
    .upload-zone {
        border: 2px dashed #1E2761; border-radius: 12px;
        padding: 2rem; text-align: center; color: #3A4A6B;
        font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
    }
 
    div[data-testid="stFileUploader"] {
        border: 1px solid #1E2761 !important;
        border-radius: 8px !important;
        background: #0D1321 !important;
    }
    .stSlider > div > div > div { background: #4FC3F7 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)
 
 
def highlight_diff(s1: str, s2: str) -> Tuple[str, str]:
    """Highlight tokens that differ between the two summaries."""
    t1 = set(meaningful_tokens(s1))
    t2 = set(meaningful_tokens(s2))
    only_in_1 = t1 - t2
    only_in_2 = t2 - t1
 
    def mark(text, unique_tokens):
        words = text.split()
        out = []
        for w in words:
            if w.lower().strip(".,;:!?") in unique_tokens:
                out.append(f"<em>{w}</em>")
            else:
                out.append(w)
        return " ".join(out)
 
    return mark(s1, only_in_1), mark(s2, only_in_2)
 
 
def render_pair(pair: Dict, idx: int):
    m = MATCH_META[pair["match_type"]]
    score_pct = int(pair["score"] * 100)
 
    h1, h2 = highlight_diff(pair["summary_1"], pair["summary_2"])
    p1_badge = priority_badge(pair["priority_1"]) if pair["priority_1"] else ""
    p2_badge = priority_badge(pair["priority_2"]) if pair["priority_2"] else ""
 
    # Explain WHY they match
    if pair["match_type"] == "exact":
        reason = "Summaries are identical — very likely the same defect reported twice."
    elif pair["match_type"] == "prefix":
        shared = set(meaningful_tokens(pair["summary_1"], 4)) & set(meaningful_tokens(pair["summary_2"], 4))
        reason = f"Same topic area (shared prefix tokens: {', '.join(shared)}). Different symptom descriptions — may be variants of the same root cause."
    else:
        shared = set(meaningful_tokens(pair["summary_1"])) & set(meaningful_tokens(pair["summary_2"]))
        reason = f"High token overlap ({score_pct}%). Shared terms: {', '.join(list(shared)[:6])}."
 
    st.markdown(f"""
    <div class="pair-card">
        <div class="pair-header">
            <span class="pair-score" style="color:{m['color']}">{m['icon']} {score_pct}%</span>
            <span class="match-badge" style="background:{m['color']}22;border:1px solid {m['color']}55;color:{m['color']}">
                {m['label']}
            </span>
            <span style="font-size:0.7rem;color:#4A5A7A;font-family:monospace">#{idx}</span>
        </div>
        <div class="issue-row">
            <span class="issue-key">{pair['key_1']}</span>
            <span class="issue-summary">{h1}</span>
            <span style="margin-left:auto;flex-shrink:0">{p1_badge}</span>
        </div>
        <div class="issue-row">
            <span class="issue-key">{pair['key_2']}</span>
            <span class="issue-summary">{h2}</span>
            <span style="margin-left:auto;flex-shrink:0">{p2_badge}</span>
        </div>
        <div class="diff-reason">💡 {reason}</div>
    </div>
    """, unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
 
def main():
    st.set_page_config(
        page_title="Detect Defect — Duplicate Finder",
        layout="wide",
        page_icon="🔍",
        initial_sidebar_state="collapsed",
    )
    inject_css()
 
    st.markdown("""
    <div class="dd-header">
        <div class="dd-title">Detect <span>Defect</span></div>
        <div class="dd-subtitle">
            BiP QA · Duplicate &amp; Similar Issue Detector &nbsp;·&nbsp;
            Exact · Same Bug Area · Similar
        </div>
    </div>
    <div class="dd-divider"></div>
    """, unsafe_allow_html=True)
 
    # ── Controls ──────────────────────────────────────────────
    col_up, col_thr, col_flt = st.columns([2, 1, 1], gap="large")
 
    with col_up:
        st.markdown('<div style="font-size:0.72rem;color:#4A5A7A;font-family:monospace;'
                    'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px">'
                    'Upload CSV (Jira Export)</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "CSV", type=["csv"], label_visibility="collapsed", key="dd_upload"
        )
 
    with col_thr:
        st.markdown('<div style="font-size:0.72rem;color:#4A5A7A;font-family:monospace;'
                    'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px">'
                    'Similarity Threshold</div>', unsafe_allow_html=True)
        threshold = st.slider(
            "Threshold", 0.30, 0.95, 0.45, 0.05,
            label_visibility="collapsed", key="dd_threshold",
            help="Lower = more pairs found (including weaker matches). 0.45 recommended."
        )
        st.markdown(
            f'<div style="font-size:0.72rem;color:#4FC3F7;font-family:monospace;'
            f'text-align:center">{int(threshold*100)}% minimum similarity</div>',
            unsafe_allow_html=True,
        )
 
    with col_flt:
        st.markdown('<div style="font-size:0.72rem;color:#4A5A7A;font-family:monospace;'
                    'letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px">'
                    'Filter by Match Type</div>', unsafe_allow_html=True)
        show_exact   = st.checkbox("🔴 Exact Duplicate", value=True,  key="dd_exact")
        show_prefix  = st.checkbox("🟠 Same Bug Area",   value=True,  key="dd_prefix")
        show_similar = st.checkbox("🔵 Similar",         value=True,  key="dd_similar")
 
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
 
    # ── Analysis ──────────────────────────────────────────────
    if not uploaded:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2rem;margin-bottom:0.5rem">📂</div>
            <div>Jira CSV export yükle</div>
            <div style="font-size:0.7rem;margin-top:0.3rem;color:#2A3A5C">
                Semicolon (;) veya comma (,) delimiter destekleniyor
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
 
    rows, delim = load_csv(uploaded)
    if not rows:
        st.error("CSV okunamadı veya boş.")
        return
 
    # ── Stats bar ─────────────────────────────────────────────
    duplicates = find_duplicates(rows, threshold)
 
    # Apply filter
    active_types = set()
    if show_exact:   active_types.add("exact")
    if show_prefix:  active_types.add("prefix")
    if show_similar: active_types.add("similar")
    filtered = [d for d in duplicates if d["match_type"] in active_types]
 
    n_exact   = sum(1 for d in duplicates if d["match_type"] == "exact")
    n_prefix  = sum(1 for d in duplicates if d["match_type"] == "prefix")
    n_similar = sum(1 for d in duplicates if d["match_type"] == "similar")
 
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, num, label, color in [
        (c1, len(rows),    "Total Issues",     "#4FC3F7"),
        (c2, len(filtered),"Pairs Found",      "#E8EEFF"),
        (c3, n_exact,      "Exact Duplicates", "#E53935"),
        (c4, n_prefix,     "Same Bug Area",    "#FB8C00"),
        (c5, n_similar,    "Similar",          "#1E88E5"),
    ]:
        col.markdown(
            f'<div class="stat-box">'
            f'<div class="stat-num" style="color:{color}">{num}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
 
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
 
    # ── Results ───────────────────────────────────────────────
    if not filtered:
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#3A4A6B;'
            'font-family:monospace;font-size:0.85rem">'
            '✅ No duplicate pairs found at this threshold.<br>'
            '<span style="font-size:0.72rem">Try lowering the threshold.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for i, pair in enumerate(filtered, 1):
            render_pair(pair, i)
 
        # Export
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        export_df = pd.DataFrame(filtered)[[
            "score", "match_type", "key_1", "summary_1", "priority_1",
            "key_2", "summary_2", "priority_2"
        ]]
        export_df.columns = [
            "Score", "Match Type", "Issue Key 1", "Summary 1", "Priority 1",
            "Issue Key 2", "Summary 2", "Priority 2"
        ]
        st.download_button(
            "⬇ Export duplicate pairs as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="duplicate_pairs.csv",
            mime="text/csv",
            key=f"dd_export_{len(filtered)}",
        )
 
 
if __name__ == "__main__":
    main()
