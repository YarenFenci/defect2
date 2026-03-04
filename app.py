import re
from collections import deque
from io import StringIO
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
TOP_K_NEIGHBORS              = 50
TFIDF_CANDIDATE_FLOOR        = 0.50   # broad pre-filter only, semantic decides final

SEMANTIC_DUPLICATE_THRESHOLD = 0.87   # sentence cosine >= this → confirmed duplicate
MIN_SIMILARITY_DISPLAY       = 0.75   # absolute floor, nothing below is shown

FLOW_GUARD = True   # both issues must share >= 1 user flow

SENTENCE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ─────────────────────────────────────────
# User-flow scenario groups
# ─────────────────────────────────────────
FLOW_GROUPS: Dict[str, Set[str]] = {
    "auth":         {"login","logout","signin","signup","register","otp","verification","verify",
                     "password","pin","biometrics","fingerprint","faceid","2fa","authentication"},
    "messaging":    {"message","chat","send","receive","delivery","delivered","read","unread",
                     "typing","sticker","emoji","gif","media","photo","video","file","document",
                     "forward","reply","delete","unsend","attachment"},
    "calling":      {"call","voice","videocall","video","ringing","ring","answer","decline",
                     "reject","missed","mute","speaker","bluetooth","headset","mic","microphone",
                     "echo","noise"},
    "notification": {"notification","push","badge","sound","vibration","alert","banner"},
    "channel":      {"channel","discovery","explore","search","find","broadcast"},
    "story":        {"story","status","highlight","viewer","reaction"},
    "settings":     {"settings","profile","privacy","account","theme","language","backup",
                     "sync","storage"},
    "permission":   {"permission","camera","contacts","location","microphone","allow","deny",
                     "granted","revoked"},
    "crash":        {"crash","freeze","hang","stuck","lag","slow","anr","unresponsive",
                     "force_close","not_responding","black_screen","white_screen"},
    "payment":      {"payment","purchase","subscription","billing","invoice","refund","card",
                     "wallet","topup","transfer","transaction"},
    "ui":           {"menu","overflow","kebab","tab","button","tap","click","press",
                     "longpress","swipe","scroll","open","close","back","gesture",
                     "layout","overlap","misalign","truncate","cut","hidden"},
}

STOPWORDS = set(
    "a an the and or but if then else when while for to of in on at by with without "
    "from into is are was were be been being this that these those it its as "
    "ve veya ama eger ise degil icin ile bir bu da de".split()
)

IGNORE_REGEXES = [
    r"\b(app\s*)?version\s*[:=]\s*[^\n\r]+",
    r"\bbuild\s*[:=]\s*[^\n\r]+",
    r"\bdevice\s*[:=]\s*[^\n\r]+",
    r"\blogs?\s*[:=]\s*[^\n\r]+",
    r"\brepro(duction)?\b.*",
    r"https?://\S+",
    r"\b\d+\.\d+\.\d+(\.\d+)?\b",
    r"\b\d+\b",
]


# ─────────────────────────────────────────
# Sentence-transformers
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading semantic model...")
def load_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(SENTENCE_MODEL_NAME)
    except ImportError:
        st.error("sentence-transformers not installed.\n\npip install sentence-transformers")
        return None


@st.cache_data(show_spinner="Encoding issue texts...")
def encode_texts(_model, texts: Tuple[str, ...]) -> Optional[np.ndarray]:
    if _model is None:
        return None
    return _model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-norm → dot product == cosine similarity
    )


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def pick_col(cols: List[str], keywords: List[str]) -> Optional[str]:
    for c in cols:
        if any(k in c.lower() for k in keywords):
            return c
    return None


def normalize_text(summary, desc) -> str:
    s = f"{'' if pd.isna(summary) else str(summary)} {'' if pd.isna(desc) else str(desc)}"
    s = s.lower()
    for pat in IGNORE_REGEXES:
        s = re.sub(pat, " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(norm: str) -> List[str]:
    return [t for t in norm.split() if len(t) >= 3 and t not in STOPWORDS]


def get_flows(token_set: Set[str]) -> Set[str]:
    return {flow for flow, kws in FLOW_GROUPS.items() if token_set & kws}


def shared_flow_count(a: Set[str], b: Set[str]) -> int:
    return len(get_flows(a) & get_flows(b))


def parse_created(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def keep_delete_order(i: int, j: int, created_dt: Optional[pd.Series]) -> Tuple[int, int]:
    """Older issue is kept; ties resolved by row index."""
    if created_dt is not None:
        ci, cj = created_dt.iloc[i], created_dt.iloc[j]
        if pd.notna(ci) and pd.notna(cj):
            return (i, j) if ci <= cj else (j, i)
    return (i, j) if i < j else (j, i)


def connected_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        if a != b:
            adj[a].append(b)
            adj[b].append(a)
    seen = [False] * n
    comps = []
    for i in range(n):
        if seen[i] or not adj[i]:
            continue
        q = deque([i])
        seen[i] = True
        comp = [i]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        if len(comp) >= 2:
            comps.append(sorted(comp))
    return comps


# ─────────────────────────────────────────
# Data prep
# ─────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(raw_csv: str):
    df = pd.read_csv(StringIO(raw_csv), sep=None, engine="python", on_bad_lines="skip")
    cols = list(df.columns)

    key_col     = pick_col(cols, ["issue key", "issue_key", "key"]) or cols[0]
    summary_col = pick_col(cols, ["summary", "title"])              or cols[0]
    desc_col    = pick_col(cols, ["description"])                   or cols[0]
    created_col = pick_col(cols, ["created", "created date", "created_at", "createdat"])

    w = df.copy()
    w["_key"]    = w[key_col].astype(str).str.strip()
    w["_norm"]   = [normalize_text(a, b) for a, b in zip(w[summary_col], w[desc_col])]
    w["_tokens"] = w["_norm"].apply(tokenize)
    w["_set"]    = w["_tokens"].apply(set)

    # Raw text for sentence encoder — full meaning preserved, no stripping
    w["_raw"] = [
        f"{'' if pd.isna(a) else str(a).strip()} {'' if pd.isna(b) else str(b).strip()}".strip()
        for a, b in zip(w[summary_col], w[desc_col])
    ]

    created_dt = parse_created(w[created_col]) if created_col else None
    detected = {
        "Issue Key":          key_col,
        "Summary / Title":    summary_col,
        "Description":        desc_col,
        "Created (optional)": created_col or "(not found)",
    }
    return w, created_dt, detected


@st.cache_data(show_spinner="Building TF-IDF candidate index...")
def tfidf_neighbors(texts: List[str], topk: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    X = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(texts)
    nn = NearestNeighbors(
        n_neighbors=min(topk + 1, X.shape[0]),
        metric="cosine",
        algorithm="brute",
    ).fit(X)
    dist, idx = nn.kneighbors(X)
    return idx[:, 1:], 1.0 - dist[:, 1:]


# ─────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────
def run_pipeline(
    work: pd.DataFrame,
    created_dt: Optional[pd.Series],
    emb: Optional[np.ndarray],
):
    n       = len(work)
    use_emb = emb is not None

    # Step 1 — cheap TF-IDF candidate generation
    nn_idx, nn_sim = tfidf_neighbors(work["_norm"].tolist(), TOP_K_NEIGHBORS)

    cand: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        for pos in range(nn_idx.shape[1]):
            j = int(nn_idx[i, pos])
            c = float(nn_sim[i, pos])
            if c < TFIDF_CANDIDATE_FLOOR:
                continue
            a, b = (i, j) if i < j else (j, i)
            if a != b and c > cand.get((a, b), 0.0):
                cand[(a, b)] = c

    # Step 2 — sentence-level semantic gate + flow guard
    rows:    List[Dict]            = []
    deleted: Set[int]              = set()
    edges:   List[Tuple[int, int]] = []

    for (a, b), tfidf_c in sorted(cand.items(), key=lambda x: x[1], reverse=True):
        ki, di = keep_delete_order(a, b, created_dt)
        if di in deleted:
            continue

        # Flow guard — same user scenario required
        if FLOW_GUARD and shared_flow_count(work["_set"].iloc[ki], work["_set"].iloc[di]) < 1:
            continue

        # Sentence-level semantic similarity (primary decision signal)
        sem = float(np.dot(emb[ki], emb[di])) if use_emb else tfidf_c

        if sem < SEMANTIC_DUPLICATE_THRESHOLD:
            continue
        if round(sem, 3) < MIN_SIMILARITY_DISPLAY:
            continue

        rows.append({
            "Issue (Keep)":      work["_key"].iloc[ki],
            "Issue (Duplicate)": work["_key"].iloc[di],
            "Similarity":        round(sem, 3),
        })
        deleted.add(di)
        edges.append((ki, di))

    # Step 3 — output dataframes
    dup_df = pd.DataFrame(rows)
    if not dup_df.empty:
        dup_df = dup_df.sort_values("Similarity", ascending=False).reset_index(drop=True)

    cluster_df = None
    clusters = connected_components(n, edges) if edges else []
    if clusters:
        cluster_rows = []
        for cid, comp in enumerate(clusters, start=1):
            all_flows: Set[str] = set()
            for i in comp:
                all_flows |= get_flows(work["_set"].iloc[i])
            cluster_rows.append({
                "Cluster": cid,
                "Size":    len(comp),
                "Flow":    ", ".join(sorted(all_flows)),
                "Members": ", ".join(work["_key"].iloc[i] for i in comp),
            })
        cluster_df = (
            pd.DataFrame(cluster_rows)
            .sort_values(["Size", "Cluster"], ascending=[False, True])
            .reset_index(drop=True)
        )

    summary_df = pd.DataFrame([
        {"Metric": "Total issues",         "Value": int(n)},
        {"Metric": "Semantic model",       "Value": SENTENCE_MODEL_NAME if use_emb else "TF-IDF only"},
        {"Metric": "Semantic threshold",   "Value": SEMANTIC_DUPLICATE_THRESHOLD},
        {"Metric": "Min similarity shown", "Value": MIN_SIMILARITY_DISPLAY},
        {"Metric": "Flow guard",           "Value": "on" if FLOW_GUARD else "off"},
        {"Metric": "Duplicates found",     "Value": int(len(dup_df)) if not dup_df.empty else 0},
        {"Metric": "Clusters",             "Value": int(len(cluster_df)) if cluster_df is not None else 0},
    ])

    return summary_df, dup_df, cluster_df


# ─────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────
def main():
    st.set_page_config(page_title="Defect Duplicate Detector", layout="wide")
    st.title("Defect Duplicate Detector")

    model = load_sentence_model()
    if model is None:
        st.warning("Running in TF-IDF fallback mode — install sentence-transformers for full semantic detection.")

    uploaded = st.file_uploader("Upload defects CSV", type=["csv"])
    if not uploaded:
        st.stop()

    raw  = uploaded.getvalue().decode("utf-8", errors="replace")
    work, created_dt, detected = load_and_preprocess(raw)

    with st.expander("Detected columns", expanded=False):
        st.json(detected)

    emb = encode_texts(model, tuple(work["_raw"].tolist())) if model else None

    with st.spinner("Detecting duplicates..."):
        summary_df, dup_df, cluster_df = run_pipeline(work, created_dt, emb)

    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.subheader("Duplicate Issues")
    if dup_df.empty:
        st.info("No duplicates found above the confidence threshold.")
    else:
        st.dataframe(dup_df, use_container_width=True, hide_index=True)

    st.subheader("Clusters")
    if cluster_df is None:
        st.info("No clusters.")
    else:
        st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    st.divider()

    def to_csv_section(df: Optional[pd.DataFrame], title: str) -> str:
        if df is None or df.empty:
            return f"# {title}\n(empty)\n\n"
        return f"# {title}\n{df.to_csv(index=False)}\n"

    csv_out  = to_csv_section(dup_df,     "DUPLICATES")
    csv_out += to_csv_section(cluster_df, "CLUSTERS")

    st.download_button(
        label="Download Results CSV",
        data=csv_out.encode("utf-8"),
        file_name="defect_duplicates.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

