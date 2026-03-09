import streamlit as st
import json
import csv
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_filter_chunks():
    """Load filtered chunks from filter_chunk/ (all papers, method-agnostic)."""
    chunks_dir = os.path.join(BASE_DIR, "filter_chunk")
    data = {}
    for fpath in sorted(glob.glob(os.path.join(chunks_dir, "*.json"))):
        paper_id = int(os.path.splitext(os.path.basename(fpath))[0])
        with open(fpath, "r", encoding="utf-8") as f:
            data[paper_id] = json.load(f)
    return data


@st.cache_data
def load_metadata():
    meta_path = os.path.join(BASE_DIR, "data", "metadata", "meta_data.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {item["paper_id"]: item for item in raw}


@st.cache_data
def load_csv_assignments(filename):
    fpath = os.path.join(BASE_DIR, "assignments", filename)
    if not os.path.exists(fpath):
        return {}
    data = {}
    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["paper_id"])
            data[pid] = row
    return data


@st.cache_data
def load_semantic_raw():
    fpath = os.path.join(BASE_DIR, "assignments", "semantic_matching.csv")
    if not os.path.exists(fpath):
        return {}
    data = {}
    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["paper_id"])
            l1_methods = row.get("l1_method", "").split("; ") if row.get("l1_method") else []
            l1_sims = row.get("l1_similarity", "").split("; ") if row.get("l1_similarity") else []
            l2_methods = row.get("l2_method", "").split("; ") if row.get("l2_method") else []
            l2_sims = row.get("l2_similarity", "").split("; ") if row.get("l2_similarity") else []

            l1_pairs = list(zip(l1_methods, l1_sims)) if len(l1_methods) == len(l1_sims) else []
            l2_pairs = list(zip(l2_methods, l2_sims)) if len(l2_methods) == len(l2_sims) else []

            data[pid] = {"l1": l1_pairs, "l2": l2_pairs}
    return data


@st.cache_data
def load_l2_to_l1_map():
    fpath = os.path.join(BASE_DIR, "predefine", "l2_methods.json")
    with open(fpath, "r", encoding="utf-8") as f:
        l2_methods = json.load(f)
    return {m["method_name"]: m["level_1_label"] for m in l2_methods}


TIER_COLORS = {1: "#16a34a", 2: "#ca8a04"}
TIER_LABELS = {1: "Tier 1", 2: "Tier 2"}


def render_notebooklm(row):
    if not row:
        st.caption("No NotebookLM assignment for this paper")
        return
    l1 = row.get("l1_method", "")
    l2 = row.get("l2_method", "")
    html = ""
    if l1:
        html += (
            f'<span style="background-color:#6366f1;color:white;padding:2px 8px;'
            f'border-radius:4px;margin:2px;display:inline-block;font-size:0.85em">'
            f'L1: {l1}</span> '
        )
    if l2:
        html += (
            f'<span style="background-color:#14b8a6;color:white;padding:2px 8px;'
            f'border-radius:4px;margin:2px;display:inline-block;font-size:0.85em">'
            f'L2: {l2}</span>'
        )
    if html:
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.caption("None")


def render_keyword_labels(row):
    if not row:
        st.caption("No keyword matches")
        return
    l1 = row.get("l1_method", "")
    l2 = row.get("l2_method", "")
    l1_list = [m.strip() for m in l1.split("; ")] if l1 else []
    l2_list = [m.strip() for m in l2.split("; ")] if l2 else []

    if not l1_list:
        st.caption("No keyword matches")
        return

    for l1_name in l1_list:
        st.markdown(
            f'<span style="background-color:#3b82f6;color:white;padding:2px 8px;'
            f'border-radius:4px;margin:2px;display:inline-block;font-size:0.85em">'
            f'L1: {l1_name}</span>',
            unsafe_allow_html=True,
        )
        child_l2 = [m for m in l2_list if _l2_belongs_to_l1(m, l1_name)]
        for l2_name in child_l2:
            st.markdown(
                f'<span style="margin-left:20px;background-color:#22c55e;color:white;padding:2px 8px;'
                f'border-radius:4px;margin-top:2px;display:inline-block;font-size:0.85em">'
                f'L2: {l2_name}</span>',
                unsafe_allow_html=True,
            )

    orphan_l2 = [m for m in l2_list if not any(_l2_belongs_to_l1(m, l1n) for l1n in l1_list)]
    for l2_name in orphan_l2:
        st.markdown(
            f'<span style="background-color:#22c55e;color:white;padding:2px 8px;'
            f'border-radius:4px;margin:2px;display:inline-block;font-size:0.85em">'
            f'L2: {l2_name}</span>',
            unsafe_allow_html=True,
        )


def render_semantic_labels(semantic_data, l1_thresh, l2_thresh, l2_to_l1):
    if not semantic_data:
        st.caption("No semantic matches")
        return

    active_l1 = []
    for name, sim_str in semantic_data.get("l1", []):
        try:
            sim = float(sim_str)
        except ValueError:
            continue
        if sim >= l1_thresh:
            active_l1.append((name, sim))

    active_l2 = []
    for name, sim_str in semantic_data.get("l2", []):
        try:
            sim = float(sim_str)
        except ValueError:
            continue
        if sim >= l2_thresh:
            active_l2.append((name, sim))

    if not active_l1:
        st.caption("None above threshold")
        return

    for l1_name, l1_sim in active_l1:
        st.markdown(
            f'<span style="background-color:#f97316;color:white;padding:2px 8px;'
            f'border-radius:4px;margin:2px;display:inline-block;font-size:0.85em">'
            f'L1: {l1_name} ({l1_sim:.3f})</span>',
            unsafe_allow_html=True,
        )
        child_l2 = [(n, s) for n, s in active_l2 if l2_to_l1.get(n) == l1_name]
        for l2_name, l2_sim in child_l2:
            st.markdown(
                f'<span style="margin-left:20px;background-color:#a855f7;color:white;padding:2px 8px;'
                f'border-radius:4px;margin-top:2px;display:inline-block;font-size:0.85em">'
                f'L2: {l2_name} ({l2_sim:.3f})</span>',
                unsafe_allow_html=True,
            )

    shown_l1_names = {n for n, _ in active_l1}
    orphans = [(n, s) for n, s in active_l2 if l2_to_l1.get(n) not in shown_l1_names]
    for l2_name, l2_sim in orphans:
        st.markdown(
            f'<span style="background-color:#a855f7;color:white;padding:2px 8px;'
            f'border-radius:4px;margin:2px;display:inline-block;font-size:0.85em">'
            f'L2: {l2_name} ({l2_sim:.3f})</span>',
            unsafe_allow_html=True,
        )


_L2_TO_L1 = None


def _l2_belongs_to_l1(l2_name, l1_name):
    global _L2_TO_L1
    if _L2_TO_L1 is None:
        _L2_TO_L1 = load_l2_to_l1_map()
    return _L2_TO_L1.get(l2_name) == l1_name


def go_prev():
    st.session_state.paper_idx = max(0, st.session_state.paper_idx - 1)


def go_next(max_idx):
    st.session_state.paper_idx = min(max_idx, st.session_state.paper_idx + 1)


def on_select(paper_ids):
    selected_pid = st.session_state._pid_select
    st.session_state.paper_idx = paper_ids.index(selected_pid)


def main():
    st.set_page_config(page_title="Method Classification Viewer", layout="wide")
    st.title("Method Classification Viewer")

    # Load data
    filter_chunks = load_filter_chunks()
    metadata = load_metadata()
    keyword_data = load_csv_assignments("keyword_match.csv")
    notebooklm_data = load_csv_assignments("NotebookLM.csv")
    semantic_raw = load_semantic_raw()
    l2_to_l1 = load_l2_to_l1_map()

    if not filter_chunks:
        st.warning("No filtered chunks found. Run main.py first.")
        return

    paper_ids = sorted(filter_chunks.keys())

    if "paper_idx" not in st.session_state:
        st.session_state.paper_idx = 0

    # Sidebar
    with st.sidebar:
        st.header("Navigation")

        col_prev, col_counter, col_next = st.columns([1, 2, 1])
        with col_prev:
            st.button("Prev", on_click=go_prev, use_container_width=True)
        with col_counter:
            st.markdown(
                f"<div style='text-align:center;padding-top:6px'>"
                f"<b>{st.session_state.paper_idx + 1} / {len(paper_ids)}</b></div>",
                unsafe_allow_html=True,
            )
        with col_next:
            st.button("Next", on_click=go_next, args=(len(paper_ids) - 1,), use_container_width=True)

        st.selectbox(
            "Paper ID",
            paper_ids,
            index=st.session_state.paper_idx,
            key="_pid_select",
            on_change=on_select,
            args=(paper_ids,),
        )

        st.divider()
        st.header("Semantic Thresholds")
        st.caption("Override z-score filtering for display")
        l1_thresh = st.slider("L1 Similarity Threshold", 0.0, 1.0, 0.0, 0.01,
                              help="Show L1 methods with similarity above this value (0 = show all z-score assigned)")
        l2_thresh = st.slider("L2 Similarity Threshold", 0.0, 1.0, 0.0, 0.01,
                              help="Show L2 methods with similarity above this value (0 = show all z-score assigned)")

        st.divider()
        st.header("Stats")
        n_with_kw = sum(1 for pid in paper_ids if keyword_data.get(pid, {}).get("l1_method", ""))
        n_with_sem = sum(1 for pid in paper_ids if semantic_raw.get(pid, {}).get("l1", []))
        st.metric("Total papers", len(paper_ids))
        st.metric("With keyword matches", n_with_kw)
        st.metric("With semantic matches", n_with_sem)

    current_pid = paper_ids[st.session_state.paper_idx]

    # Paper metadata header
    meta = metadata.get(current_pid, {})
    st.markdown(f"### Paper {current_pid}: {meta.get('title', 'Unknown')}")
    st.markdown(f"**Author:** {meta.get('author', 'N/A')} | **University:** {meta.get('university', 'N/A')}")

    # --- Filtered Chunks ---
    st.divider()
    st.subheader("Filtered Chunks")
    chunks = filter_chunks.get(current_pid, [])

    if not chunks:
        st.caption("No chunks passed filtering for this paper.")
    else:
        for i, chunk in enumerate(chunks):
            section = chunk.get("section_name", "Unknown")
            text = chunk.get("text", "")
            keywords = chunk.get("matched_keywords", [])
            l1 = chunk.get("matched_l1_methods", [])
            l2 = chunk.get("matched_l2_methods", [])
            tier = chunk.get("tier", 1)

            tier_color = TIER_COLORS.get(tier, "#6b7280")
            tier_label = TIER_LABELS.get(tier, f"Tier {tier}")

            with st.expander(
                f"[{tier_label}] Section: {section} | Paragraph {chunk.get('paragraph_id', i)}",
                expanded=True,
            ):
                highlighted = text
                for kw in sorted(keywords, key=len, reverse=True):
                    highlighted = highlighted.replace(kw, f"**:orange[{kw}]**")
                st.markdown(highlighted)

                tag_html = (
                    f'<span style="background-color:{tier_color};color:white;padding:1px 6px;'
                    f'border-radius:3px;margin:1px;font-size:0.75em">{tier_label}</span> '
                )
                for m in l1:
                    tag_html += (
                        f'<span style="background-color:#3b82f6;color:white;padding:1px 6px;'
                        f'border-radius:3px;margin:1px;font-size:0.75em">L1: {m}</span> '
                    )
                for m in l2:
                    tag_html += (
                        f'<span style="background-color:#22c55e;color:white;padding:1px 6px;'
                        f'border-radius:3px;margin:1px;font-size:0.75em">L2: {m}</span> '
                    )
                st.markdown(tag_html, unsafe_allow_html=True)

    # --- Method Assignments ---
    st.divider()
    st.subheader("Method Assignments")

    col_nlm, col_kw, col_sem = st.columns(3)

    with col_nlm:
        st.markdown("#### NotebookLM")
        render_notebooklm(notebooklm_data.get(current_pid))

    with col_kw:
        st.markdown("#### Keyword Match")
        render_keyword_labels(keyword_data.get(current_pid))

    with col_sem:
        st.markdown("#### Semantic Match")
        render_semantic_labels(semantic_raw.get(current_pid), l1_thresh, l2_thresh, l2_to_l1)


if __name__ == "__main__":
    main()
