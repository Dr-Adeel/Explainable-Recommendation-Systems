"""
Explainable Fashion Recommendation System — Unified Streamlit Interface
=======================================================================
Single-page application that puts the **Hybrid engine** front-and-center
and integrates SHAP-based explanations directly into the results.
Secondary tabs give access to the individual Image Similarity and
User (ALS) engines for exploration.
"""
from __future__ import annotations

import requests
import streamlit as st
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except Exception:
    px = None
    go = None
    _HAS_PLOTLY = False


# ──────────────────────────── config ────────────────────────────
DEFAULT_API_BASE = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Explainable Fashion Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── CSS ───────────────────────────────
st.markdown(
    """
    <style>
    /* ── theme-aware variables ── */
    :root {
        --card-bg: #ffffff;
        --card-border: #e5e7eb;
        --card-shadow: rgba(0,0,0,.06);
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-muted: #9ca3af;
        --reason-bg: #eef2ff;
        --reason-text: #3730a3;
        --reason-border: #c7d2fe;
        --bar-bg: #e5e7eb;
        --score-bg: #ecfdf5;
        --score-text: #065f46;
        --score-border: #a7f3d0;
        --metric-bg: #f9fafb;
        --query-bg: #fefce8;
        --query-border: #fde68a;
        --section-bg: #f8fafc;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: #1e1e2e; --card-border: #3b3b4f; --card-shadow: rgba(0,0,0,.25);
            --text-primary: #e4e4e7; --text-secondary: #a1a1aa; --text-muted: #71717a;
            --reason-bg: #1e293b; --reason-text: #93c5fd; --reason-border: #334155;
            --bar-bg: #3b3b4f;
            --score-bg: #14532d; --score-text: #86efac; --score-border: #166534;
            --metric-bg: #27272a;
            --query-bg: #422006; --query-border: #78350f;
            --section-bg: #18181b;
        }
    }
    [data-testid="stAppViewContainer"][data-theme="dark"],
    .stApp[data-theme="dark"] {
        --card-bg: #1e1e2e; --card-border: #3b3b4f; --card-shadow: rgba(0,0,0,.25);
        --text-primary: #e4e4e7; --text-secondary: #a1a1aa; --text-muted: #71717a;
        --reason-bg: #1e293b; --reason-text: #93c5fd; --reason-border: #334155;
        --bar-bg: #3b3b4f;
        --score-bg: #14532d; --score-text: #86efac; --score-border: #166534;
        --metric-bg: #27272a;
        --query-bg: #422006; --query-border: #78350f;
        --section-bg: #18181b;
    }

    /* ─── cards ─── */
    .reco-card {border-radius:12px; padding:16px; border:1px solid var(--card-border);
                box-shadow:0 2px 12px var(--card-shadow); margin:8px 0; background:var(--card-bg)}
    .reco-title {font-weight:700; font-size:14px; margin-bottom:2px; color:var(--text-primary);
                 line-height:1.3; user-select:text; -webkit-user-select:text}
    .reco-meta  {color:var(--text-muted); font-size:12px; margin-bottom:6px;
                 user-select:text; -webkit-user-select:text}
    .reco-reason{background:var(--reason-bg); color:var(--reason-text);
                 border:1px solid var(--reason-border);
                 padding:10px 12px; border-radius:8px;
                 margin:10px 0 8px; font-size:13px; line-height:1.5;
                 user-select:text; -webkit-user-select:text}

    /* ─── hero ─── */
    .hero {text-align:center; padding:16px 0 4px}
    .hero h1 {font-size:2.1rem; margin-bottom:2px; color:var(--text-primary); letter-spacing:-0.5px}
    .hero p  {color:var(--text-secondary); font-size:1rem; margin-top:0}

    /* ─── query card ─── */
    .query-card {background:var(--query-bg); border:1px solid var(--query-border);
                 border-radius:12px; padding:12px 16px; margin:8px 0}
    .query-card h3 {margin:0 0 4px; font-size:16px; color:var(--text-primary)}
    .query-card .caption {color:var(--text-secondary); font-size:13px}

    /* ─── score bars ─── */
    .score-bars-block {background:var(--card-bg); border:1px solid var(--card-border);
                       border-radius:10px; padding:12px 14px; margin:8px 0}
    .score-bar-wrap {margin-top:7px; user-select:text; -webkit-user-select:text}
    .score-bar-wrap:first-child {margin-top:0}
    .score-bar-label {font-size:12px; color:var(--text-secondary); margin-bottom:3px;
                      display:flex; justify-content:space-between; font-weight:500;
                      user-select:text; -webkit-user-select:text}
    .score-bar-bg {background:var(--bar-bg); border-radius:6px; height:10px; overflow:hidden}
    .score-bar-fill {height:100%; border-radius:6px; transition:width 0.5s ease}
    .score-bar-fill.img  {background:linear-gradient(90deg,#a5b4fc,#6366f1)}
    .score-bar-fill.als  {background:linear-gradient(90deg,#fdba74,#f97316)}
    .score-bar-fill.pop  {background:linear-gradient(90deg,#6ee7b7,#10b981)}

    /* ─── score total badge ─── */
    .score-total {text-align:center; margin:8px 0 4px; padding:6px 10px;
                  background:var(--score-bg); border:1px solid var(--score-border);
                  border-radius:8px; font-size:13px; font-weight:600; color:var(--score-text);
                  user-select:text; -webkit-user-select:text}

    /* ─── param section ─── */
    .param-section {background:var(--section-bg); border:1px solid var(--card-border);
                    border-radius:12px; padding:16px; margin-bottom:12px}
    .param-title {font-size:14px; font-weight:700; color:var(--text-primary); margin-bottom:10px}

    /* ─── uniform image sizing ─── */
    [data-testid="stImage"] {
        width:100%; overflow:hidden; border-radius:10px;
    }
    [data-testid="stImage"] img {
        width:100%; height:200px; object-fit:cover;
        object-position:center top; border-radius:10px;
        border:1px solid var(--card-border);
    }

    /* ─── selectable text everywhere ─── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] div {
        user-select:text !important;
        -webkit-user-select:text !important;
    }

    /* ─── slider labels tighter ─── */
    [data-testid="stSlider"] {margin-bottom:-4px}

    /* ─── feedback buttons ─── */
    .feedback-row {display:flex; gap:6px; margin-top:6px}
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────── helpers ───────────────────────────
def api_base() -> str:
    return st.session_state.get("api_base", DEFAULT_API_BASE).rstrip("/")


def api_get(path: str, params: dict | None = None, timeout: int = 30):
    r = requests.get(f"{api_base()}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def img_url(item_idx: int) -> str:
    return f"{api_base()}/amazon/image/{int(item_idx)}"


@st.cache_data(ttl=30)
def cached_health(base: str) -> dict:
    r = requests.get(f"{base}/health", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=120)
def sample_items(base: str, n: int = 100) -> list[dict]:
    r = requests.get(
        f"{base}/amazon/sample-items",
        params={"n": n, "with_images": True, "seed": 42},
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("items", [])


def item_meta(item_idx: int) -> dict | None:
    try:
        return api_get(f"/amazon/item/{int(item_idx)}", timeout=6)
    except Exception:
        return None


def send_feedback(query: int, candidate: int, helpful: bool):
    try:
        r = requests.post(
            f"{api_base()}/amazon/feedback",
            json={"query_item": query, "candidate_item": candidate, "helpful": helpful},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False


def pct(v) -> str:
    """Format a 0-1 float as percentage."""
    try:
        return f"{float(v) * 100:.1f} %"
    except Exception:
        return "-"


# ──────────────────────────── render helpers ────────────────────
def render_query_card(item: dict) -> None:
    """Show the query product in a prominent card."""
    cols = st.columns([1, 4])
    with cols[0]:
        st.image(img_url(int(item["item_idx"])), use_container_width=True)
    with cols[1]:
        st.markdown(
            f"<div class='query-card'>"
            f"<h3>🎯 {item.get('title', 'Produit inconnu')}</h3>"
            f"<div class='caption'>🏷️ {item.get('main_category', '')}  •  id = {item.get('item_idx')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_simple_grid(items: list[dict], cols_count: int = 5, score_key: str | None = None) -> None:
    """Simple grid for image-similarity / ALS results (no SHAP).
    
    If *score_key* is provided (e.g. ``"score"``), the raw score is shown as a percentage bar.
    """
    if not items:
        st.info("Aucun résultat.")
        return

    # normalise scores to [0, 100] for display
    raw_scores = []
    if score_key:
        raw_scores = [float(rec.get(score_key, 0)) for rec in items]
    max_score = max(raw_scores) if raw_scores and max(raw_scores) > 0 else 1.0

    cols = st.columns(cols_count)
    for i, rec in enumerate(items):
        with cols[i % cols_count]:
            iid = int(rec.get("item_idx"))
            st.image(img_url(iid), use_container_width=True)
            st.caption(rec.get("title", "")[:55] or f"id {iid}")
            cat = rec.get("main_category")
            if cat:
                st.caption(f"🏷️ {cat}")

            # score as percentage bar
            if raw_scores:
                pct_val = raw_scores[i] / max_score * 100
                st.markdown(
                    f"<div class='score-bar-wrap'>"
                    f"<div class='score-bar-label'><span>Score</span><span>{pct_val:.1f} %</span></div>"
                    f"<div class='score-bar-bg'><div class='score-bar-fill img' style='width:{pct_val:.0f}%'></div></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            reason = rec.get("reason")
            if reason:
                st.markdown(
                    f"<div class='reco-reason'>{reason}</div>",
                    unsafe_allow_html=True,
                )


def _contrib_label(name: str) -> str:
    return {"multimodal_cosine": "Image (CLIP)", "als_dot": "Utilisateur (ALS)", "popularity": "Popularité"}.get(
        name, name
    )


def _compute_shares(features: dict) -> dict:
    """Compute percentage shares from raw feature values.

    ALS dot-product can range into hundreds while cosine and popularity
    are in [0, 1].  We apply a sigmoid-like squash on ALS so all three
    signals live on a comparable 0-1 scale before computing shares.
    """
    import math

    img_v = max(float(features.get("multimodal_cosine", 0)), 0)
    raw_als = float(features.get("als_dot", 0))
    pop_v = max(float(features.get("popularity", 0)), 0)

    # squash ALS to [0, 1] via sigmoid: 1/(1+exp(-x/scale))
    # scale=30 makes values around 30 map to ~0.63, around 100 → ~0.96
    als_v = 1.0 / (1.0 + math.exp(-raw_als / 30.0)) if raw_als > -500 else 0.0

    total = img_v + als_v + pop_v
    if total < 1e-12:
        return {"image": 0, "als": 0, "pop": 0, "total": 0}
    return {
        "image": img_v / total * 100,
        "als": als_v / total * 100,
        "pop": pop_v / total * 100,
        "total": total,
    }


def _build_reason_fr(shares: dict) -> str:
    """Build a clear French explanation sentence from percentage shares."""
    img_pct = shares["image"]
    als_pct = shares["als"]
    pop_pct = shares["pop"]

    parts = sorted(
        [("image", img_pct), ("als", als_pct), ("pop", pop_pct)],
        key=lambda x: x[1],
        reverse=True,
    )
    top_name, top_val = parts[0]
    second_name, second_val = parts[1]

    if top_val < 1:
        return "Recommandé sur la base de signaux combinés."

    if top_name == "image":
        reason = f"Produit visuellement similaire (similarité image : {img_pct:.0f} %)"
        if second_val > 15:
            if second_name == "als":
                reason += f", renforcé par l'historique d'achat ({als_pct:.0f} %)"
            else:
                reason += f", soutenu par la popularité ({pop_pct:.0f} %)"
        return reason + "."
    elif top_name == "als":
        reason = f"Les acheteurs de ce produit ont aussi acheté cet article (signal utilisateur : {als_pct:.0f} %)"
        if second_val > 15:
            if second_name == "image":
                reason += f", avec une similarité visuelle ({img_pct:.0f} %)"
            else:
                reason += f", soutenu par la popularité ({pop_pct:.0f} %)"
        return reason + "."
    else:
        reason = f"Article populaire (popularité : {pop_pct:.0f} %)"
        if second_val > 15:
            if second_name == "image":
                reason += f", avec une similarité visuelle ({img_pct:.0f} %)"
            else:
                reason += f", renforcé par l'historique d'achat ({als_pct:.0f} %)"
        return reason + "."


def render_hybrid_cards(
    explanations: list[dict],
    query_idx: int,
    cols_per_row: int = 3,
) -> None:
    """Render hybrid results with integrated SHAP / contribution explanations."""
    if not explanations:
        st.info("Aucun résultat hybride.")
        return

    rows = [explanations[i : i + cols_per_row] for i in range(0, len(explanations), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, cand in zip(cols, row):
            cid = int(cand.get("item_idx", 0))
            pred = cand.get("surrogate_pred")
            features = cand.get("features") or {}
            contribs = cand.get("contribs") or {}

            with col:
                # image
                col.image(img_url(cid), use_container_width=True)

                # title + category
                meta = item_meta(cid)
                title = meta.get("title", f"Item {cid}") if meta else f"Item {cid}"
                category = meta.get("main_category", "") if meta else ""
                col.markdown(f"<div class='reco-title'>{title[:70]}</div>", unsafe_allow_html=True)
                col.markdown(f"<div class='reco-meta'>🏷️ {category}  •  id {cid}</div>", unsafe_allow_html=True)

                # ── compute percentage shares ──
                shares = _compute_shares(features)
                img_pct = shares["image"]
                als_pct = shares["als"]
                pop_pct = shares["pop"]

                # ── explanation reason in French ──
                reason_text = _build_reason_fr(shares)
                col.markdown(f"<div class='reco-reason'>💡 {reason_text}</div>", unsafe_allow_html=True)

                # ── visual score bars inside a card block ──
                def score_bar(label: str, value: float, css_class: str) -> str:
                    w = max(min(value, 100), 0)
                    return (
                        f"<div class='score-bar-wrap'>"
                        f"<div class='score-bar-label'><span>{label}</span><span>{value:.1f} %</span></div>"
                        f"<div class='score-bar-bg'><div class='score-bar-fill {css_class}' style='width:{w}%'></div></div>"
                        f"</div>"
                    )

                bars_html = (
                    "<div class='score-bars-block'>"
                    + score_bar("🖼️ Image", img_pct, "img")
                    + score_bar("👥 Utilisateur", als_pct, "als")
                    + score_bar("📈 Popularité", pop_pct, "pop")
                    + "</div>"
                )
                col.markdown(bars_html, unsafe_allow_html=True)

                # ── surrogate score badge ──
                if pred is not None:
                    col.markdown(
                        f"<div class='score-total'>🎯 Score : {pred:.4f}</div>",
                        unsafe_allow_html=True,
                    )

                # ── SHAP bar chart (if plotly available) ──
                feat_names = contribs.get("features", [])
                vals = contribs.get("shap_values") or contribs.get("contributions", [])
                if _HAS_PLOTLY and feat_names and vals and len(feat_names) == len(vals):
                    df_bar = pd.DataFrame(
                        {"Signal": [_contrib_label(f) for f in feat_names], "Contribution": [float(v) for v in vals]}
                    )
                    fig = px.bar(
                        df_bar,
                        x="Contribution",
                        y="Signal",
                        orientation="h",
                        color="Signal",
                        color_discrete_sequence=["#6366f1", "#f97316", "#10b981"],
                        height=140,
                    )
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        showlegend=False,
                        yaxis_title="",
                        xaxis_title="Contribution SHAP",
                        font=dict(size=11),
                    )
                    col.plotly_chart(fig, use_container_width=True, key=f"shap_{cid}")

                # ── feedback buttons ──
                fb1, fb2 = col.columns(2)
                if fb1.button("👍 Utile", key=f"up_{cid}"):
                    if send_feedback(query_idx, cid, True):
                        col.success("Merci !")
                    else:
                        col.error("Échec")
                if fb2.button("👎 Pas utile", key=f"dn_{cid}"):
                    if send_feedback(query_idx, cid, False):
                        col.success("Merci !")
                    else:
                        col.error("Échec")


# ──────────────────────────── SIDEBAR ───────────────────────────
def sidebar():
    st.sidebar.markdown("## ⚙️ Configuration")
    st.sidebar.text_input("API Base URL", value=api_base(), key="api_base")
    base = api_base()

    try:
        h = cached_health(base)
        items_n = h.get("items_total", "?")
        img_n = h.get("image_rows", "?")
        als_u = h.get("als_users", "?")
        st.sidebar.success(f"✅  API connectée — {items_n} items, {img_n} embeddings, {als_u} users ALS")
    except Exception as e:
        st.sidebar.error("❌ API inaccessible")
        st.sidebar.caption(str(e))
        st.stop()

    st.sidebar.divider()

    # load sample items for selectors
    try:
        items = sample_items(base, n=100)
    except Exception:
        items = []
    return items


# ──────────────────────────── MAIN ──────────────────────────────
def main() -> None:
    catalog = sidebar()

    # ── header ──
    st.markdown(
        "<div class='hero'><h1>🛍️ Explainable Fashion Recommender</h1>"
        "<p>Système de recommandation hybride explicable — Amazon Fashion</p></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── tabs (Hybrid first) ──
    tab_hybrid, tab_image, tab_user, tab_global = st.tabs(
        ["🔀  Hybride (principal)", "🖼️  Similarité Image", "👤  Utilisateur (ALS)", "🌍  Vue Globale"]
    )

    # build label map from catalog
    label_map = {
        f"{it['item_idx']} — {it.get('title', '')[:65]}": it
        for it in catalog
        if "item_idx" in it
    }
    label_list = list(label_map.keys())

    # ────────────────── TAB 1 : HYBRID (main) ──────────────────
    with tab_hybrid:
        st.subheader("Recommandation Hybride (CLIP + ALS + Popularité)")
        st.caption(
            "Combine la similarité visuelle, le filtrage collaboratif et la popularité. "
            "Les résultats sont accompagnés d'explications SHAP."
        )

        # ── product selector ──
        choice_h = st.selectbox(
            "📦 Sélectionner un produit du catalogue",
            label_list,
            index=0,
            key="hy_sel",
        )
        chosen_h = label_map.get(choice_h, {})
        default_idx = int(chosen_h.get("item_idx", 0))

        # ── parameters in 3 clean columns ──
        st.markdown("")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("##### ⚖️ Poids des signaux")
            alpha = st.slider("α — Image (CLIP)", 0.0, 1.0, 0.5, 0.05, key="hy_a")
            beta = st.slider("β — Utilisateur (ALS)", 0.0, 1.0, 0.4, 0.05, key="hy_b")
            gamma_val = st.slider("γ — Popularité", 0.0, 1.0, 0.1, 0.05, key="hy_g")
        with p2:
            st.markdown("##### 🔧 Paramètres")
            k_h = st.slider("Nombre de résultats (k)", 1, 20, 6, key="hy_k")
            pool_h = st.slider("Taille du pool candidat", 50, 5000, 500, 50, key="hy_pool")
            filter_cat = st.checkbox("Filtrer par catégorie", value=True, key="hy_fc")
        with p3:
            st.markdown("##### 📝 Saisie manuelle")
            item_idx_h = st.number_input(
                "item_idx",
                min_value=0,
                value=default_idx,
                step=1,
                key="hy_idx",
            )
            st.markdown("")

        # ── query preview (uses the actual item_idx_h so it updates on manual input) ──
        actual_idx = int(item_idx_h)
        if actual_idx == default_idx and chosen_h:
            # still matches the selectbox → use cached catalog data
            preview_meta = {
                "item_idx": actual_idx,
                "title": chosen_h.get("title", ""),
                "main_category": chosen_h.get("main_category", ""),
            }
        else:
            # manual override → fetch metadata from the API
            meta = item_meta(actual_idx)
            preview_meta = {
                "item_idx": actual_idx,
                "title": meta.get("title", f"Produit {actual_idx}") if meta else f"Produit {actual_idx}",
                "main_category": meta.get("main_category", "") if meta else "",
            }
        render_query_card(preview_meta)

        st.markdown("---")

        if st.button("🚀  Recommander + Expliquer", key="hy_run", type="primary"):
            with st.spinner("Calcul des recommandations hybrides et des explications SHAP…"):
                try:
                    data = api_get(
                        "/amazon/explain-recommendation",
                        params={
                            "item_idx": int(item_idx_h),
                            "k": int(k_h),
                            "pool": int(pool_h),
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "gamma": float(gamma_val),
                        },
                    )
                except requests.HTTPError as e:
                    st.error(f"Erreur API : {e}")
                    data = None
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    data = None

                if data:
                    explanations = data.get("explanations") or data.get("candidates") or []
                    st.markdown(f"### Résultats — {len(explanations)} recommandations")
                    render_hybrid_cards(explanations, query_idx=int(item_idx_h), cols_per_row=3)

                    # ── summary bar chart across all candidates ──
                    if _HAS_PLOTLY and explanations:
                        st.divider()
                        st.markdown("#### 📊 Vue d'ensemble — Répartition des signaux (en %)")
                        chart_rows = []
                        for ex in explanations:
                            feats = ex.get("features", {})
                            sh = _compute_shares(feats)
                            m = item_meta(int(ex["item_idx"]))
                            lbl = (m.get("title", "")[:30] if m else "") or f"id {ex['item_idx']}"
                            chart_rows.append(
                                {
                                    "Produit": lbl,
                                    "🖼️ Image (CLIP)": round(sh["image"], 1),
                                    "👥 Utilisateur (ALS)": round(sh["als"], 1),
                                    "📈 Popularité": round(sh["pop"], 1),
                                }
                            )
                        df_overview = pd.DataFrame(chart_rows)
                        fig2 = px.bar(
                            df_overview,
                            x="Produit",
                            y=["🖼️ Image (CLIP)", "👥 Utilisateur (ALS)", "📈 Popularité"],
                            barmode="stack",
                            color_discrete_sequence=["#6366f1", "#f97316", "#10b981"],
                            height=380,
                            labels={"value": "Part (%)", "variable": "Signal"},
                        )
                        fig2.update_layout(
                            legend_title_text="Signal",
                            xaxis_tickangle=-30,
                            yaxis_title="Répartition (%)",
                            margin=dict(l=20, r=20, t=30, b=80),
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                    # ── Counterfactual analysis ──
                    st.divider()
                    st.markdown("#### 🔄 Analyse Contrefactuelle")
                    st.caption(
                        "Que se passerait-il si on retirait un signal ? "
                        "Montre l'impact de chaque signal sur le classement de chaque produit."
                    )
                    try:
                        cf_data = api_get(
                            "/amazon/counterfactual",
                            params={
                                "item_idx": int(item_idx_h),
                                "k": int(k_h),
                                "pool": int(pool_h),
                                "alpha": float(alpha),
                                "beta": float(beta),
                                "gamma": float(gamma_val),
                                "filter_category": bool(filter_cat),
                            },
                        )
                    except Exception:
                        cf_data = None

                    if cf_data and cf_data.get("counterfactuals"):
                        for cf in cf_data["counterfactuals"]:
                            title = cf.get("title", f"Produit {cf['item_idx']}")[:60]
                            with st.expander(f"#{cf['original_rank']} — {title} (score {cf['original_score']:.3f})"):
                                st.markdown(f"**💡 {cf['summary_fr']}**")
                                cols_cf = st.columns(3)
                                scenarios = cf.get("scenarios", {})
                                labels_map = {
                                    "sans_image": ("🖼️ Sans Image", "similarité visuelle"),
                                    "sans_als": ("👥 Sans ALS", "filtrage collaboratif"),
                                    "sans_popularite": ("📈 Sans Popularité", "popularité"),
                                }
                                for i, (skey, (icon_label, _)) in enumerate(labels_map.items()):
                                    s = scenarios.get(skey, {})
                                    delta = s.get("rank_delta", 0)
                                    with cols_cf[i]:
                                        if delta > 0:
                                            st.metric(icon_label, f"Rang {s.get('new_rank', '?')}", f"+{delta} rangs", delta_color="inverse")
                                        elif delta < 0:
                                            st.metric(icon_label, f"Rang {s.get('new_rank', '?')}", f"{delta} rangs", delta_color="inverse")
                                        else:
                                            st.metric(icon_label, f"Rang {s.get('new_rank', '?')}", "Aucun changement")

    # ────────────────── TAB 2 : IMAGE SIMILARITY ───────────────
    with tab_image:
        st.subheader("Similarité Visuelle (CLIP)")
        st.caption("Trouve les produits visuellement les plus proches via embeddings CLIP + KNN cosine.")

        choice_i = st.selectbox("📦 Sélectionner un produit", label_list, index=0, key="img_sel")
        chosen_i = label_map.get(choice_i, {})
        default_img_idx = int(chosen_i.get("item_idx", 0))

        c1, c2, c3 = st.columns(3)
        with c1:
            item_idx_i = st.number_input("item_idx", min_value=0, value=default_img_idx, step=1, key="img_idx")
        with c2:
            k_i = st.slider("Nombre de résultats (k)", 1, 50, 10, key="img_k")
        with c3:
            pool_i = st.slider("Taille du pool", 10, 5000, 200, 10, key="img_pool")
        filter_cat_i = st.checkbox("Filtrer par catégorie", value=True, key="img_fc")

        # query preview — reacts to manual item_idx override
        actual_img_idx = int(item_idx_i)
        if actual_img_idx == default_img_idx and chosen_i:
            img_preview = {"item_idx": actual_img_idx, "title": chosen_i.get("title", ""), "main_category": chosen_i.get("main_category", "")}
        else:
            meta_i = item_meta(actual_img_idx)
            img_preview = {
                "item_idx": actual_img_idx,
                "title": meta_i.get("title", f"Produit {actual_img_idx}") if meta_i else f"Produit {actual_img_idx}",
                "main_category": meta_i.get("main_category", "") if meta_i else "",
            }
        render_query_card(img_preview)

        st.markdown("---")
        if st.button("🔍  Rechercher les produits similaires", key="img_run"):
            with st.spinner("Recherche…"):
                try:
                    data = api_get(
                        "/amazon/similar-items",
                        params={"item_idx": int(item_idx_i), "k": int(k_i), "pool": int(pool_i), "filter_category": bool(filter_cat_i)},
                    )
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    data = None
                if data:
                    st.markdown("### Produits similaires")
                    render_simple_grid(data.get("recommendations", []), cols_count=5, score_key="score")

    # ────────────────── TAB 3 : USER ALS ───────────────────────
    with tab_user:
        st.subheader("Recommandation Utilisateur (ALS)")
        st.caption("Filtrage collaboratif — recommande des produits basés sur l'historique d'achat d'un utilisateur.")

        c1, c2, c3 = st.columns(3)
        with c1:
            user_idx = st.number_input("👤 user_idx", min_value=0, value=0, step=1, key="als_uid")
        with c2:
            k_u = st.slider("Nombre de résultats (k)", 1, 50, 10, key="als_k")
        with c3:
            pool_u = st.number_input("Taille du pool", min_value=10, value=2000, step=10, key="als_pool")

        st.markdown("---")

        if st.button("👤  Recommander pour cet utilisateur", key="als_run"):
            with st.spinner("Calcul…"):
                try:
                    data = api_get(
                        "/amazon/recommend-user",
                        params={"user_idx": int(user_idx), "k": int(k_u), "pool": int(pool_u)},
                    )
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    data = None
                if data:
                    st.markdown("### Recommandations")
                    render_simple_grid(data.get("recommendations", []), cols_count=5, score_key="score")

    # ────────────────── TAB 4 : GLOBAL EXPLANATIONS ────────────
    with tab_global:
        st.subheader("🌍 Explications Globales du Modèle")
        st.caption(
            "Vue d'ensemble du comportement du modèle : importance des features, "
            "confiance des prédictions, et patterns du dataset."
        )

        n_samples_g = st.slider("Nombre d'échantillons pour la confiance", 500, 5000, 2000, 100, key="glob_n")

        if st.button("📊  Charger les explications globales", key="glob_run", type="primary"):
            with st.spinner("Calcul des explications globales…"):
                try:
                    gdata = api_get("/amazon/global-explanations", params={"n_samples": int(n_samples_g)})
                except Exception as e:
                    st.error(f"Erreur : {e}")
                    gdata = None

                if gdata:
                    # ── 1. Feature Importance ──
                    fi = gdata.get("feature_importance")
                    if fi:
                        st.markdown("### 📌 Importance Globale des Features")
                        st.info(fi.get("description_fr", ""))

                        if _HAS_PLOTLY:
                            labels = fi["feature_names"]
                            values = [round(v * 100, 2) for v in fi["importances"]]
                            fig_fi = go.Figure(go.Bar(
                                x=values, y=labels, orientation="h",
                                marker_color=["#6366f1", "#f97316", "#10b981"],
                                text=[f"{v}%" for v in values],
                                textposition="outside",
                            ))
                            fig_fi.update_layout(
                                xaxis_title="Importance (%)",
                                yaxis_title="Feature",
                                height=250,
                                margin=dict(l=20, r=20, t=10, b=30),
                            )
                            st.plotly_chart(fig_fi, use_container_width=True)
                        else:
                            for fname, imp in zip(fi["feature_names"], fi["importances"]):
                                st.write(f"- **{fname}** : {imp * 100:.2f}%")

                    st.divider()

                    # ── 2. Confidence ──
                    conf = gdata.get("confidence")
                    if conf:
                        st.markdown("### 🎯 Distribution de Confiance du Modèle")
                        st.info(conf.get("description_fr", ""))

                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("Confiance moyenne", f"{conf['mean']:.1%}")
                        mc2.metric("Confiance médiane", f"{conf['median']:.1%}")
                        mc3.metric("Écart-type", f"{conf['std']:.4f}")

                        if _HAS_PLOTLY:
                            bins = conf["histogram_bins"]
                            counts = conf["histogram_counts"]
                            bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(counts))]
                            fig_conf = go.Figure(go.Bar(
                                x=bin_labels, y=counts,
                                marker_color="#6366f1",
                            ))
                            fig_conf.update_layout(
                                xaxis_title="Niveau de confiance",
                                yaxis_title="Nombre de paires",
                                height=300,
                                margin=dict(l=20, r=20, t=10, b=50),
                            )
                            st.plotly_chart(fig_conf, use_container_width=True)

                    st.divider()

                    # ── 3. Data Patterns ──
                    dp = gdata.get("data_patterns")
                    if dp:
                        st.markdown("### 📊 Patterns du Dataset")
                        dc1, dc2, dc3, dc4 = st.columns(4)
                        dc1.metric("Articles", f"{dp['n_items']:,}")
                        dc2.metric("Utilisateurs", f"{dp['n_users']:,}")
                        dc3.metric("Interactions", f"{dp['n_interactions']:,}")
                        dc4.metric("Densité", f"{dp['density_pct']:.3f}%")

                        col_left, col_right = st.columns(2)
                        with col_left:
                            st.markdown("#### 🏷️ Top Catégories")
                            if dp.get("top_categories"):
                                cat_df = pd.DataFrame(dp["top_categories"])
                                cat_df.columns = ["Catégorie", "Nombre", "Part (%)"]
                                st.dataframe(cat_df, use_container_width=True, hide_index=True)

                        with col_right:
                            st.markdown("#### 🔥 Articles les Plus Populaires")
                            if dp.get("top_popular_items"):
                                pop_df = pd.DataFrame(dp["top_popular_items"])
                                pop_df.columns = ["item_idx", "Titre", "Score Pop."]
                                st.dataframe(pop_df, use_container_width=True, hide_index=True)

                        # Popularity score distribution
                        sd = dp.get("score_distribution", {})
                        if sd:
                            st.markdown("#### 📈 Distribution des Scores de Popularité")
                            sd1, sd2, sd3, sd4, sd5 = st.columns(5)
                            sd1.metric("Moyenne", f"{sd.get('mean', 0):.4f}")
                            sd2.metric("Médiane", f"{sd.get('median', 0):.4f}")
                            sd3.metric("Écart-type", f"{sd.get('std', 0):.4f}")
                            sd4.metric("Min", f"{sd.get('min', 0):.4f}")
                            sd5.metric("Max", f"{sd.get('max', 0):.4f}")

    # ── footer ──
    st.divider()
    st.caption(
        "Explainable Fashion Recommendation System — "
        "CLIP · ALS · SHAP · Counterfactual · Global · FastAPI · Streamlit"
    )


if __name__ == "__main__":
    main()
