import streamlit as st
import requests
import pandas as pd

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    px = None
    _HAS_PLOTLY = False

st.set_page_config(page_title="Explain Recommendations", layout="wide")

st.markdown(
    """
    <style>
    .card {border-radius:8px; padding:12px; border:1px solid #eee; box-shadow: 0 6px 18px rgba(15,15,15,0.06); margin:8px}
    .meta {color:#6b7280; font-size:13px; margin-bottom:6px}
    .reason {background:#f3f6ff; padding:8px; border-radius:6px; margin-top:8px; font-size:14px}
    .title {font-weight:600; font-size:16px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Explainable Recommendations")

base_url = st.text_input("API base URL", "http://127.0.0.1:8001")
col1, col2 = st.columns([1, 3])

with col1:
    def fetch_sample_items(base_url, n=100):
        try:
            r = requests.get(f"{base_url}/amazon/sample-items", params={"n": n, "with_images": True}, timeout=6)
            r.raise_for_status()
            return r.json().get("items", [])
        except Exception:
            return []

    sample_items = fetch_sample_items(base_url, n=100)
    sample_map = {f"{it.get('item_idx')} — {it.get('title','')[:60]}": it.get('item_idx') for it in sample_items}
    sel = st.selectbox("Pick from catalog (sample)", options=["Manual enter"] + list(sample_map.keys()))
    if sel and sel != "Manual enter":
        item_idx = int(sample_map[sel])
        st.text_input("Selected item_idx", value=str(item_idx), disabled=True)
    else:
        item_idx = st.number_input("Item index", min_value=0, value=1, step=1)

    st.markdown("**Blend weights (alpha=image, beta=als, gamma=pop)**")
    alpha = st.slider("alpha (image)", 0.0, 1.0, 0.5, step=0.01)
    beta = st.slider("beta (als)", 0.0, 1.0, 0.4, step=0.01)
    gamma = st.slider("gamma (pop)", 0.0, 1.0, 0.1, step=0.01)

    lang = st.selectbox("Language / Langue", options=["English", "Français"], index=1)

    k = st.slider("k (neighbors)", 1, 20, 5)
    pool = st.number_input("pool (candidate pool size)", min_value=10, value=500, step=10)
    filter_category = st.checkbox("filter_category (same main_category)", value=True)
    get_btn = st.button("Get explanations")


def fetch_explanations(base_url, item_idx, k, pool, alpha, beta, gamma, filter_category):
    try:
        params = {"item_idx": int(item_idx), "k": int(k), "pool": int(pool), "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma), "filter_category": bool(filter_category)}
        r = requests.get(f"{base_url}/amazon/explain-recommendation", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def fetch_item_meta(base_url, item_idx):
    try:
        r = requests.get(f"{base_url}/amazon/item/{item_idx}", timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def image_url(base_url, item_idx):
    return f"{base_url}/amazon/image/{item_idx}"


def send_feedback(base_url, query_item, candidate_item, helpful: bool, lang="en"):
    payload = {
        "query_item": int(query_item),
        "candidate_item": int(candidate_item),
        "helpful": bool(helpful),
        "lang": lang,
    }
    try:
        r = requests.post(f"{base_url}/amazon/feedback", json=payload, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def human_reason_from_contribs(contribs, lang="English"):
    feats = contribs.get("features") or []
    vals = contribs.get("contributions") or []
    if "shap_values" in contribs:
        vals = contribs.get("shap_values") or []
    if not feats or not vals:
        return "Recommended based on combined signals." if not lang.startswith("Fr") else "Recommandé sur la base de signaux combinés."
    arr = [abs(float(v)) for v in vals]
    s = sum(arr)
    shares = [v / s if s > 0 else 0 for v in arr]
    top_idx = int(max(range(len(shares)), key=lambda i: shares[i]))
    label = feats[top_idx]
    pct = int(round(shares[top_idx] * 100))
    if lang.startswith("Fr"):
        if label == "multimodal_cosine":
            return f"Produit visuellement similaire — la similarité d'image contribue à {pct}% du score."
        if label == "als_dot":
            return f"Les acheteurs de ce produit ont également acheté cet article — ce signal contribue à {pct}% du score."
        if label == "popularity":
            return f"Article populaire — la popularité contribue à {pct}% du score."
        return "Recommandation basée sur plusieurs signaux combinés."
    else:
        if label == "multimodal_cosine":
            return f"Visually similar product — image similarity contributes {pct}% to the score."
        if label == "als_dot":
            return f"People who bought this also bought that — user-item signal contributes {pct}% to the score."
        if label == "popularity":
            return f"Popular item — popularity contributes {pct}% to the score."
        return "Recommended based on combined signals."


def render_cards(items, cols_per_row=3, lang="English"):
    if not items:
        st.info("No recommendations available.")
        return
    # compute relative ALS percentage within this page of candidates
    als_vals = []
    for it in items:
        try:
            als_vals.append(float((it.get('features') or {}).get('als_dot', 0)))
        except Exception:
            als_vals.append(0.0)
    max_abs_als = max((abs(v) for v in als_vals), default=0.0)

    rows = [items[i: i + cols_per_row] for i in range(0, len(items), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for c, cand in zip(cols, row):
            cid = cand.get("item_idx") or cand.get("candidate_idx") or cand.get("id")
            pred = cand.get("surrogate_pred")
            features = cand.get("features") or {}
            contribs = cand.get("contribs") or {}

            with c:
                c.markdown("<div class='card'>", unsafe_allow_html=True)
                c.image(image_url(base_url, cid), width=260)
                m = fetch_item_meta(base_url, cid)
                title = m.get('title','(no title)') if m else f"Item {cid}"
                category = m.get('main_category','') if m else ''
                c.markdown(f"<div class='title'>{title}</div>", unsafe_allow_html=True)
                c.markdown(f"<div class='meta'>{category} — id: {cid}</div>", unsafe_allow_html=True)

                # reason (prefer server-provided French text when available)
                # prefer server-provided French text; normalize and fallback if empty
                reason_text = None
                if cand.get("reason_fr"):
                    reason_text = str(cand.get("reason_fr"))
                # if user selected English, prefer server English reason if available
                if not reason_text and not lang.startswith("Fr") and cand.get("reason"):
                    reason_text = str(cand.get("reason"))
                # fallback to local generator
                if not reason_text or not reason_text.strip():
                    reason_text = human_reason_from_contribs(contribs, lang=lang)
                # normalize whitespace/newlines and punctuation
                reason_text = reason_text.replace('\n', ' ').replace('\r', ' ').strip()
                if reason_text and not reason_text.endswith(('.', '!', '?')):
                    reason_text = reason_text + '.'
                c.markdown(f"<div class='reason'>{reason_text}</div>", unsafe_allow_html=True)

                # format metric values: show percentages for image similarity and popularity
                def fmt_percent_if_fraction(v):
                    try:
                        fv = float(v)
                    except Exception:
                        return "-"
                    if 0.0 <= fv <= 1.0:
                        return f"{fv * 100:.1f}%"
                    return f"{fv:.3f}"

                img_val = features.get('multimodal_cosine', 0)
                als_val = features.get('als_dot', 0)
                pop_val = features.get('popularity', 0)

                # display ALS as relative percentage (per-page) but show raw value in small text
                if max_abs_als > 0:
                    try:
                        als_pct = float(als_val) / max_abs_als * 100.0
                    except Exception:
                        als_pct = 0.0
                else:
                    als_pct = 0.0

                metrics = c.columns([1,1,1])
                metrics[0].metric("Image", fmt_percent_if_fraction(img_val))
                metrics[1].metric("ALS", f"{als_pct:.1f}%")
                metrics[2].metric("Pop", fmt_percent_if_fraction(pop_val))
                # small raw ALS value
                c.markdown(f"<div style='font-size:12px;color:#6b7280;margin-top:4px'>ALS raw: {als_val:.3f}</div>", unsafe_allow_html=True)

                foot_left, foot_right = c.columns([2,1])
                score_val = f"{pred:.3f}" if pred is not None else "-"
                foot_left.metric("Surrogate", score_val)
                if foot_right.button("👍", key=f"help_{cid}"):
                    ok = send_feedback(base_url, item_idx, cid, True, lang=lang[:2].lower())
                    if ok:
                        c.success("Merci — feedback enregistré")
                    else:
                        c.error("Échec de l'envoi")
                if foot_right.button("👎", key=f"no_{cid}"):
                    ok = send_feedback(base_url, item_idx, cid, False, lang=lang[:2].lower())
                    if ok:
                        c.success("Merci — feedback enregistré")
                    else:
                        c.error("Échec de l'envoi")

                c.markdown("</div>", unsafe_allow_html=True)


if get_btn:
    # validate item exists
    try:
        meta_check = requests.get(f"{base_url}/amazon/item/{int(item_idx)}", timeout=6)
        if meta_check.status_code == 404:
            st.error("item_idx not found in catalog — pick a valid item from the sample selector")
            st.stop()
        meta_check.raise_for_status()
    except Exception as e:
        st.error(f"Item validation failed: {e}")
        st.stop()

    data = fetch_explanations(base_url, item_idx, k, pool, alpha, beta, gamma, filter_category)
    if not data:
        st.stop()

    explanations = data.get("explanations") or data.get("candidates") or []

    # show query item
    meta = fetch_item_meta(base_url, item_idx)
    if meta:
        st.markdown("**Query item**")
        qcols = st.columns([1, 3])
        with qcols[0]:
            st.image(image_url(base_url, item_idx), width=160)
        with qcols[1]:
            st.markdown(f"**{meta.get('title','')}**")
            st.markdown(f"Category: {meta.get('main_category','')}")

    render_cards(explanations, cols_per_row=min(3, max(1, k)), lang=lang)
