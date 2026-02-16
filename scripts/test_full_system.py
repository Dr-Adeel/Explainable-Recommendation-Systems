"""
Test complet du système de recommandation.
Vérifie : API, pourcentages, formule hybride, SHAP, CLIP, ALS, cohérence.
"""

import math
import sys
import requests

API = "http://127.0.0.1:8001"
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  —  {detail}")


def _sigmoid(x, scale=30.0):
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x / scale))


def _compute_shares(features):
    img_v = max(float(features.get("multimodal_cosine", 0)), 0)
    raw_als = float(features.get("als_dot", 0))
    pop_v = max(float(features.get("popularity", 0)), 0)
    als_v = _sigmoid(raw_als)
    total = img_v + als_v + pop_v
    if total < 1e-12:
        return {"image": 0, "als": 0, "pop": 0}
    return {
        "image": img_v / total * 100,
        "als": als_v / total * 100,
        "pop": pop_v / total * 100,
    }


# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 1 — API Health")
print("=" * 60)

r = requests.get(f"{API}/health")
h = r.json()
check("Status 200", r.status_code == 200)
check("status == ok", h.get("status") == "ok")
check("CLIP ready", h.get("clip_ready") is True)
check("ALS ready", h.get("als_ready") is True)
check("items_total > 0", h.get("items_total", 0) > 0)
check("clip_rows > 0", h.get("clip_rows", 0) > 0)
check("als_users > 0", h.get("als_users", 0) > 0)

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 2 — Endpoint /amazon/item/{id}")
print("=" * 60)

for iid in [0, 100, 500, 2000]:
    r = requests.get(f"{API}/amazon/item/{iid}")
    data = r.json()
    check(f"item {iid} returns data", r.status_code == 200 and "item_idx" in data, str(data)[:80])
    if r.status_code == 200:
        check(f"item {iid} has title", bool(data.get("title")), "title missing")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 3 — Endpoint /amazon/sample-items")
print("=" * 60)

r = requests.get(f"{API}/amazon/sample-items", params={"n": 10})
data = r.json()
items = data.get("items", []) if isinstance(data, dict) else data
check("sample returns items", isinstance(items, list))
check("sample has 10 items", len(items) == 10, f"got {len(items)}")
check("each item has item_idx", all("item_idx" in it for it in items))

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 4 — Endpoint /amazon/similar-items (CLIP)")
print("=" * 60)

for iid in [100, 500]:
    r = requests.get(f"{API}/amazon/similar-items", params={"item_idx": iid, "k": 5})
    data = r.json()
    recs = data.get("recommendations", [])
    check(f"CLIP similar for {iid}: status 200", r.status_code == 200)
    check(f"CLIP similar for {iid}: got results", len(recs) > 0, f"got {len(recs)}")
    if recs:
        scores = [rec.get("score", 0) for rec in recs]
        check(f"CLIP similar for {iid}: scores in [0,1]",
              all(0 <= s <= 1.01 for s in scores),
              f"scores={scores}")
        check(f"CLIP similar for {iid}: sorted desc",
              scores == sorted(scores, reverse=True),
              f"scores={scores}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 5 — Endpoint /amazon/recommend-user (ALS)")
print("=" * 60)

for uid in [0, 10, 100]:
    r = requests.get(f"{API}/amazon/recommend-user", params={"user_idx": uid, "k": 5})
    data = r.json()
    recs = data.get("recommendations", [])
    check(f"ALS user {uid}: status 200", r.status_code == 200)
    check(f"ALS user {uid}: got results", len(recs) > 0, f"got {len(recs)}")
    if recs:
        scores = [rec.get("score", 0) for rec in recs]
        check(f"ALS user {uid}: sorted desc", scores == sorted(scores, reverse=True), f"scores={scores}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 6 — Endpoint /amazon/recommend-hybrid")
print("=" * 60)

for iid in [100, 500, 2000]:
    r = requests.get(f"{API}/amazon/recommend-hybrid", params={
        "item_idx": iid, "k": 5, "alpha": 0.5, "beta": 0.4, "gamma": 0.1
    })
    data = r.json()
    recs = data.get("recommendations", [])
    check(f"Hybrid for {iid}: status 200", r.status_code == 200)
    check(f"Hybrid for {iid}: got results", len(recs) > 0, f"got {len(recs)}")
    if recs:
        scores = [rec.get("score", 0) for rec in recs]
        check(f"Hybrid for {iid}: sorted desc", scores == sorted(scores, reverse=True), f"scores={scores}")
        # no self-recommendation
        item_ids = [rec.get("item_idx") for rec in recs]
        check(f"Hybrid for {iid}: no self-reco", iid not in item_ids, f"found self in results")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 7 — Endpoint /amazon/explain-recommendation (SHAP)")
print("=" * 60)

for iid in [100, 500, 2000]:
    r = requests.get(f"{API}/amazon/explain-recommendation", params={"item_idx": iid, "k": 4})
    data = r.json()
    exps = data.get("explanations", data.get("candidates", []))
    check(f"Explain for {iid}: status 200", r.status_code == 200)
    check(f"Explain for {iid}: got explanations", len(exps) > 0, f"got {len(exps)}")

    for ex in exps:
        cid = ex.get("item_idx")
        feat = ex.get("features", {})
        contribs = ex.get("contribs", {})

        # features must exist
        check(f"  item {cid}: has features", bool(feat), "features missing")
        check(f"  item {cid}: has multimodal_cosine",
              "multimodal_cosine" in feat,
              f"keys={list(feat.keys())}")
        check(f"  item {cid}: has als_dot", "als_dot" in feat)
        check(f"  item {cid}: has popularity", "popularity" in feat)

        # cosine must be in [0, 1]
        cos_v = float(feat.get("multimodal_cosine", -1))
        check(f"  item {cid}: cosine in [0,1]", 0 <= cos_v <= 1.01, f"got {cos_v}")

        # popularity must be in [0, 1]
        pop_v = float(feat.get("popularity", -1))
        check(f"  item {cid}: popularity in [0,1]", 0 <= pop_v <= 1.01, f"got {pop_v}")

        # surrogate pred exists
        pred = ex.get("surrogate_pred")
        check(f"  item {cid}: has surrogate_pred", pred is not None, "missing")

        # contribs must have features list
        c_feats = contribs.get("features", [])
        c_vals = contribs.get("shap_values") or contribs.get("contributions", [])
        check(f"  item {cid}: contribs has features", len(c_feats) == 3, f"got {len(c_feats)}")
        check(f"  item {cid}: contribs has values", len(c_vals) == 3, f"got {len(c_vals)}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 8 — Cohérence des pourcentages (frontend)")
print("=" * 60)

r = requests.get(f"{API}/amazon/explain-recommendation", params={"item_idx": 100, "k": 6})
exps = r.json().get("explanations", r.json().get("candidates", []))

for ex in exps:
    cid = ex.get("item_idx")
    feat = ex.get("features", {})
    shares = _compute_shares(feat)

    img_pct = shares["image"]
    als_pct = shares["als"]
    pop_pct = shares["pop"]
    total_pct = img_pct + als_pct + pop_pct

    # sum must be ~100%
    check(f"item {cid}: sum = {total_pct:.1f}%", abs(total_pct - 100) < 0.5, f"got {total_pct}")

    # all must be >= 0
    check(f"item {cid}: all >= 0", img_pct >= 0 and als_pct >= 0 and pop_pct >= 0,
          f"img={img_pct:.1f} als={als_pct:.1f} pop={pop_pct:.1f}")

    # no single signal > 90% (balanced after sigmoid fix)
    check(f"item {cid}: no signal > 90%",
          img_pct < 90 and als_pct < 90 and pop_pct < 90,
          f"img={img_pct:.1f}% als={als_pct:.1f}% pop={pop_pct:.1f}%")

    # image should be meaningful (> 5%) for visually similar items
    cos_v = float(feat.get("multimodal_cosine", 0))
    if cos_v > 0.5:
        check(f"item {cid}: image > 5% when cosine={cos_v:.2f}",
              img_pct > 5,
              f"img_pct={img_pct:.1f}%")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 9 — Raison française cohérente")
print("=" * 60)


def _build_reason_fr(shares):
    img_pct = shares["image"]
    als_pct = shares["als"]
    pop_pct = shares["pop"]
    parts = sorted(
        [("image", img_pct), ("als", als_pct), ("pop", pop_pct)],
        key=lambda x: x[1], reverse=True
    )
    top_name, top_val = parts[0]
    second_name, second_val = parts[1]
    if top_val < 1:
        return "Recommandé sur la base de signaux combinés."
    if top_name == "image":
        reason = f"Produit visuellement similaire (similarité image : {img_pct:.0f} %)"
    elif top_name == "als":
        reason = f"Les acheteurs de ce produit ont aussi acheté cet article (signal utilisateur : {als_pct:.0f} %)"
    else:
        reason = f"Article populaire (popularité : {pop_pct:.0f} %)"
    return reason + "."


for ex in exps:
    cid = ex.get("item_idx")
    feat = ex.get("features", {})
    shares = _compute_shares(feat)
    reason = _build_reason_fr(shares)

    # determine dominant signal
    dominant = max(shares, key=shares.get)
    if dominant == "image":
        check(f"item {cid}: reason mentions 'image'", "image" in reason.lower(), reason[:60])
    elif dominant == "als":
        check(f"item {cid}: reason mentions 'utilisateur/acheté'",
              "utilisateur" in reason.lower() or "acheté" in reason.lower(), reason[:60])
    else:
        check(f"item {cid}: reason mentions 'populaire'", "populaire" in reason.lower(), reason[:60])

    # reason must match the actual top percentage
    top_pct = max(shares.values())
    pct_str = f"{top_pct:.0f}"
    check(f"item {cid}: reason has correct %", pct_str in reason, f"expected {pct_str}% in: {reason[:60]}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 10 — Vérification croisée hybrid vs explain")
print("=" * 60)

# same query should give same top items
r_hyb = requests.get(f"{API}/amazon/recommend-hybrid", params={"item_idx": 100, "k": 4})
r_exp = requests.get(f"{API}/amazon/explain-recommendation", params={"item_idx": 100, "k": 4})

hyb_ids = [r.get("item_idx") for r in r_hyb.json().get("recommendations", [])]
exp_ids = [e.get("item_idx") for e in r_exp.json().get("explanations", r_exp.json().get("candidates", []))]

check("hybrid and explain give same items", set(hyb_ids) == set(exp_ids),
      f"hybrid={hyb_ids}  explain={exp_ids}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 11 — Sensibilité aux poids (alpha/beta/gamma)")
print("=" * 60)

# alpha=1 → image only
r1 = requests.get(f"{API}/amazon/recommend-hybrid", params={"item_idx": 100, "k": 5, "alpha": 1.0, "beta": 0.0, "gamma": 0.0})
# beta=1 → ALS only
r2 = requests.get(f"{API}/amazon/recommend-hybrid", params={"item_idx": 100, "k": 5, "alpha": 0.0, "beta": 1.0, "gamma": 0.0})

ids_img = [r.get("item_idx") for r in r1.json().get("recommendations", [])]
ids_als = [r.get("item_idx") for r in r2.json().get("recommendations", [])]

check("alpha=1 vs beta=1 gives different results", ids_img != ids_als,
      f"both gave {ids_img}")

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 12 — Image endpoint")
print("=" * 60)

for iid in [100, 500]:
    r = requests.get(f"{API}/amazon/image/{iid}")
    check(f"image {iid}: status 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        check(f"image {iid}: content-type image", "image" in r.headers.get("content-type", ""),
              r.headers.get("content-type"))

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 13 — Feedback endpoint")
print("=" * 60)

r = requests.post(f"{API}/amazon/feedback", json={"query_item": 100, "candidate_item": 200, "helpful": True})
check("feedback POST: status 200", r.status_code == 200)
check("feedback POST: returns ok", r.json().get("status") in ["ok", "saved"], str(r.json()))

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TEST 14 — Edge cases")
print("=" * 60)

# item that doesn't exist
r = requests.get(f"{API}/amazon/item/999999")
check("non-existent item: 404", r.status_code == 404)

# user that doesn't exist (should still return 200 with empty or fallback)
r = requests.get(f"{API}/amazon/recommend-user", params={"user_idx": 999999, "k": 5})
check("non-existent user: handled gracefully", r.status_code in [200, 404, 422])

# ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  RÉSULTATS : {PASS} ✅  {FAIL} ❌")
print("=" * 60)

if FAIL > 0:
    print(f"\n⚠️  {FAIL} tests échoués — à corriger !")
    sys.exit(1)
else:
    print("\n🎉  Tous les tests passent — le système est cohérent !")
    sys.exit(0)
