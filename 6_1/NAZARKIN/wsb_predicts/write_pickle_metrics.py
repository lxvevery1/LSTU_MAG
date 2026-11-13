#!/usr/bin/env python3
"""Write metrics.txt and predictions.json for the 8k classified posts into ./pickle/."""
import os
import pickle
from collections import Counter, defaultdict

SRC = "posts_8k_labeled.pkl"
OUT_DIR = "pickle"
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = pickle.load(open(SRC, "rb"))
    years = payload["years"]

    # flat list of posts
    all_posts = [p for y in sorted(years) for p in years[y]]
    n = len(all_posts)

    dist = Counter(p["label"] for p in all_posts)
    confs = [p.get("confidence", 0.0) for p in all_posts]
    low = sum(1 for c in confs if c < 0.5)

    # ---------- predictions.json ----------
    results = {
        "model": payload.get("model", "some_model/logreg_model.joblib"),
        "classes": CLASSES,
        "note": "Labels are model predictions (pseudo-labels), not ground truth. No accuracy is computed.",
        "total_posts": n,
        "class_distribution": {c: dist.get(c, 0) for c in CLASSES},
        "low_confidence_count": low,
        "low_confidence_ratio": round(low / n, 4) if n else 0,
        "per_year_counts": {},
        "per_year_distribution": {},
        "years": {},
    }
    for y in sorted(years):
        posts = years[y]
        c = Counter(p["label"] for p in posts)
        results["per_year_counts"][y] = {"total": len(posts)}
        results["per_year_distribution"][y] = {cl: c.get(cl, 0) for cl in CLASSES}
        results["years"][y] = posts  # already contain label + probs

    json_path = os.path.join(OUT_DIR, "predictions.json")
    import json as _json
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------- metrics.txt ----------
    lines = []
    lines.append("Classification of ~8k WallStreetBets posts (1000/year, 2019-2026)")
    lines.append(f"Model: {results['model']}")
    lines.append("NOTE: labels are model predictions (pseudo-labels), not ground truth.")
    lines.append("      Therefore accuracy/F1 vs truth are NOT reported here.")
    lines.append("")
    lines.append(f"Total posts: {n}")
    lines.append(f"Low confidence (<0.5): {low} ({low / n:.1%})" if n else "")
    lines.append("")
    lines.append("Overall class distribution:")
    for c in CLASSES:
        cnt = dist.get(c, 0)
        lines.append(f"  {c:18s}: {cnt:5d}  ({cnt / n:.2%})")
    lines.append("")
    lines.append("Per-year totals:")
    for y in sorted(years):
        lines.append(f"  {y}: {results['per_year_counts'][y]['total']}")
    lines.append("")
    lines.append("Per-year distribution:")
    for y in sorted(years):
        d = results["per_year_distribution"][y]
        parts = "  ".join(f"{c.split('_')[0]}={d[c]}" for c in CLASSES)
        lines.append(f"  {y}: {parts}")
    lines.append("")

    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nsaved to ./{OUT_DIR}/metrics.txt and ./{OUT_DIR}/predictions.json")


if __name__ == "__main__":
    main()
