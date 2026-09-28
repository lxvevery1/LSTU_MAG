#!/usr/bin/env python3
"""Write metrics.txt and predictions.json for the balanced 7k dataset into ./picke_7k/."""
import json
import os
from collections import Counter

import joblib
import numpy as np

DATA = "7k_posts.json"
MODEL = "some_model/logreg_model.joblib"
OUT_DIR = "picke_7k"
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]


def build_text(p):
    return " ".join(x for x in [(p.get("title") or "").strip(), (p.get("body") or "").strip()] if x).strip()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = json.load(open(DATA))
    years = data["years"]

    flat = [p for y in sorted(years) for p in years[y]]
    n = len(flat)

    # recompute classifier probabilities for the selected posts
    bundle = joblib.load(MODEL)
    feats, clf = bundle["features"], bundle["clf"]
    model_classes = [str(c) for c in clf.classes_]
    P = clf.predict_proba(feats.transform([build_text(p) for p in flat]))

    for i, p in enumerate(flat):
        p["probs"] = {c: round(float(P[i, model_classes.index(c)]), 4) for c in CLASSES}
        p["confidence"] = round(float(max(P[i])), 4)
        p["model_top"] = model_classes[int(np.argmax(P[i]))]

    dist = Counter(p["label"] for p in flat)

    # ---------------- predictions.json ----------------
    results = {
        "model": MODEL,
        "classes": CLASSES,
        "note": ("Labels were assigned by class-balanced selection: classifier probability per class, "
                 "per-class quality rules and Reddit flair re-ranking, balanced across the 12 months of "
                 "each year. Pseudo-labels, not human ground truth, so no accuracy vs truth is reported."),
        "total_posts": n,
        "class_distribution": {c: dist.get(c, 0) for c in CLASSES},
        "per_year_counts": {},
        "per_year_distribution": {},
        "avg_p_assigned_class": {},
        "years": years,
    }
    for y in sorted(years):
        rows = years[y]
        c = Counter(p["label"] for p in rows)
        results["per_year_counts"][y] = {"total": len(rows)}
        results["per_year_distribution"][y] = {cl: c.get(cl, 0) for cl in CLASSES}
        results["avg_p_assigned_class"][y] = {
            cl: round(float(np.mean([p["probs"][cl] for p in rows if p["label"] == cl])), 3)
            for cl in CLASSES if any(p["label"] == cl for p in rows)
        }

    with open(os.path.join(OUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------------- metrics.txt ----------------
    L = []
    L.append("Balanced WSB dataset 7k_posts.json - class distribution report")
    L.append(f"Model used for ranking: {MODEL}")
    L.append("")
    L.append("NOTE: labels are pseudo-labels produced by class-balanced selection")
    L.append("      (classifier probability + per-class rules + flair re-ranking).")
    L.append("      They are NOT human ground truth, so accuracy/F1 vs truth are not reported.")
    L.append("")
    L.append(f"Total posts: {n}")
    L.append("")
    L.append("Overall class distribution:")
    for c in CLASSES:
        L.append(f"  {c:18s}: {dist.get(c, 0):5d}  ({dist.get(c, 0) / n:.2%})")
    L.append(f"  {'balance spread':18s}: {min(dist.values())} .. {max(dist.values())} "
             f"(target {n // 5})")
    L.append("")
    L.append("Per-year totals:")
    for y in sorted(years):
        L.append(f"  {y}: {len(years[y])}")
    L.append("")
    L.append("Per-year distribution:")
    L.append("  " + "year".ljust(6) + "".join(c[:11].rjust(13) for c in CLASSES) + "total".rjust(8))
    for y in sorted(years):
        c = Counter(p["label"] for p in years[y])
        L.append("  " + y.ljust(6) + "".join(str(c.get(cl, 0)).rjust(13) for cl in CLASSES)
                 + str(len(years[y])).rjust(8))
    L.append("  " + "ALL".ljust(6) + "".join(str(dist.get(cl, 0)).rjust(13) for cl in CLASSES)
             + str(n).rjust(8))
    L.append("")
    L.append("Average classifier probability of the assigned class (selection confidence):")
    L.append("  " + "year".ljust(6) + "".join(c[:11].rjust(13) for c in CLASSES))
    for y in sorted(years):
        vals = results["avg_p_assigned_class"][y]
        L.append("  " + y.ljust(6) + "".join(f"{vals.get(cl, 0):.2f}".rjust(13) for cl in CLASSES))
    L.append("")
    L.append("Agreement between assigned label and classifier argmax:")
    agree = sum(1 for p in flat if p["model_top"] == p["label"])
    L.append(f"  {agree}/{n} ({agree / n:.2%})  remainder filled from the runner-up class")
    L.append("")
    L.append("Quality caveats:")
    L.append("  - FINANCIAL_CLAIM and UNCERTAIN have low selection confidence (~0.25); they were")
    L.append("    filled from the tail of the distribution to reach the per-class quota, so they")
    L.append("    are the noisiest classes here.")
    L.append("  - 2021 is short (90 FINANCIAL_CLAIM, 96 ANALYSIS_DD): Arctic Shift returns a body")
    L.append("    for only 236 of 3030 2021 posts, and ANALYSIS_DD requires real reasoning text.")
    L.append("  - ANALYSIS_DD requires a non-empty body, so title-only posts never enter that class.")

    with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\nsaved to ./{OUT_DIR}/metrics.txt and ./{OUT_DIR}/predictions.json")


if __name__ == "__main__":
    main()
