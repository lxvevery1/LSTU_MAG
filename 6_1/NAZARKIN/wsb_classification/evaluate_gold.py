#!/usr/bin/env python3
"""Evaluate the 4-class and 5-class LogReg models on the hand-labelled gold set.

gold_candidates.json : independent candidate posts (never used for training)
gold_labels.json     : labels assigned by reading post text only (LLM/manual proxy)

Produces ./gold_eval/ with metrics.json, report.txt and predictions.json.
"""
import json
import os
from collections import Counter, defaultdict

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CANDIDATES = "gold_candidates.json"
LABELS = "gold_labels.json"
OUT_DIR = "gold_eval"
CLASSES4 = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]
CLASSES5 = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]
MODEL4 = os.path.join("some_model_4c", "logreg_model.joblib")
MODEL5 = os.path.join("some_model", "logreg_model.joblib")


def post_text(p):
    return " ".join(x for x in [(p.get("title") or ""), (p.get("body") or "")] if x).strip()


def load_gold():
    cands = json.load(open(CANDIDATES, encoding="utf-8"))
    labels = json.load(open(LABELS, encoding="utf-8"))
    rows, skipped = [], []
    for p in cands:
        lab = labels.get(p["id"])
        if lab is None:
            raise SystemExit(f"unlabelled candidate: {p['id']}")
        row = {"post": p, "text": post_text(p), "label": lab, "hint": p.get("hint_class"),
               "year": p.get("year"), "flair": p.get("flair")}
        if lab == "UNCERTAIN" or not row["text"]:
            skipped.append(row)
        else:
            rows.append(row)
    return rows, skipped


def predict(bundle, rows):
    X = bundle["features"].transform([r["text"] for r in rows])
    clf = bundle["clf"]
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)
        conf = [float(probs[i].max()) for i in range(probs.shape[0])]
    else:
        conf = [None] * len(rows)
    preds = [str(p) for p in clf.predict(X)]
    return preds, conf


def evaluate(name, bundle, rows, classes, subset=None):
    preds, conf = predict(bundle, rows)
    y = [r["label"] for r in rows]
    res = {
        "subset": subset or "all",
        "model": name,
        "n": len(rows),
        "accuracy": round(accuracy_score(y, preds), 4),
        "macro_f1": round(f1_score(y, preds, average="macro", zero_division=0), 4),
        "class_distribution": dict(Counter(y)),
        "predicted_distribution": dict(Counter(preds)),
        "per_class": {},
        "per_year_accuracy": {},
        "per_hint_accuracy": {},
    }
    rep = classification_report(y, preds, labels=classes, target_names=classes, zero_division=0,
                                output_dict=True)
    for c in classes:
        res["per_class"][c] = {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in rep[c].items()}
    cm = confusion_matrix(y, preds, labels=classes)
    res["confusion_matrix"] = {"labels": classes, "rows_true": cm.tolist()}

    by_year = defaultdict(lambda: [0, 0])
    by_hint = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        ok = int(p == r["label"])
        by_year[r["year"]][0] += ok
        by_year[r["year"]][1] += 1
        by_hint[r["hint"]][0] += ok
        by_hint[r["hint"]][1] += 1
    res["per_year_accuracy"] = {k: {"accuracy": round(v[0] / v[1], 4), "correct": v[0], "total": v[1]}
                               for k, v in sorted(by_year.items())}
    res["per_hint_accuracy"] = {k: {"accuracy": round(v[0] / v[1], 4), "correct": v[0], "total": v[1]}
                               for k, v in sorted(by_hint.items())}
    by_flair = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        fl = r["flair"] or "-"
        by_flair[fl][0] += int(p == r["label"])
        by_flair[fl][1] += 1
    res["per_flair_accuracy"] = {k: {"accuracy": round(v[0] / v[1], 4), "correct": v[0], "total": v[1]}
                               for k, v in sorted(by_flair.items(), key=lambda kv: -kv[1][1])
                               if v[1] >= 5}
    conf_err = defaultdict(lambda: [0, 0])
    for r, p, c in zip(rows, preds, conf):
        if c is None:
            continue
        b = min(int(c * 10) / 10, 0.9)
        conf_err[b][0] += int(p == r["label"])
        conf_err[b][1] += 1
    res["accuracy_by_confidence_bin"] = {
        f"{b:.1f}-{b + 0.1:.1f}": {"accuracy": round(v[0] / v[1], 4), "correct": v[0], "total": v[1]}
        for b, v in sorted(conf_err.items())}
    res["confusion_pairs"] = {f"{g}->{p}": n for (g, p), n in
                              Counter((r["label"], p) for r, p in zip(rows, preds)
                                      if r["label"] != p).most_common(10)}
    res["text_report"] = classification_report(y, preds, labels=classes, target_names=classes,
                                              zero_division=0)
    res["_preds"] = preds
    res["_conf"] = conf
    res["_rows"] = rows
    return res


def strip_private(res):
    return {k: v for k, v in res.items() if not k.startswith("_")}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, skipped = load_gold()
    print(f"gold candidates: {len(json.load(open(CANDIDATES)))}")
    print(f"usable (labelled, non-empty, non-UNCERTAIN): {len(rows)}")
    print(f"skipped: {len(skipped)}  ({dict(Counter(r['label'] for r in skipped))})")
    print("usable label distribution:", dict(Counter(r["label"] for r in rows)))

    nat = [r for r in rows if r["hint"] == "NATURAL"]
    print(f"natural-distribution subset: {len(nat)}")

    bundle4 = joblib.load(MODEL4)
    bundle5 = joblib.load(MODEL5) if os.path.exists(MODEL5) else None

    results = {
        "note": ("Gold labels were assigned by reading post text only (LLM/manual proxy), "
                 "not by humans; treat these numbers as an independent reference estimate, "
                 "not absolute ground truth."),
        "candidates_file": CANDIDATES,
        "labels_file": LABELS,
        "candidates_total": len(json.load(open(CANDIDATES))),
        "usable": len(rows),
        "skipped": len(skipped),
        "skipped_reasons": dict(Counter(r["label"] for r in skipped)),
        "usable_label_distribution": dict(Counter(r["label"] for r in rows)),
        "natural_subset_size": len(nat),
        "evaluations": [],
    }

    combos = [("4class_model", bundle4, CLASSES4, rows, "all"),
              ("4class_model", bundle4, CLASSES4, nat, "natural_only"),
              ("5class_model", bundle5, CLASSES5, rows, "all")]
    details = []
    for name, bundle, classes, sub, subset in combos:
        if bundle is None:
            continue
        r = evaluate(name, bundle, sub, classes, subset)
        details.append(r)
        results["evaluations"].append(strip_private(r))
        print(f"\n=== {name} / {subset} (n={r['n']}) ===")
        print(f"accuracy: {r['accuracy']}  macro-F1: {r['macro_f1']}")
        print(r["text_report"])

    preds_out = []
    for d in details:
        for r, p, c in zip(d["_rows"], d["_preds"], d["_conf"]):
            preds_out.append({"id": r["post"]["id"], "subset": d["subset"], "model": d["model"],
                              "gold": r["label"], "predicted": p,
                              "hint": r["hint"], "year": r["year"], "flair": r["flair"],
                              "confidence": None if c is None else round(c, 4),
                              "title": (r["post"].get("title") or "")[:120]})
    with open(os.path.join(OUT_DIR, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(preds_out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = ["Gold set evaluation (independent, model was not used to select or label posts)", ""]
    lines.append(f"Candidates: {results['candidates_total']}, usable: {results['usable']}, "
                 f"skipped: {results['skipped']} {results['skipped_reasons']}")
    lines.append(f"Usable gold label distribution: {results['usable_label_distribution']}")
    lines.append("")
    for d in details:
        lines.append("=" * 60)
        lines.append(f"{d['model']} / subset={d['subset']}  n={d['n']}")
        lines.append(f"ACCURACY: {d['accuracy']}   MACRO-F1: {d['macro_f1']}")
        lines.append("")
        lines.append(d["text_report"])
        lines.append("Confusion matrix (rows=true):")
        for lab, row in zip(d["confusion_matrix"]["labels"], d["confusion_matrix"]["rows_true"]):
            lines.append(f"  {lab:18s} {row}")
        lines.append("")
        lines.append("Per-year accuracy:")
        for k, v in d["per_year_accuracy"].items():
            lines.append(f"  {k}: {v['accuracy']} ({v['correct']}/{v['total']})")
        lines.append("")
        lines.append("Per-sampler-hint accuracy (sampling group, not gold label):")
        for k, v in d["per_hint_accuracy"].items():
            lines.append(f"  {k}: {v['accuracy']} ({v['correct']}/{v['total']})")
        lines.append("")
        lines.append("Per-flair accuracy (flair >= 5):")
        for k, v in d["per_flair_accuracy"].items():
            lines.append(f"  {k}: {v['accuracy']} ({v['correct']}/{v['total']})")
        lines.append("")
        lines.append("Accuracy by predicted probability:")
        for k, v in d["accuracy_by_confidence_bin"].items():
            lines.append(f"  {k}: {v['accuracy']} ({v['correct']}/{v['total']})")
        lines.append("")
        lines.append("Top confusions (true -> predicted):")
        for k, v in d["confusion_pairs"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    lines.append(results["note"])
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nsaved to ./" + OUT_DIR + "/")


if __name__ == "__main__":
    main()
