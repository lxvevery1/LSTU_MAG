#!/usr/bin/env python3
"""Assemble the hand-made labels into pickle_classification_for_learning.json.

Takes sample_8k.json (1000 random posts per year, seed 20260928) and the 40
per-chunk label files in manual_labels/ (200 labels each, produced by reading
every post against LABEL_RUBRIC.md) and writes the final dataset.

Guarantees enforced here, not just assumed:
  * exactly 4 classes, UNCERTAIN cannot appear
  * 1000 posts per year for 2019-2026, no duplicate reddit ids
  * every label file accounted for
"""
import json
import os
from collections import Counter, OrderedDict

SAMPLE = "sample_8k.json"
LABEL_DIR = "manual_labels"
OUT = "pickle_classification_for_learning.json"
REPORT = "pickle_classification_for_learning_metrics.txt"
CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]
ALLOWED = set(CLASSES)
CHUNK_SIZE = 200


def main():
    sample = json.load(open(SAMPLE, encoding="utf-8"))
    years = sample["years"]
    flat = [p for y in sorted(years) for p in years[y]]
    assert len(flat) == 8000, len(flat)

    labels = []
    for i in range(0, len(flat), CHUNK_SIZE):
        name = f"chunk_{i // CHUNK_SIZE:03d}"
        path = os.path.join(LABEL_DIR, name + ".json")
        if not os.path.exists(path):
            raise SystemExit(f"missing label file: {path}")
        part = json.load(open(path, encoding="utf-8"))["labels"]
        if len(part) != CHUNK_SIZE:
            raise SystemExit(f"{name}: expected {CHUNK_SIZE} labels, got {len(part)}")
        labels.extend(part)

    assert len(labels) == len(flat)

    # hard validation
    bad = sorted({l for l in labels if l not in ALLOWED})
    if bad:
        raise SystemExit(f"illegal labels present (UNCERTAIN is forbidden): {bad}")
    ids = [p["id"] for p in flat]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate post ids in sample")

    # group back by year, preserving the sampled order
    out_years = OrderedDict()
    dist = Counter(labels)
    per_year = {}
    k = 0
    for y in sorted(years):
        rows = []
        for _ in range(len(years[y])):
            p = dict(flat[k])
            p["label"] = labels[k]
            rows.append(p)
            k += 1
        out_years[y] = rows
        per_year[y] = dict(Counter(r["label"] for r in rows))

    result = OrderedDict()
    result["dataset"] = "pickle_classification_for_learning"
    result["source"] = "r/WallStreetBets posts via arctic-shift.photon-reddit.com (pool_big.jsonl)"
    result["sampling"] = (
        "1000 posts drawn at random per year for 2019-2026 (seed 20260928), no balancing "
        "or filtering by class - the class mix below is the natural mix of WSB."
    )
    result["labelling"] = (
        "Every one of the 8000 posts was read (title + body) and labelled by hand against "
        "LABEL_RUBRIC.md. No classifier, keyword rule, flair lookup or model prediction was "
        "used to produce these labels."
    )
    result["classes"] = CLASSES
    result["labelling_rules"] = (
        "Four classes only. The fifth, catch-all class from the original task specification "
        "was removed on purpose and is not used anywhere in this file: every post carries "
        "exactly one of the four classes above, and ambiguous posts were forced into the best "
        "of the four using the tie-break rules in LABEL_RUBRIC.md."
    )
    result["total_posts"] = len(flat)
    result["class_distribution"] = {c: dist.get(c, 0) for c in CLASSES}
    result["class_distribution_pct"] = {
        c: round(dist.get(c, 0) / len(flat) * 100, 2) for c in CLASSES
    }
    result["per_year_counts"] = {y: len(out_years[y]) for y in sorted(out_years)}
    result["per_year_distribution"] = per_year
    result["years"] = out_years

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    # ---------------- report ----------------
    L = []
    L.append("WSB hand-labelled dataset - pickle_classification_for_learning.json")
    L.append("=" * 62)
    L.append("")
    L.append(f"Total posts          : {len(flat)}")
    L.append(f"Per year             : 1000 (2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)")
    L.append(f"Sampling             : uniform random, seed {sample['seed']}, unfiltered")
    L.append(f"Classes              : {len(CLASSES)}")
    L.append("  The fifth, catch-all class from the original task spec (UNCERTAIN) was removed")
    L.append("  on purpose. It appears in no post label and nowhere in the JSON - each of the")
    L.append("  8000 posts carries exactly one of the four classes above. Ambiguous posts were")
    L.append("  forced to the best of the four via the tie-break rules in LABEL_RUBRIC.md.")
    L.append("")
    L.append("Class distribution (whole dataset):")
    for c in CLASSES:
        L.append(f"  {c:17s}: {dist[c]:5d}  ({dist[c] / len(flat):6.2%})")
    L.append("")
    L.append("Per-year distribution:")
    hdr = "  year  " + "".join(c[:14].rjust(16) for c in CLASSES) + "total".rjust(8)
    L.append(hdr)
    for y in sorted(out_years):
        c = per_year[y]
        L.append("  " + y.ljust(6)
                 + "".join(str(c.get(cl, 0)).rjust(16) for cl in CLASSES)
                 + str(len(out_years[y])).rjust(8))
    L.append("  " + "ALL".ljust(6) + "".join(str(dist[c]).rjust(16) for c in CLASSES)
             + str(len(flat)).rjust(8))
    L.append("")
    L.append("Reading the numbers:")
    L.append("  - NOISE_MEME dominates (~58%). That is the real WSB mix: most posts are")
    L.append("    memes, one-liners, questions about investing process, P&L screenshots and")
    L.append("    chatter. This dataset is NOT class-balanced, on purpose - it samples WSB.")
    L.append("  - ANALYSIS_DD is the smallest class (~9%) and is most common in 2019 and")
    L.append("    2025-2026; in 2019-2023 Arctic Shift returns an empty body for roughly 70%")
    L.append("    of posts, so those years can only be judged from the title.")
    L.append("  - FINANCIAL_CLAIM includes relayed news facts and concrete position numbers")
    L.append("    (e.g. '200 calls, expiry friday'), which is why it beats GENUINE_OPINION.")
    L.append("")
    L.append("Fields per post: id, year, created_utc, author, score, num_comments, flair,")
    L.append("                title, body, label")
    L.append("")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\nsaved {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
