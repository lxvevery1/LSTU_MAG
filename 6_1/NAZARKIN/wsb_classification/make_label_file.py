#!/usr/bin/env python3
"""Build an easy-to-label file (teachers_to_label.json) from teacher candidates.

Each post gets a "label" field set to null. Fill it with one of:
  NOISE_MEME, GENUINE_OPINION, FINANCIAL_CLAIM, ANALYSIS_DD, UNCERTAIN
"""
import json

IN = "teachers_candidates.json"
OUT = "teachers_to_label.json"

CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD", "UNCERTAIN"]

data = json.load(open(IN))
out = {
    "classes": CLASSES,
    "instructions": {
        "NOISE_MEME": "No meaningful financial/investment information. Pure joke, reaction, shitpost, image caption.",
        "GENUINE_OPINION": "Genuine bullish/bearish/trading opinion without substantial analysis.",
        "FINANCIAL_CLAIM": "Concrete factual claim about a company, market, event, price, earnings, metric.",
        "ANALYSIS_DD": "Detailed reasoning, DD, calculations, valuation, catalysts, risks, sources.",
        "UNCERTAIN": "Cannot confidently assign another label.",
    },
    "years": {},
}

for year, posts in data.items():
    items = []
    for p in posts:
        items.append({
            "id": p["id"],
            "created": p.get("created_utc"),
            "score": p.get("score"),
            "flair": p.get("link_flair_text"),
            "title": p.get("title", ""),
            "body": p.get("body", ""),
            "label": None,
        })
    out["years"][year] = items

json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)
total = sum(len(v) for v in out["years"].values())
print(f"Wrote {OUT}: {total} posts to label")
