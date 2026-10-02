#!/usr/bin/env python3
"""Render each sample chunk as a compact numbered text file for manual labelling."""
import json
import os

CHUNK_DIR = "sample_chunks"
TXT_DIR = "sample_text"
BODY_CHARS = 900


def render(p, i):
    head = f"[{i}] ({p.get('flair') or 'noflair'} | score {p.get('score')} | id {p['id']})"
    lines = [head, f"TITLE: {p.get('title') or ''}"]
    body = (p.get("body") or "").strip()
    if body:
        if len(body) > BODY_CHARS:
            body = body[:BODY_CHARS] + " [...truncated]"
        lines.append(f"BODY: {body}")
    else:
        lines.append("BODY: <empty>")
    return "\n".join(lines)


def main():
    os.makedirs(TXT_DIR, exist_ok=True)
    names = sorted(n for n in os.listdir(CHUNK_DIR)
                   if n.startswith("chunk_") and n.endswith(".json"))
    for name in names:
        rows = json.load(open(os.path.join(CHUNK_DIR, name), encoding="utf-8"))
        text = "\n\n".join(render(p, i) for i, p in enumerate(rows))
        with open(os.path.join(TXT_DIR, name.replace(".json", ".txt")), "w",
                  encoding="utf-8") as f:
            f.write(text + "\n")
    print(f"rendered {len(names)} chunks to {TXT_DIR}/")


if __name__ == "__main__":
    main()
