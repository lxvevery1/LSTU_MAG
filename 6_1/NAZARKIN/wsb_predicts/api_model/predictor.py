#!/usr/bin/env python3
"""Inference API for the WSB post classifier.

Library use:
    from predictor import WSBClassifier
    clf = WSBClassifier()                      # loads ./model.joblib
    clf.predict_one("GME earnings tomorrow", "GME calls 12/31 $80")
    # {'label': 'GENUINE_OPINION', 'confidence': 0.61, ...}

    clf.predict([
        {"title": "SPY all time highs", "body": ""},
        {"title": "Why do sector rotations lag?", "body": "..."},
    ])

Accepted input shapes (auto-detected):
    str                      -> used as title+body
    (title, body) tuple      -> 2-element sequences
    dict                     -> any of: title/body, title/selftext,
                                 title/text, or a bare "text" key

Every returned record always has "label". "probabilities" and "confidence" are
included unless include_proba=False.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import joblib

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

Post = Union[str, Sequence[str], Dict[str, Any]]


class WSBClassifier:
    """Thin, dependency-light wrapper around the pickled sklearn pipeline."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"model not found: {model_path}. Run fit_api_model.py to build it."
            )
        self.model_path = model_path
        bundle = joblib.load(model_path)
        self.features = bundle["features"]      # fitted FeatureUnion (word + char tfidf)
        self.clf = bundle["clf"]                # fitted LogisticRegression
        self.classes: List[str] = [str(c) for c in bundle["classes"]]
        self.config: Dict[str, Any] = bundle.get("config", {})
        # sklearn sorts classes_ alphabetically, which is NOT the order stored in
        # bundle["classes"] - probabilities must be indexed by the fitted order
        self._model_classes: List[str] = [str(c) for c in self.clf.classes_]
        self._index = {c: i for i, c in enumerate(self._model_classes)}
        if set(self._model_classes) != set(self.classes):
            raise ValueError(
                f"class mismatch: bundle has {sorted(self.classes)} but fitted "
                f"model has {sorted(self._model_classes)}"
            )

    # ------------------------------------------------------------------ input
    @staticmethod
    def _to_text(post: Post) -> str:
        if isinstance(post, str):
            title, body = post, ""
        elif isinstance(post, dict):
            title = post.get("title") or ""
            body = (post.get("body") or post.get("selftext")
                    or post.get("text") or "")
        elif isinstance(post, (list, tuple)):
            title = post[0] if len(post) > 0 else ""
            body = post[1] if len(post) > 1 else ""
        else:
            raise TypeError(
                f"unsupported post type {type(post).__name__}; expected str, "
                f"(title, body) sequence or dict"
            )
        parts = [str(title).strip(), str(body).strip()]
        return " ".join(p for p in parts if p).strip()

    # ----------------------------------------------------------------- predict
    def predict(self, posts: Iterable[Post], include_proba: bool = True
                ) -> List[Dict[str, Any]]:
        """Classify a batch of posts. Returns one record per input post, in order."""
        posts = list(posts)
        if not posts:
            return []
        texts = [self._to_text(p) for p in posts]
        X = self.features.transform(texts)
        P = self.clf.predict_proba(X)
        pred = self.clf.predict(X)

        out = []
        for i, (text, p_idx, row) in enumerate(zip(texts, pred, P)):
            rec: Dict[str, Any] = {
                "index": i,
                "label": str(p_idx),
                "empty_input": len(text) == 0,
            }
            if include_proba:
                # emit keys in the stable self.classes order, values from the
                # fitted (alphabetical) column order
                probs = {c: round(float(row[self._index[c]]), 6) for c in self.classes}
                rec["confidence"] = round(max(probs.values()), 6)
                rec["probabilities"] = probs
            out.append(rec)
        return out

    def predict_one(self, title: str = "", body: str = "",
                    include_proba: bool = True) -> Dict[str, Any]:
        """Classify a single post."""
        return self.predict([{"title": title, "body": body}],
                            include_proba=include_proba)[0]

    def predict_text(self, text: str, include_proba: bool = True) -> Dict[str, Any]:
        """Classify a single already-joined string."""
        return self.predict([text], include_proba=include_proba)[0]

    def predict_labels(self, posts: Iterable[Post]) -> List[str]:
        """Just the labels, no probabilities."""
        return [r["label"] for r in self.predict(posts, include_proba=False)]

    def __repr__(self) -> str:
        return (f"<WSBClassifier {os.path.basename(self.model_path)} "
                f"classes={self.classes}>")


# --------------------------------------------------------------------- CLI
def _main(argv: List[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Classify WSB posts from a JSON file or stdin.")
    ap.add_argument("input", nargs="?", default="-",
                    help="JSON file with a list of posts, or '-' for stdin")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("--labels-only", action="store_true")
    args = ap.parse_args(argv)

    raw = sys.stdin.read() if args.input == "-" else open(args.input, encoding="utf-8").read()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("posts", [data])

    clf = WSBClassifier(args.model)
    res = clf.predict(data, include_proba=not args.labels_only)
    if args.labels_only:
        res = [r["label"] for r in res]
    json.dump(res, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))