#!/usr/bin/env python3
"""Inference API for the WSB post classifier (fine-tuned FinBERT).

Library use:
    from predictor import WSBClassifier
    clf = WSBClassifier()                      # loads ./finbert
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

Backends
    directory (default ./finbert)  -> fine-tuned ProsusAI/finbert (torch)
    *.joblib                       -> legacy TF-IDF + logistic regression

The FinBERT backend needs torch + transformers, so the server must be started
with a python that has them (e.g. the wsb_classification/.venv). The legacy
joblib backend needs scikit-learn + joblib.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "finbert")

Post = Union[str, Sequence[str], Dict[str, Any]]


def _join_text(title: str, body: str) -> str:
    parts = [str(title).strip(), str(body).strip()]
    return " ".join(p for p in parts if p).strip()


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
    return _join_text(title, body)


class WSBClassifier:
    """Classifies WSB posts. Dispatches to the FinBERT or legacy sklearn backend."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        if model_path.endswith(".joblib") or os.path.isfile(model_path):
            self._backend = _SklearnBackend(model_path)
        else:
            self._backend = _FinBertBackend(model_path)
        self.model_path = model_path

    @property
    def classes(self) -> List[str]:
        return self._backend.classes

    @property
    def config(self) -> Dict[str, Any]:
        return self._backend.config

    @property
    def backend(self) -> str:
        return self._backend.name

    def predict(self, posts: Iterable[Post], include_proba: bool = True
                ) -> List[Dict[str, Any]]:
        posts = list(posts)
        if not posts:
            return []
        texts = [_to_text(p) for p in posts]
        return self._backend.predict(texts, include_proba)

    def predict_one(self, title: str = "", body: str = "",
                    include_proba: bool = True) -> Dict[str, Any]:
        return self.predict([{"title": title, "body": body}],
                            include_proba=include_proba)[0]

    def predict_text(self, text: str, include_proba: bool = True) -> Dict[str, Any]:
        return self.predict([text], include_proba=include_proba)[0]

    def predict_labels(self, posts: Iterable[Post]) -> List[str]:
        return [r["label"] for r in self.predict(posts, include_proba=False)]

    def __repr__(self) -> str:
        return (f"<WSBClassifier {self.backend} "
                f"{os.path.basename(self.model_path)} classes={self.classes}>")


# --------------------------------------------------------------------- backends
class _FinBertBackend:
    name = "finbert"

    def __init__(self, model_dir: str):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"model directory not found: {model_dir}. Train it with "
                f"wsb_classification/train_finbert.py and copy the output here."
            )
        try:
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
        except ImportError as e:                               # pragma: no cover
            raise SystemExit(
                f"the FinBERT backend needs torch and transformers ({e}). "
                f"Start the server with a python that has them, e.g. "
                f"wsb_classification/.venv/bin/python."
            )

        meta_path = os.path.join(model_dir, "results.json")
        self.config: Dict[str, Any] = {}
        if os.path.exists(meta_path):
            self.config = json.load(open(meta_path, encoding="utf-8"))

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        cfg = self.model.config
        id2label = {int(k): v for k, v in cfg.id2label.items()}
        self.classes: List[str] = [str(id2label[i]) for i in sorted(id2label)]
        self.config.setdefault("classes", self.classes)
        self.config.setdefault("max_len", 256)
        self.config["device"] = self.device
        self.max_len = int(self.config.get("max_len", 256))
        # a single /predict request can carry a large list; encode it in slices so
        # a big batch cannot exhaust GPU memory
        self.batch_size = int(os.environ.get("WSB_BATCH_SIZE", "32"))
        self._index = {c: i for i, c in enumerate(self.classes)}

    def predict(self, texts: List[str], include_proba: bool = True
                ) -> List[Dict[str, Any]]:
        torch = self.torch
        out: List[Dict[str, Any]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            with torch.no_grad():
                enc = self.tokenizer(chunk, truncation=True, max_length=self.max_len,
                                     padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for j, (text, row) in enumerate(zip(chunk, probs)):
                rec: Dict[str, Any] = {
                    "index": start + j,
                    "label": self.classes[int(row.argmax())],
                    "empty_input": len(text) == 0,
                }
                if include_proba:
                    p = {c: round(float(row[self._index[c]]), 6) for c in self.classes}
                    rec["confidence"] = round(max(p.values()), 6)
                    rec["probabilities"] = p
                out.append(rec)
        return out


class _SklearnBackend:
    name = "sklearn"

    def __init__(self, model_path: str):
        import joblib
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        self.model_path = model_path
        bundle = joblib.load(model_path)
        self.features = bundle["features"]
        self.clf = bundle["clf"]
        self.classes: List[str] = [str(c) for c in bundle["classes"]]
        self.config: Dict[str, Any] = bundle.get("config", {})
        self._model_classes = [str(c) for c in self.clf.classes_]
        self._index = {c: i for i, c in enumerate(self._model_classes)}
        if set(self._model_classes) != set(self.classes):
            raise ValueError(
                f"class mismatch: bundle has {sorted(self.classes)} but fitted "
                f"model has {sorted(self._model_classes)}"
            )

    def predict(self, texts: List[str], include_proba: bool = True
                ) -> List[Dict[str, Any]]:
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
                probs = {c: round(float(row[self._index[c]]), 6) for c in self.classes}
                rec["confidence"] = round(max(probs.values()), 6)
                rec["probabilities"] = probs
            out.append(rec)
        return out


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
