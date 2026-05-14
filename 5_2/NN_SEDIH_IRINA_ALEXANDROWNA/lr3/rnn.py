"""
ЛР3: character-level LSTM и GRU на NumPy (forward + BPTT), Adam, автодополнение.

Зависимости (ставить только из каталога проекта, без глобального pip):
  ./venv/bin/python -m pip install -r requirements.txt

Если в venv нет pip (Arch):
  ./venv/bin/python -m ensurepip --upgrade
  ./venv/bin/python -m pip install -r requirements.txt
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from sklearn.metrics import top_k_accuracy_score
except ImportError:  # pragma: no cover
    top_k_accuracy_score = None


# --- гиперпараметры из задания (раздел 3.2) ---
T = 128
B = 128
LR = 2e-3
BETA1 = 0.9
BETA2 = 0.999
EPS_ADAM = 1e-8
CLIP = 5.0
D_EMB = 24
H_HIDDEN = 96
MAX_EPOCHS = 20
PATIENCE = 6
STEPS_PER_EPOCH = 120
EVAL_BATCHES = 20
SEED = 42

RNG = np.random.default_rng(SEED)
random.seed(SEED)


def k_init(h: int) -> float:
    return 1.0 / math.sqrt(h)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid_from_output(s: np.ndarray) -> np.ndarray:
    return s * (1.0 - s)


def init_matrix(rows: int, cols: int, k: float) -> np.ndarray:
    return RNG.uniform(-k, k, size=(rows, cols)).astype(np.float64)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-15)


def cross_entropy_grad(logits: np.ndarray, targets: np.ndarray, scale: float) -> tuple[np.ndarray, float]:
    """Средний CE по (B*T); scale = 1/(B*T). logits (B,T,V), targets (B,T)."""
    b, t, v = logits.shape
    flat = logits.reshape(-1, v)
    p = softmax(flat)
    y = targets.reshape(-1)
    chosen = p[np.arange(y.shape[0], dtype=np.int64), y]
    loss = -float(np.mean(np.log(np.clip(chosen, 1e-15, None))))
    g = p.copy()
    g[np.arange(y.shape[0], dtype=np.int64), y] -= 1.0
    g *= scale
    return g.reshape(b, t, v), loss


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Доля попаданий истинного класса в top-k по последней размерности."""
    if top_k_accuracy_score is not None:
        y = targets.reshape(-1)
        s = logits.reshape(-1, logits.shape[-1])
        return float(top_k_accuracy_score(y, s, k=k, labels=np.arange(s.shape[1])))
    correct = 0
    total = targets.size
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_t = targets.reshape(-1)
    for i in range(total):
        idx = np.argpartition(-flat_logits[i], kth=min(k, flat_logits.shape[1] - 1))[:k]
        if flat_t[i] in idx:
            correct += 1
    return correct / max(total, 1)


def clip_norm(grads: dict[str, np.ndarray], max_norm: float) -> None:
    total = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
    if total > max_norm and total > 0:
        coef = max_norm / total
        for k in grads:
            grads[k] *= coef


@dataclass
class AdamState:
    m: dict[str, np.ndarray]
    v: dict[str, np.ndarray]
    t: int


def adam_update(params: dict[str, np.ndarray], grads: dict[str, np.ndarray], state: AdamState) -> None:
    state.t += 1
    t = state.t
    lr_t = LR * math.sqrt(1.0 - BETA2**t) / (1.0 - BETA1**t)
    for name, p in params.items():
        g = grads[name]
        if name not in state.m:
            state.m[name] = np.zeros_like(p)
            state.v[name] = np.zeros_like(p)
        state.m[name] = BETA1 * state.m[name] + (1.0 - BETA1) * g
        state.v[name] = BETA2 * state.v[name] + (1.0 - BETA2) * (g * g)
        p -= lr_t * state.m[name] / (np.sqrt(state.v[name]) + EPS_ADAM)


def sample_windows(stream: np.ndarray, batch: int, seq_len: int, n_batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Случайные окна: x (B,T), y (B,T) — следующий символ."""
    max_start = len(stream) - seq_len - 1
    starts = RNG.integers(0, max_start + 1, size=(n_batch * batch,))
    x = np.empty((n_batch * batch, seq_len), dtype=np.int64)
    y = np.empty((n_batch * batch, seq_len), dtype=np.int64)
    for i, s in enumerate(starts):
        sl = stream[s : s + seq_len + 1]
        x[i] = sl[:-1]
        y[i] = sl[1:]
    return x, y


class CharLM_LSTM:
    def __init__(self, vocab: int, d: int, h: int):
        self.V = vocab
        self.d = d
        self.H = h
        k = k_init(h)
        self.E = init_matrix(vocab, d, k)
        self.Wf, self.Uf = init_matrix(h, d, k), init_matrix(h, h, k)
        self.Wi, self.Ui = init_matrix(h, d, k), init_matrix(h, h, k)
        self.Wc, self.Uc = init_matrix(h, d, k), init_matrix(h, h, k)
        self.Wo, self.Uo = init_matrix(h, d, k), init_matrix(h, h, k)
        self.bf = np.zeros(h, dtype=np.float64)
        self.bi = np.zeros(h, dtype=np.float64)
        self.bc = np.zeros(h, dtype=np.float64)
        self.bo = np.zeros(h, dtype=np.float64)
        self.Wy = init_matrix(h, vocab, k)
        self.by = np.zeros(vocab, dtype=np.float64)

    def params(self) -> dict[str, np.ndarray]:
        return {
            "E": self.E,
            "Wf": self.Wf,
            "Uf": self.Uf,
            "Wi": self.Wi,
            "Ui": self.Ui,
            "Wc": self.Wc,
            "Uc": self.Uc,
            "Wo": self.Wo,
            "Uo": self.Uo,
            "bf": self.bf,
            "bi": self.bi,
            "bc": self.bc,
            "bo": self.bo,
            "Wy": self.Wy,
            "by": self.by,
        }

    def zero_grads(self) -> dict[str, np.ndarray]:
        return {k: np.zeros_like(v) for k, v in self.params().items()}

    def forward_backward(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        """x, y: (B, T)."""
        Bb, Tt = x.shape
        scale = 1.0 / (Bb * Tt)
        h = self.H

        emb = self.E[x]
        caches: list[dict] = []
        hp = np.zeros((Bb, h), dtype=np.float64)
        cp = np.zeros((Bb, h), dtype=np.float64)
        h_out = np.zeros((Bb, Tt, h), dtype=np.float64)

        for t in range(Tt):
            xt = emb[:, t, :]
            pre_f = xt @ self.Wf.T + hp @ self.Uf.T + self.bf
            pre_i = xt @ self.Wi.T + hp @ self.Ui.T + self.bi
            pre_c = xt @ self.Wc.T + hp @ self.Uc.T + self.bc
            pre_o = xt @ self.Wo.T + hp @ self.Uo.T + self.bo
            f = sigmoid(pre_f)
            ii = sigmoid(pre_i)
            c_tilde = np.tanh(pre_c)
            o = sigmoid(pre_o)
            c = f * cp + ii * c_tilde
            tanh_c = np.tanh(c)
            hn = o * tanh_c
            caches.append(
                {
                    "x": xt,
                    "h_prev": hp.copy(),
                    "c_prev": cp.copy(),
                    "f": f,
                    "i": ii,
                    "o": o,
                    "c_tilde": c_tilde,
                    "c": c,
                    "tanh_c": tanh_c,
                    "pre_f": pre_f,
                    "pre_i": pre_i,
                    "pre_o": pre_o,
                    "pre_c": pre_c,
                }
            )
            hp, cp = hn, c
            h_out[:, t, :] = hn

        logits = np.einsum("bth,hv->btv", h_out, self.Wy) + self.by
        dlog, loss = cross_entropy_grad(logits, y, scale)

        g = self.zero_grads()
        dWy = np.einsum("bth,btv->hv", h_out, dlog)
        g["Wy"] += dWy
        g["by"] += np.sum(dlog, axis=(0, 1))
        dh_all = np.einsum("btv,hv->bth", dlog, self.Wy)

        dh_next = np.zeros((Bb, h), dtype=np.float64)
        dc_next = np.zeros((Bb, h), dtype=np.float64)

        for t in range(Tt - 1, -1, -1):
            cch = caches[t]
            xt = cch["x"]
            h_prev = cch["h_prev"]
            c_prev = cch["c_prev"]
            f, ii, o = cch["f"], cch["i"], cch["o"]
            c_tilde = cch["c_tilde"]
            c = cch["c"]
            tanh_c = cch["tanh_c"]

            dh = dh_all[:, t, :] + dh_next
            dC = dc_next + dh * o * (1.0 - tanh_c * tanh_c)

            df = dC * c_prev
            di = dC * c_tilde
            dc_tilde = dC * ii * (1.0 - c_tilde * c_tilde)
            dcp = dC * f
            do = dh * tanh_c

            dpre_o = do * dsigmoid_from_output(o)
            dpre_f = df * dsigmoid_from_output(f)
            dpre_i = di * dsigmoid_from_output(ii)
            dpre_c = dc_tilde * (1.0 - np.tanh(cch["pre_c"]) ** 2)

            dxt = (
                dpre_f @ self.Wf
                + dpre_i @ self.Wi
                + dpre_c @ self.Wc
                + dpre_o @ self.Wo
            )
            dh_prev = (
                dpre_f @ self.Uf
                + dpre_i @ self.Ui
                + dpre_c @ self.Uc
                + dpre_o @ self.Uo
            )

            g["Wf"] += dpre_f.T @ xt
            g["Wi"] += dpre_i.T @ xt
            g["Wc"] += dpre_c.T @ xt
            g["Wo"] += dpre_o.T @ xt
            g["Uf"] += dpre_f.T @ h_prev
            g["Ui"] += dpre_i.T @ h_prev
            g["Uc"] += dpre_c.T @ h_prev
            g["Uo"] += dpre_o.T @ h_prev
            g["bf"] += np.sum(dpre_f, axis=0)
            g["bi"] += np.sum(dpre_i, axis=0)
            g["bc"] += np.sum(dpre_c, axis=0)
            g["bo"] += np.sum(dpre_o, axis=0)

            np.add.at(g["E"], x[:, t], dxt)

            dh_next = dh_prev
            dc_next = dcp

        return loss, g


class CharLM_GRU:
    def __init__(self, vocab: int, d: int, h: int):
        self.V = vocab
        self.d = d
        self.H = h
        k = k_init(h)
        self.E = init_matrix(vocab, d, k)
        self.Wr, self.Ur = init_matrix(h, d, k), init_matrix(h, h, k)
        self.Wz, self.Uz = init_matrix(h, d, k), init_matrix(h, h, k)
        self.Wh, self.Uh = init_matrix(h, d, k), init_matrix(h, h, k)
        self.br = np.zeros(h, dtype=np.float64)
        self.bz = np.zeros(h, dtype=np.float64)
        self.bh = np.zeros(h, dtype=np.float64)
        self.Wy = init_matrix(h, vocab, k)
        self.by = np.zeros(vocab, dtype=np.float64)

    def params(self) -> dict[str, np.ndarray]:
        return {
            "E": self.E,
            "Wr": self.Wr,
            "Ur": self.Ur,
            "Wz": self.Wz,
            "Uz": self.Uz,
            "Wh": self.Wh,
            "Uh": self.Uh,
            "br": self.br,
            "bz": self.bz,
            "bh": self.bh,
            "Wy": self.Wy,
            "by": self.by,
        }

    def zero_grads(self) -> dict[str, np.ndarray]:
        return {k: np.zeros_like(v) for k, v in self.params().items()}

    def forward_backward(self, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        Bb, Tt = x.shape
        scale = 1.0 / (Bb * Tt)
        h = self.H
        emb = self.E[x]
        caches: list[dict] = []
        hp = np.zeros((Bb, h), dtype=np.float64)
        h_out = np.zeros((Bb, Tt, h), dtype=np.float64)

        for t in range(Tt):
            xt = emb[:, t, :]
            pre_r = xt @ self.Wr.T + hp @ self.Ur.T + self.br
            pre_z = xt @ self.Wz.T + hp @ self.Uz.T + self.bz
            r = sigmoid(pre_r)
            z = sigmoid(pre_z)
            q = r * hp
            pre_h = xt @ self.Wh.T + q @ self.Uh.T + self.bh
            h_tilde = np.tanh(pre_h)
            hn = (1.0 - z) * hp + z * h_tilde
            caches.append(
                {
                    "x": xt,
                    "h_prev": hp.copy(),
                    "r": r,
                    "z": z,
                    "q": q,
                    "pre_r": pre_r,
                    "pre_z": pre_z,
                    "pre_h": pre_h,
                    "h_tilde": h_tilde,
                }
            )
            hp = hn
            h_out[:, t, :] = hn

        logits = np.einsum("bth,hv->btv", h_out, self.Wy) + self.by
        dlog, loss = cross_entropy_grad(logits, y, scale)

        g = self.zero_grads()
        g["Wy"] += np.einsum("bth,btv->hv", h_out, dlog)
        g["by"] += np.sum(dlog, axis=(0, 1))
        dh_all = np.einsum("btv,hv->bth", dlog, self.Wy)

        dh_next = np.zeros((Bb, h), dtype=np.float64)
        for t in range(Tt - 1, -1, -1):
            cch = caches[t]
            xt = cch["x"]
            h_prev = cch["h_prev"]
            r, z = cch["r"], cch["z"]
            pre_r, pre_z, pre_h = cch["pre_r"], cch["pre_z"], cch["pre_h"]
            h_tilde = cch["h_tilde"]

            dh = dh_all[:, t, :] + dh_next
            dhtilde = dh * z
            dz = dh * (h_tilde - h_prev)
            dh_prev = dh * (1.0 - z)

            dpre_h = dhtilde * (1.0 - h_tilde * h_tilde)
            dq = dpre_h @ self.Uh
            dh_prev += dq * r
            dr = dq * h_prev
            dpre_r = dr * dsigmoid_from_output(r)
            dpre_z = dz * dsigmoid_from_output(z)

            dxt = dpre_r @ self.Wr + dpre_z @ self.Wz + dpre_h @ self.Wh
            dh_prev += dpre_r @ self.Ur + dpre_z @ self.Uz

            g["Wr"] += dpre_r.T @ xt
            g["Wz"] += dpre_z.T @ xt
            g["Wh"] += dpre_h.T @ xt
            g["Ur"] += dpre_r.T @ h_prev
            g["Uz"] += dpre_z.T @ h_prev
            g["Uh"] += dpre_h.T @ cch["q"]
            g["br"] += np.sum(dpre_r, axis=0)
            g["bz"] += np.sum(dpre_z, axis=0)
            g["bh"] += np.sum(dpre_h, axis=0)

            np.add.at(g["E"], x[:, t], dxt)

            dh_next = dh_prev

        return loss, g


@dataclass
class History:
    train_ppl: list[float]
    val_ppl: list[float]
    val_top1: list[float]
    val_top5: list[float]
    val_top10: list[float]
    best_epoch: int


def evaluate_model(
    model: CharLM_LSTM | CharLM_GRU,
    stream: np.ndarray,
    seq_len: int,
    batch: int,
    n_batches: int,
) -> tuple[float, float, float, float]:
    """Средний CE -> perplexity; top-1/5/10 accuracy."""
    xb, yb = sample_windows(stream, batch, seq_len, n_batches)
    losses = []
    logits_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for i in range(n_batches):
        sl = slice(i * batch, (i + 1) * batch)
        xx, yy = xb[sl], yb[sl]
        if isinstance(model, CharLM_LSTM):
            emb = model.E[xx]
            hp = np.zeros((batch, model.H))
            cp = np.zeros((batch, model.H))
            h_out = np.zeros((batch, seq_len, model.H))
            for t in range(seq_len):
                xt = emb[:, t, :]
                f = sigmoid(xt @ model.Wf.T + hp @ model.Uf.T + model.bf)
                ii = sigmoid(xt @ model.Wi.T + hp @ model.Ui.T + model.bi)
                c_tilde = np.tanh(xt @ model.Wc.T + hp @ model.Uc.T + model.bc)
                o = sigmoid(xt @ model.Wo.T + hp @ model.Uo.T + model.bo)
                cp = f * cp + ii * c_tilde
                hp = o * np.tanh(cp)
                h_out[:, t, :] = hp
        else:
            emb = model.E[xx]
            hp = np.zeros((batch, model.H))
            h_out = np.zeros((batch, seq_len, model.H))
            for t in range(seq_len):
                xt = emb[:, t, :]
                r = sigmoid(xt @ model.Wr.T + hp @ model.Ur.T + model.br)
                z = sigmoid(xt @ model.Wz.T + hp @ model.Uz.T + model.bz)
                q = r * hp
                h_tilde = np.tanh(xt @ model.Wh.T + q @ model.Uh.T + model.bh)
                hp = (1.0 - z) * hp + z * h_tilde
                h_out[:, t, :] = hp
        logits = np.einsum("bth,hv->btv", h_out, model.Wy) + model.by
        _, loss = cross_entropy_grad(logits, yy, 1.0 / (batch * seq_len))
        losses.append(loss)
        logits_chunks.append(logits.reshape(-1, model.V))
        y_chunks.append(yy.reshape(-1))
    mean_ce = float(np.mean(losses))
    ppl = math.exp(mean_ce)
    log_all = np.concatenate(logits_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    t1 = topk_accuracy(log_all, y_all, 1)
    t5 = topk_accuracy(log_all, y_all, 5)
    t10 = topk_accuracy(log_all, y_all, 10)
    return ppl, t1, t5, t10


def train_one(
    name: str,
    model: CharLM_LSTM | CharLM_GRU,
    train: np.ndarray,
    val: np.ndarray,
    out_dir: Path,
) -> History:
    state = AdamState(m={}, v={}, t=0)
    hist = History(
        train_ppl=[],
        val_ppl=[],
        val_top1=[],
        val_top5=[],
        val_top10=[],
        best_epoch=0,
    )
    best_val = float("inf")
    bad = 0
    params = model.params()

    for ep in range(1, MAX_EPOCHS + 1):
        ep_losses = []
        for _ in range(STEPS_PER_EPOCH):
            xb, yb = sample_windows(train, B, T, 1)
            loss, grads = model.forward_backward(xb, yb)
            clip_norm(grads, CLIP)
            adam_update(params, grads, state)
            ep_losses.append(loss)
        tr_ppl = math.exp(float(np.mean(ep_losses)))
        vp, v1, v5, v10 = evaluate_model(model, val, T, B, EVAL_BATCHES)
        hist.train_ppl.append(tr_ppl)
        hist.val_ppl.append(vp)
        hist.val_top1.append(v1)
        hist.val_top5.append(v5)
        hist.val_top10.append(v10)
        print(f"[{name}] epoch {ep}/{MAX_EPOCHS}  train_ppl={tr_ppl:.2f}  val_ppl={vp:.2f}  val_top1={v1:.4f}")
        if vp < best_val:
            best_val = vp
            hist.best_epoch = ep
            bad = 0
            np.savez_compressed(out_dir / f"best_{name.lower()}.npz", **{k: v.copy() for k, v in params.items()})
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[{name}] early stop at epoch {ep}, best val_ppl={best_val:.2f} @ epoch {hist.best_epoch}")
                break
    return hist


def plot_history(name: str, hist: History, out: Path) -> None:
    if plt is None:
        print("matplotlib не установлен — графики пропущены. Локально: ./venv/bin/python -m pip install matplotlib")
        return
    epochs = np.arange(1, len(hist.train_ppl) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, hist.train_ppl, label="train")
    axes[0].plot(epochs, hist.val_ppl, label="val")
    axes[0].set_title(f"{name} — perplexity")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, hist.val_top1, label="top-1")
    axes[1].plot(epochs, hist.val_top5, label="top-5")
    axes[1].plot(epochs, hist.val_top10, label="top-10")
    axes[1].set_title(f"{name} — val top-k accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def load_weights(model: CharLM_LSTM | CharLM_GRU, path: Path) -> None:
    z = np.load(path, allow_pickle=True)
    for k, v in model.params().items():
        v[:] = z[k]


def next_char_logits(
    model: CharLM_LSTM | CharLM_GRU,
    prefix_ids: np.ndarray,
) -> np.ndarray:
    """Одно распределение на следующий символ после prefix_ids (1D int)."""
    h_dim = model.H
    if isinstance(model, CharLM_LSTM):
        hp = np.zeros((1, h_dim))
        cp = np.zeros((1, h_dim))
        for tid in prefix_ids:
            xt = model.E[tid : tid + 1, :]
            f = sigmoid(xt @ model.Wf.T + hp @ model.Uf.T + model.bf)
            ii = sigmoid(xt @ model.Wi.T + hp @ model.Ui.T + model.bi)
            c_tilde = np.tanh(xt @ model.Wc.T + hp @ model.Uc.T + model.bc)
            o = sigmoid(xt @ model.Wo.T + hp @ model.Uo.T + model.bo)
            cp = f * cp + ii * c_tilde
            hp = o * np.tanh(cp)
        logits = (hp @ model.Wy + model.by).ravel()
    else:
        hp = np.zeros((1, h_dim))
        for tid in prefix_ids:
            xt = model.E[tid : tid + 1, :]
            r = sigmoid(xt @ model.Wr.T + hp @ model.Ur.T + model.br)
            z = sigmoid(xt @ model.Wz.T + hp @ model.Uz.T + model.bz)
            q = r * hp
            h_tilde = np.tanh(xt @ model.Wh.T + q @ model.Uh.T + model.bh)
            hp = (1.0 - z) * hp + z * h_tilde
        logits = (hp @ model.Wy + model.by).ravel()
    return logits


def id_to_char(vocab_cp: np.ndarray, i: int) -> str:
    if i == 0:
        return "?"
    return chr(int(vocab_cp[i]))


def main() -> None:
    root = Path(__file__).resolve().parent
    npz_path = root / "data_char" / "dataset_char.npz"
    meta_path = root / "data_char" / "meta.json"
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    npz = np.load(npz_path, allow_pickle=True)
    vocab_cp = npz["vocab"]
    train = npz["train"].astype(np.int64)
    val = npz["val"].astype(np.int64)
    test = npz["test"].astype(np.int64)
    V = int(len(vocab_cp))

    print("vocab_size:", V)
    print("train_tokens:", len(train))
    print("sample text:", "".join(id_to_char(vocab_cp, i) for i in train[:300]))
    if meta_path.is_file():
        print("meta:", json.loads(meta_path.read_text()))

    print("\nГиперпараметры: T=%d B=%d d=%d H=%d lr=%g Adam(β1=%g,β2=%g,ε=%g) clip=%g epochs≤%d patience=%d steps/epoch=%d eval_batches=%d\n"
          % (T, B, D_EMB, H_HIDDEN, LR, BETA1, BETA2, EPS_ADAM, CLIP, MAX_EPOCHS, PATIENCE, STEPS_PER_EPOCH, EVAL_BATCHES))

    lstm = CharLM_LSTM(V, D_EMB, H_HIDDEN)
    gru = CharLM_GRU(V, D_EMB, H_HIDDEN)

    hist_l = train_one("LSTM", lstm, train, val, out_dir)
    plot_history("LSTM", hist_l, out_dir / "plots_lstm.png")

    hist_g = train_one("GRU", gru, train, val, out_dir)
    plot_history("GRU", hist_g, out_dir / "plots_gru.png")

    # Финальные метрики по лучшим весам val
    load_weights(lstm, out_dir / "best_lstm.npz")
    load_weights(gru, out_dir / "best_gru.npz")

    l_val = evaluate_model(lstm, val, T, B, EVAL_BATCHES)
    l_test = evaluate_model(lstm, test, T, B, EVAL_BATCHES)
    g_val = evaluate_model(gru, val, T, B, EVAL_BATCHES)
    g_test = evaluate_model(gru, test, T, B, EVAL_BATCHES)

    print("\n=== Таблица (лучшая эпоха по val, веса перезагружены из checkpoint) ===")
    print(f"LSTM  best_epoch={hist_l.best_epoch}  val_ppl={l_val[0]:.2f} top1/5/10={l_val[1]:.4f}/{l_val[2]:.4f}/{l_val[3]:.4f}  test_ppl={l_test[0]:.2f} top1/5/10={l_test[1]:.4f}/{l_test[2]:.4f}/{l_test[3]:.4f}")
    print(f"GRU   best_epoch={hist_g.best_epoch}  val_ppl={g_val[0]:.2f} top1/5/10={g_val[1]:.4f}/{g_val[2]:.4f}/{g_val[3]:.4f}  test_ppl={g_test[0]:.2f} top1/5/10={g_test[1]:.4f}/{g_test[2]:.4f}/{g_test[3]:.4f}")

    # Автодополнение: ≥10 префиксов из val, top-10 следующих символов
    print("\n=== Автодополнение (LSTM): top-10 следующего символа ===")
    rng = np.random.default_rng(123)
    for k in range(12):
        s = int(rng.integers(0, len(val) - 200))
        pref_len = int(rng.integers(20, 80))
        ids = val[s : s + pref_len]
        pref = "".join(id_to_char(vocab_cp, int(i)) for i in ids)
        logits = next_char_logits(lstm, ids)
        p = softmax(logits.reshape(1, -1)).ravel()
        top = np.argsort(-p)[:10]
        print(f"\n--- prefix #{k+1} ({pref_len} chars) ---")
        print(repr(pref[-120:]))
        for rank, tid in enumerate(top, 1):
            ch = id_to_char(vocab_cp, int(tid))
            print(f"  {rank:2d}. {repr(ch):6s}  p={p[tid]:.4f}")


if __name__ == "__main__":
    main()
