import json, math, random, re, time
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

VARIANT = 16
SEED = 1000 + VARIANT
N_TRAIN = 1000
N_EVAL = 50

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_base(four_bit=False):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16) if four_bit else None
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map={"": 0}, quantization_config=q)

def chat(instruction, response=None):
    msgs = [{"role": "user", "content": instruction}]
    if response is None:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    msgs.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(msgs, tokenize=False)

def cut(text, n=512):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= n else tokenizer.decode(ids[:n])

@torch.no_grad()
def generate(model, instructions, max_new_tokens=100, **gen):
    model.eval()
    outs = []
    for s in instructions:
        enc = tokenizer(chat(s), return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=1536).to(model.device)
        o = model.generate(**enc, max_new_tokens=max_new_tokens,
                           pad_token_id=tokenizer.eos_token_id, **gen)
        outs.append(tokenizer.decode(o[0, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip())
    return outs

import string
from collections import Counter
from rouge_score import rouge_scorer
import sacrebleu

def rouge(kind):
    sc = rouge_scorer.RougeScorer([kind], use_stemmer=True)
    return lambda P, R: float(np.mean([sc.score(r, p)[kind].fmeasure for p, r in zip(P, R)]))

def bleu(P, R):
    return sacrebleu.corpus_bleu(P, [R]).score

def _norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())

def _f1(p, r):
    p, r = _norm(p).split(), _norm(r).split()
    c = sum((Counter(p) & Counter(r)).values())
    if not p or not r:
        return float(p == r)
    return 0.0 if c == 0 else 2 * c / (len(p) + len(r))

def squad_f1(P, R):
    return float(np.mean([max(_f1(p, x) for x in r.split(" ||| ")) for p, r in zip(P, R)]))

def exact_match(P, R):
    return float(np.mean([any(_norm(p) == _norm(x) for x in r.split(" ||| ")) for p, r in zip(P, R)]))

def label_accuracy(labels):
    def f(P, R):
        def parse(p):
            hits = [l for l in labels if re.search(r"\b" + re.escape(l.lower()) + r"\b", p.lower())]
            return hits[0].lower() if len(hits) == 1 else None
        return float(np.mean([parse(p) == r.lower() for p, r in zip(P, R)]))
    return f

def _entities(s):
    out = set()
    for part in s.split(";"):
        if ":" in part:
            t, xs = part.split(":", 1)
            out |= {(t.strip().upper(), x.strip().lower()) for x in xs.split(",") if x.strip()}
    return out

def entity_prf(P, R):
    tp = fp = fn = 0
    for p, r in zip(P, R):
        a, b = _entities(p), _entities(r)
        tp += len(a & b); fp += len(a - b); fn += len(b - a)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"precision": prec, "recall": rec, "f1": 2 * tp / max(1, 2 * tp + fp + fn)}

def json_fields(P, R):
    def parse(s):
        try:
            d = json.loads(s.strip().split("\n")[0])
            return d if isinstance(d, dict) else {}
        except ValueError:
            return {}
    return float(np.mean([np.mean([parse(p).get(k) == v for k, v in parse(r).items()])
                          for p, r in zip(P, R)]))

def length_ratio(P, R):
    return float(np.mean([len(p.split()) / max(1, len(r.split())) for p, r in zip(P, R)]))

def style_ok(P, R):
    return float(np.mean([(len(p.split()) <= 40) == (len(r.split()) <= 40) for p, r in zip(P, R)]))

def _last_number(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$", ""))
    return m[-1].replace(",", "").rstrip(".") if m else None

def gsm_format(P, R):
    return float(np.mean([bool(re.search(r"####\s*-?[\d,.]+", p)) for p in P]))

def gsm_accuracy(P, R):
    return float(np.mean([_last_number(p) == _last_number(r) for p, r in zip(P, R)]))

def semantic_similarity(P, R):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    a = st.encode(P, normalize_embeddings=True); b = st.encode(R, normalize_embeddings=True)
    return float(np.mean((a * b).sum(-1)))

def distinct2(texts):
    bg = [tuple(t.split()[i:i + 2]) for t in texts for i in range(len(t.split()) - 1)]
    return len(set(bg)) / max(1, len(bg))

def distinct2_metric(P, R):
    return distinct2(P)

# Вариант 16: перевод EN->DE
ds = load_dataset("Helsinki-NLP/opus-100", "de-en")
train_raw, test_raw = ds["train"], ds["test"]
to_pair = lambda ex: ("Translate from English to German.\n\n" + ex["translation"]["en"],
                      ex["translation"]["de"])
metric = bleu
MAX_NEW_TOKENS = 96
RANKS = [4, 8, 16]

tr = train_raw.shuffle(seed=SEED).select(range(min(N_TRAIN + 100, len(train_raw))))
pairs = [to_pair(ex) for ex in tr]
train_pairs, val_pairs = pairs[:-100], pairs[-100:]
test_raw_s = test_raw.shuffle(seed=SEED)
test_pairs = [to_pair(ex) for ex in test_raw_s.select(range(min(100, len(test_raw_s))))]

def to_records(pairs):
    rec = []
    for p, r in pairs:
        n_p = len(tokenizer(chat(p), add_special_tokens=False)["input_ids"])
        n_all = len(tokenizer(chat(p, r), add_special_tokens=False)["input_ids"])
        rec.append({"prompt": p, "response": r, "mask": [0] * n_p + [1] * (n_all - n_p)})
    return rec

records = to_records(train_pairs)

def to_sft(pairs):
    return Dataset.from_list([{"prompt": [{"role": "user", "content": p}],
                               "completion": [{"role": "assistant", "content": r}]}
                              for p, r in pairs])

train_sft, val_sft = to_sft(train_pairs), to_sft(val_pairs)
print(f"Данные: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

base = load_base()

# ===== Задание 1 =====
print("\n===== Шаг 1.1. Стратегии декодирования =====")
MODES = {
    "Greedy":      dict(do_sample=False),
    "Temperature": dict(do_sample=True, temperature=0.7),
    "Top-k":       dict(do_sample=True, top_k=40, temperature=0.7),
    "Top-p":       dict(do_sample=True, top_p=0.9, temperature=0.7),
    "Beam":        dict(do_sample=False, num_beams=4),
}

def repetition_rate(texts):
    r = []
    for t in texts:
        w = t.split(); tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
        r.append(1 - len(set(tri)) / len(tri) if tri else 0.0)
    return float(np.mean(r))

prompts10 = [x for x, _ in test_pairs[:10]]
outs_by_mode = {}
for name, g in MODES.items():
    outs = generate(base, prompts10, max_new_tokens=100, **g)
    outs_by_mode[name] = outs
    print(f"{name:12} Distinct-2={distinct2(outs):.3f} Rep={repetition_rate(outs):.3f} "
          f"длина={np.mean([len(o.split()) for o in outs]):.1f}")

best = ("", "", -1.0); best_i = -1
for name, outs in outs_by_mode.items():
    for i, o in enumerate(outs):
        w = o.split(); tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
        rep = 1 - len(set(tri)) / len(tri) if tri else 0.0
        if rep > best[2]:
            best = (name, o, rep); best_i = i
print("\n--- Пример зацикливания (макс. доля повторных триграмм) ---")
print(f"режим: {best[0]}, Rep={best[2]:.3f}")
src = prompts10[best_i].split("\n\n", 1)[1]
print(f"EN: {src[:300]}")
print(f"DE: {best[1][:600]}")

print("\n--- Переводы Greedy (поиск галлюцинации) ---")
for i, (p, o) in enumerate(zip(prompts10, outs_by_mode["Greedy"])):
    s = p.split("\n\n", 1)[1]
    print(f"[{i}] EN: {s[:220]}")
    print(f"    DE: {o[:220]}")

print("\n===== Шаг 1.2. Маска потерь =====")
print("Ручной пример: промпт 6 токенов (mt=0, сумма потерь 3.6), ответ 4 токена (mt=1, сумма 4.0)")
print("без маски: (3.6+4.0)/10 = 0.76; с маской: 4.0/4 = 1.00")
print("Напрямую не сравнивать: разные нормировки (все токены ленты vs только токены ответа).")

def build_labels(input_ids, prompt_len):
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return labels

instr, resp = train_pairs[0]
full_ids = tokenizer(chat(instr, resp), add_special_tokens=False,
                     return_tensors="pt")["input_ids"][0]
prompt_len = len(tokenizer(chat(instr), add_special_tokens=False)["input_ids"])
labels = build_labels(full_ids, prompt_len)
print("\nm_t = 1 на токенах:", repr(tokenizer.decode(full_ids[labels != -100])))
print("ожидаемый ответ   :", repr(resp))

with torch.no_grad():
    logits = base(full_ids[None].to(base.device)).logits[0, :-1].float()
    tok_loss = torch.nn.functional.cross_entropy(logits, full_ids[1:].to(base.device),
                                                 reduction="none")
    m = (labels[1:] != -100).to(base.device)

print("без маски:", tok_loss.mean().item())
print("с маской :", tok_loss[m].mean().item())

n_prompt = prompt_len
print(f"\nГраница в labels: метки != -100 с позиции {n_prompt} "
      f"(токен {tokenizer.decode([full_ids[n_prompt].item()])!r} — первый токен ответа)")
print(f"Граница в векторе потерь (labels[1:]): маска включается с позиции {n_prompt - 1} "
      f"в ленте logits")
print(f"Позиции logits, участвующие в потере: {n_prompt - 1}..{len(full_ids) - 2} "
      f"(всего {m.sum().item()})")
assert m.sum().item() == len(full_ids) - prompt_len
assert bool(m[n_prompt - 1]) and not bool(m[:n_prompt - 1].any())
print("OK: в потере ровно токены ответа (+ EOS); первый токен ответа учится по последнему токену промпта.")
