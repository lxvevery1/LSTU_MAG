"""ЛР1, вариант 16 — Классификация текста: от классического ML до дообучения трансформера.

Датасет: financial_phrasebank, разметка sentences_allagree (все аннотаторы согласны).
Задача: тональность финансовых новостей, 3 класса.
K = 3, MAX_LEN = 64, SEED = 1016, EPOCHS = 5, LR = 2e-5.

Оригинал financial_phrasebank грузится только скриптом загрузки, который
современные версии datasets не поддерживают, поэтому используется parquet-зеркало.

Скрипт покрывает Задание 1 (пункты 1 и 2).
"""

import collections
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoTokenizer

# --- вариант, зерно и импорты ---
VARIANT = 16        # номер своего варианта, 1..30
SEED = 1000 + VARIANT
EPOCHS, LR = 5, 2e-5   # варианты 13 и 14 переопределяют в своём коде

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- функция приведения данных к виду text / label ---
def to_pool(parts, text_fn, label_fn):
    """Склеивает части датасета и приводит их к двум колонкам: text, label."""
    d = concatenate_datasets(parts)
    d = d.map(lambda ex: {"text": text_fn(ex), "label": label_fn(ex)},
              remove_columns=d.column_names)
    return d.filter(lambda ex: bool(ex["text"]) and ex["label"] != -1)


# --- код варианта 16 ---
# зеркало financial_phrasebank (sentences_allagree): оригинал загружается только скриптом
ds = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
pool = to_pool(list(ds.values()), lambda ex: ex["sentence"], lambda ex: ex["label"])
LABELS = ["negative", "neutral", "positive"]
MAX_LEN = 64

# шаг 3.4: разметка с согласием >= 50% аннотаторов (те же метки 0/1/2)
ds50 = load_dataset("warwickai/financial_phrasebank_mirror")

# --- разбиение 70 / 15 / 15 ---
# pool, LABELS, MAX_LEN заданы кодом варианта (раздел 2)
N_MAX = 2000 if VARIANT == 28 else 5000
pool = pool.shuffle(seed=SEED).select(range(min(N_MAX, len(pool))))

a = pool.train_test_split(test_size=0.30, seed=SEED)
b = a["test"].train_test_split(test_size=0.50, seed=SEED)
train_ds, val_ds, test_ds = a["train"], b["train"], b["test"]
print(len(train_ds), len(val_ds), len(test_ds))

# --- Задание 1, пункт 1 — разведка данных ---
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# распределение классов в обучающей части
train_labels = np.array(train_ds["label"])
cnt = collections.Counter(train_labels.tolist())
counts = [cnt.get(i, 0) for i in range(len(LABELS))]
fracs = [c / len(train_ds) for c in counts]

print(f"обучающая часть: {len(train_ds)} примеров")
for name, c, f in zip(LABELS, counts, fracs):
    print(f"  {name:<9} {c:>5}  ({f:.2%})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].bar(LABELS, counts, color=["#c1666b", "#8a8f98", "#4f7cac"])
for i, (c, f) in enumerate(zip(counts, fracs)):
    ax[0].text(i, c + 12, f"{c}\n{f:.1%}", ha="center", va="bottom", fontsize=10)
ax[0].set_title("Распределение классов, обучающая часть")
ax[0].set_xlabel("класс")
ax[0].set_ylabel("число примеров")
ax[0].set_ylim(0, max(counts) * 1.18)

# гистограмма длин текстов в токенах DistilBERT
lengths = np.array([len(tokenizer(t, truncation=False)["input_ids"])
                    for t in train_ds["text"]])

ax[1].hist(lengths, bins=40, color="#4f7cac", edgecolor="white")
ax[1].axvline(MAX_LEN, color="#c1666b", linestyle="--", linewidth=2,
              label=f"MAX_LEN = {MAX_LEN}")
ax[1].set_title("Длины текстов в токенах DistilBERT, обучающая часть")
ax[1].set_xlabel("число токенов")
ax[1].set_ylabel("число текстов")
ax[1].legend()

plt.tight_layout()
plt.savefig("eda_plot.png", dpi=120)
plt.show()

# доля текстов длиннее MAX_LEN
print(f"длины: min={lengths.min()}, mean={lengths.mean():.1f}, "
      f"median={np.median(lengths):.0f}, p90={np.percentile(lengths, 90):.0f}, "
      f"max={lengths.max()}")

for name, d in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
    L = np.array([len(tokenizer(t, truncation=False)["input_ids"])
                  for t in d["text"]])
    over = L > MAX_LEN
    print(f"  {name}: длиннее {MAX_LEN} — {int(over.sum())} из {len(L)} "
          f"({over.mean():.2%}), средняя длина {L.mean():.1f}")

# --- Задание 1, пункт 2 — классический baseline: TF-IDF + логистическая регрессия ---
t0 = time.time()
vec = TfidfVectorizer(max_features=10000)
X_train = vec.fit_transform(train_ds["text"])
X_test = vec.transform(test_ds["text"])
clf = LogisticRegression(max_iter=1000).fit(X_train, train_ds["label"])
pred = clf.predict(X_test)
print(f"время обучения: {time.time() - t0:.1f} с")
print("Accuracy:", accuracy_score(test_ds["label"], pred))
print("Macro-F1:", f1_score(test_ds["label"], pred, average="macro"))

# сохраняем для сводной таблицы задания 2
BASELINE = {"model": "TF-IDF + LogReg",
            "accuracy": accuracy_score(test_ds["label"], pred),
            "f1_macro": f1_score(test_ds["label"], pred, average="macro"),
            "time_s": time.time() - t0}
print("размерность матрицы:", X_train.shape, "| признаков в словаре:", X_train.shape[1])

# метрики по классам — по ним видно, где baseline теряет качество
print(classification_report(test_ds["label"], pred, target_names=LABELS,
                            digits=4, zero_division=0))
print("ConfusionMatrix (строки — истинные классы, столбцы — предсказанные):")
print(confusion_matrix(test_ds["label"], pred))
