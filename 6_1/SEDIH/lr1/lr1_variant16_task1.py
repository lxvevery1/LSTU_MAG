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

# --- Задание 2. Дообучение предобученного трансформера ---
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          EarlyStoppingCallback)
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(LABELS))


def encode(d):
    d = d.map(lambda b: tokenizer(b["text"], truncation=True, max_length=MAX_LEN),
              batched=True)
    return d.rename_column("label", "labels").remove_columns(["text"])


train_enc, val_enc, test_enc = encode(train_ds), encode(val_ds), encode(test_ds)


def compute_metrics(p):
    pr = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, pr),
            "f1_macro": f1_score(p.label_ids, pr, average="macro")}


def make_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir, num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        save_total_limit=1, seed=SEED, report_to="none",
        fp16=torch.cuda.is_available())


def make_trainer(output_dir, train_set):
    m = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS))
    return Trainer(
        model=m, args=make_args(output_dir),
        train_dataset=train_set, eval_dataset=val_enc,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])


trainer = make_trainer("./out", train_enc)

t0 = time.time()
trainer.train()
TRAIN_TIME = time.time() - t0
print(f"время обучения DistilBERT: {TRAIN_TIME:.1f} с")
test_metrics = trainer.evaluate(test_enc, metric_key_prefix="test")
print("test:", test_metrics)

# кривые обучения: train loss / val loss / val Macro-F1 по эпохам
hist = trainer.state.log_history
train_pts = [(h["epoch"], h["loss"]) for h in hist if "loss" in h]
val_loss_pts = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
val_f1_pts = [(h["epoch"], h["eval_f1_macro"]) for h in hist if "eval_f1_macro" in h]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot([e for e, _ in train_pts], [v for _, v in train_pts], "o-", label="train loss")
ax[0].plot([e for e, _ in val_loss_pts], [v for _, v in val_loss_pts], "s-", label="val loss")
ax[0].set_title("Кривые обучения: loss по эпохам")
ax[0].set_xlabel("эпоха")
ax[0].set_ylabel("loss")
ax[0].set_xticks(range(1, EPOCHS + 1))
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].plot([e for e, _ in val_f1_pts], [v for _, v in val_f1_pts], "o-", color="#4f7cac")
ax[1].set_title("Валидационный Macro-F1 по эпохам")
ax[1].set_xlabel("эпоха")
ax[1].set_ylabel("val Macro-F1")
ax[1].set_xticks(range(1, EPOCHS + 1))
ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=120)
plt.show()

# сводная таблица: baseline против DistilBERT
results = pd.DataFrame([
    {"постановка": "TF-IDF + LogReg (baseline)",
     "Accuracy": BASELINE["accuracy"],
     "Macro-F1": BASELINE["f1_macro"],
     "время, с": round(BASELINE["time_s"], 1)},
    {"постановка": "DistilBERT (дообучение)",
     "Accuracy": test_metrics["test_accuracy"],
     "Macro-F1": test_metrics["test_f1_macro"],
     "время, с": round(TRAIN_TIME, 1)},
])
print(results.to_string(index=False))

# --- Задание 3, шаг 3.1: динамика потерь по эпохам ---
last_train_ep, last_train_loss = train_pts[-1]
last_val_ep, last_val_loss = val_loss_pts[-1]
best_loss_ep, best_loss = min(val_loss_pts, key=lambda x: x[1])
best_f1_ep, best_f1 = max(val_f1_pts, key=lambda x: x[1])
epochs_run = int(max(e for e, _ in val_loss_pts))

print(f"эпох выполнено: {epochs_run} из {EPOCHS} "
      f"-> ранняя остановка: {'да' if epochs_run < EPOCHS else 'нет'}")
print(f"лучший чекпоинт: {trainer.state.best_model_checkpoint}")
print(f"минимум val loss:     {best_loss:.4f} на эпохе {int(best_loss_ep)}")
print(f"максимум val Macro-F1: {best_f1:.4f} на эпохе {int(best_f1_ep)}")
print(f"train loss на последней эпохе ({int(last_train_ep)}): {last_train_loss:.4f}")
print(f"val  loss на последней эпохе ({int(last_val_ep)}): {last_val_loss:.4f}")
print(f"разрыв val - train loss: {last_val_loss - last_train_loss:+.4f}")

# --- Задание 3, шаг 3.2: матрица ошибок на тесте ---
p_test = trainer.predict(test_enc)
pred_test = p_test.predictions.argmax(-1)
y_test = p_test.label_ids.astype(int)

print(classification_report(y_test, pred_test, target_names=LABELS,
                            digits=4, zero_division=0))

cm = confusion_matrix(y_test, pred_test)
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues",
                                                        values_format="d")
ax.set_title("Матрица ошибок DistilBERT, тест")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.show()

pairs = sorted(((LABELS[i], LABELS[j], int(cm[i, j]))
                for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j),
               key=lambda x: -x[2])
print("топ пар путаницы (истинный -> предсказанный):")
for a, b, c in pairs[:4]:
    print(f"  {a} -> {b}: {c}")

# --- Задание 3, шаг 3.3: разбор трёх ошибок ---
logits = p_test.predictions.astype(np.float64)
e = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = e / e.sum(axis=1, keepdims=True)

err_idx = np.where(pred_test != y_test)[0]
print(f"ошибок: {len(err_idx)} из {len(y_test)}")
order = err_idx[np.argsort(-probs[err_idx, pred_test[err_idx]])]  # самые уверенные

for i in order[:3]:
    print("=" * 90)
    print(f"текст: {test_ds['text'][i]}")
    print(f"истина: {LABELS[y_test[i]]} | предсказание: {LABELS[pred_test[i]]}")
    print("вероятности:", {l: round(float(probs[i, k]), 4)
                           for k, l in enumerate(LABELS)})

# --- Задание 3, шаг 3.4: эксперимент варианта 16 (разметка >= 50%) ---
held_out = set(val_ds["text"]) | set(test_ds["text"])
ds50_pool = to_pool([ds50["train"]], lambda ex: ex["sentence"], lambda ex: ex["label"])
ds50_full = len(ds50_pool)
ds50_pool = ds50_pool.filter(lambda ex: ex["text"] not in held_out)
print(f"ds50: {ds50_full} -> после исключения val/test: {len(ds50_pool)}")

exp_train = ds50_pool.shuffle(seed=SEED).select(range(len(train_ds)))
print(f"экспериментальное обучение: {len(exp_train)} примеров "
      f"(base-разметка в обучении: {len(train_ds)})")

exp_enc = encode(exp_train)
exp_trainer = make_trainer("./out_exp", exp_enc)
t0 = time.time()
exp_trainer.train()
EXP_TIME = time.time() - t0
exp_metrics = exp_trainer.evaluate(test_enc, metric_key_prefix="test")
print(f"время обучения: {EXP_TIME:.1f} с")
print("test:", exp_metrics)

# таблица «постановка x Accuracy x Macro-F1 x время» для всех трёх моделей
results = pd.DataFrame([
    {"постановка": "TF-IDF + LogReg (baseline)",
     "Accuracy": BASELINE["accuracy"],
     "Macro-F1": BASELINE["f1_macro"],
     "время, с": round(BASELINE["time_s"], 1)},
    {"постановка": "DistilBERT, все аннотаторы согласны (основная)",
     "Accuracy": test_metrics["test_accuracy"],
     "Macro-F1": test_metrics["test_f1_macro"],
     "время, с": round(TRAIN_TIME, 1)},
    {"постановка": "DistilBERT, согласие >= 50% (эксперимент)",
     "Accuracy": exp_metrics["test_accuracy"],
     "Macro-F1": exp_metrics["test_f1_macro"],
     "время, с": round(EXP_TIME, 1)},
])
print(results.to_string(index=False))
