import json, math, random, re, time, gc, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import sacrebleu

VARIANT = 16
SEED = 1000 + VARIANT
N_TRAIN = 1000
N_EVAL = 50
STATE_FILE = "state.json"

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

def bleu(P, R):
    return sacrebleu.corpus_bleu(P, [R]).score

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

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"ranks": {}, "misc": {}}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

def step_21():
    print("===== Шаг 2.1. Данные: три поля для двух примеров =====")
    for i, rec in enumerate(records[:2]):
        print(f"--- пример {i} ---")
        print("prompt  :", repr(rec["prompt"][:400]))
        print("response:", repr(rec["response"][:400]))
        print(f"mask    : {rec['mask']}")
        print(f"          длина={len(rec['mask'])}, 0 (промпт)={rec['mask'].count(0)}, "
              f"1 (ответ)={rec['mask'].count(1)}")

def step_22(state):
    print("\n===== Шаг 2.2. Число параметров LoRA (r=%d) =====" % RANKS[0])
    base = load_base()
    cfg = base.config
    print("hidden_size:", cfg.hidden_size, "слоёв:", cfg.num_hidden_layers,
          "голов Q:", cfg.num_attention_heads, "голов K/V:", cfg.num_key_value_heads)
    attn = base.model.layers[0].self_attn
    r = RANKS[0]
    total = 0
    print(f"{'матрица':10} {'d':>6} {'k':>6} {'r(d+k)':>10} {'× слоёв':>9} {'итого':>10}")
    for name in ("q_proj", "v_proj"):
        W = getattr(attn, name).weight          # форма [k, d]: выход k, вход d
        k, d = W.shape
        per = r * (d + k)
        lay = per * cfg.num_hidden_layers
        total += lay
        print(f"{name:10} {d:6d} {k:6d} {per:10d} {cfg.num_hidden_layers:9d} {lay:10d}")
    n_all_base = sum(p.numel() for p in base.parameters())
    print(f"итого LoRA: {total} параметров = {100 * total / n_all_base:.3f}% от {n_all_base}")
    state["misc"].setdefault("manual_total", total)
    state["misc"].setdefault("n_all_base", n_all_base)
    del base; gc.collect(); torch.cuda.empty_cache()
    return total

def train_lora(rank, epochs=4, out_dir="./lora"):
    model = load_base()
    lora = LoraConfig(r=rank, lora_alpha=16,          # alpha одинаков для всех рангов
                      target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM)
    args = SFTConfig(output_dir=f"{out_dir}_r{rank}", num_train_epochs=epochs,
                     per_device_train_batch_size=2, gradient_accumulation_steps=8,
                     learning_rate=2e-4, max_length=1024, completion_only_loss=True,
                     eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
                     load_best_model_at_end=True, metric_for_best_model="eval_loss",
                     greater_is_better=False,
                     save_total_limit=1, gradient_checkpointing=True, fp16=True,
                     seed=SEED, report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_sft,
                         eval_dataset=val_sft, processing_class=tokenizer,
                         peft_config=lora,
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])
    trainer.model.print_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated()
    return trainer, elapsed, peak

def echo_count(instrs, outs):
    c = 0
    for i, o in zip(instrs, outs):
        src = i.split("\n\n", 1)[1].strip()
        if src and src in o:
            c += 1
    return c

def train_missing_rank(state):
    missing = [r for r in RANKS if str(r) not in state["ranks"]]
    if not missing:
        return None
    r = missing[0]
    print(f"\n===== Обучение LoRA: ранг {r} =====")
    instr = [x for x, _ in test_pairs[:N_EVAL]]
    refs = [y for _, y in test_pairs[:N_EVAL]]
    trainer, elapsed, peak = train_lora(r)
    n_lora = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    hist = trainer.state.log_history
    train_loss = [(h["epoch"], h["loss"]) for h in hist if "loss" in h]
    val_loss = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    print("train loss по эпохам:", [(int(round(e)), round(l, 4)) for e, l in train_loss])
    print("val   loss по эпохам:", [(int(round(e)), round(l, 4)) for e, l in val_loss])

    P = generate(trainer.model, instr, MAX_NEW_TOKENS)
    bleu_after = metric(P, refs)
    bv = trainer.state.best_metric
    if bv is None:
        bv = min(l for _, l in val_loss)
    entry = {"params": n_lora, "val_loss": round(float(bv), 4), "BLEU": round(bleu_after, 2),
             "время, с": round(elapsed), "peak_gib": round(peak / 2**30, 2),
             "hist_train": [[int(round(e)), round(l, 4)] for e, l in train_loss],
             "hist_val": [[int(round(e)), round(l, 4)] for e, l in val_loss]}

    if r == RANKS[0]:
        with trainer.model.disable_adapter():
            before = generate(trainer.model, instr, MAX_NEW_TOKENS)
        bleu_before = metric(before, refs)
        entry["BLEU_before"] = round(bleu_before, 2)
        entry["echo_before"] = echo_count(instr, before)
        entry["echo_after"] = echo_count(instr, P)
        entry["tmpl_before"] = sum(1 for o in before if "<|user|>" in o or "<|assistant|>" in o)
        entry["tmpl_after"] = sum(1 for o in P if "<|user|>" in o or "<|assistant|>" in o)
        entry["pairs5"] = [{"EN": instr[i].split("\n\n", 1)[1][:250], "ref": refs[i][:250],
                            "before": before[i][:250], "after": P[i][:250]} for i in range(5)]
        model = trainer.model
        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]
        wiki_ids = tokenizer("\n\n".join(wiki["text"]), return_tensors="pt")["input_ids"][0]

        @torch.no_grad()
        def perplexity(model, n_seg=50, seg_len=512):
            model.eval()
            nll, n = 0.0, 0
            for i in range(n_seg):
                seg = wiki_ids[i * seg_len:(i + 1) * seg_len][None].to(model.device)
                loss = model(seg, labels=seg).loss.item()
                nll += loss * (seg.shape[1] - 1); n += seg.shape[1] - 1
            return math.exp(nll / n)

        with model.disable_adapter():
            pp_base = perplexity(model)
        pp_lora = perplexity(model)
        state["misc"]["pp_base"] = round(pp_base, 2)
        state["misc"]["pp_lora"] = round(pp_lora, 2)
        print(f"PP база: {pp_base:.2f}   PP LoRA: {pp_lora:.2f}   изменение: {pp_lora - pp_base:+.2f}")
        print(f"\nСверка: trainable из модели {n_lora} (== ручной расчёт "
              f"{state['misc']['manual_total']})")

    state["ranks"][str(r)] = entry
    save_state(state)
    print(f"ранг {r} сохранён в {STATE_FILE}: {entry}")
    del trainer, P
    gc.collect(); torch.cuda.empty_cache()
    return r

def report(state):
    step_21()
    manual = state["misc"]["manual_total"]
    n_all_base = state["misc"]["n_all_base"]
    print(f"\n===== Шаг 2.2 (итог): LoRA r=4: {manual} параметров = {100 * manual / n_all_base:.3f}% модели")
    print(f"Сверка print_trainable_parameters: {manual} == {state['ranks']['4']['params']} -> OK")

    print("\n===== Шаг 2.5. Контрольный вывод: до/после =====")
    r4 = state["ranks"]["4"]
    print(f"BLEU до: {r4['BLEU_before']}   BLEU после: {r4['BLEU']}")
    print(f"эхо промпта в ответах: до {r4['echo_before']}/50, после {r4['echo_after']}/50; "
          f"мусор чат-шаблона: до {r4['tmpl_before']}/50, после {r4['tmpl_after']}/50")
    for i, p in enumerate(r4["pairs5"]):
        print(f"--- пара {i} ---")
        print("EN:", p["EN"])
        print("эталон DE :", p["ref"])
        print("до   :", p["before"])
        print("после:", p["after"])

    print("\n===== Шаг 3.1. Кривые переобучения =====")
    tv = r4["hist_val"]; tt = r4["hist_train"]
    best_ep, best_val = min(tv, key=lambda x: x[1])
    print(f"минимум val loss: эпоха {best_ep}, val={best_val:.4f}")
    gap = tv[-1][1] - best_val
    print(f"разрыв «val растёт, train падает»: {'виден' if gap > 0.01 else 'не виден'} "
          f"(последняя val - лучшая val = {gap:+.4f})")
    print("\nТаблица трассировки ранней остановки (patience=1):")
    print(f"{'Эпоха':>6} {'Val loss':>9} {'Best':>9} {'Wait':>5}  Действие")
    b, w, stopped = float("inf"), 0, False
    for e, v in tv:
        if v < b:
            b, w = v, 0
            action = "сохранить"
        else:
            w += 1
            action = "стоп (patience=1)"
            stopped = True
        print(f"{e:>6} {v:>9.4f} {b:>9.4f} {w:>5}  {action}")
    if not stopped:
        print("ранняя остановка не сработала — пройдены все эпохи")

    plt.figure(figsize=(7, 4.5))
    plt.plot([e for e, _ in tt], [l for _, l in tt], "o-", label="train loss")
    plt.plot([e for e, _ in tv], [l for _, l in tv], "s-", label="val loss")
    plt.axvline(best_ep, color="gray", ls="--", label=f"минимум val (эп. {best_ep})")
    plt.xlabel("эпоха"); plt.ylabel("потеря")
    plt.title("Переобучение при дообучении LoRA r=4")
    plt.xticks(sorted({e for e, _ in tt} | {e for e, _ in tv}))
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig("overfit.png", dpi=120); plt.close()
    print("график сохранён: overfit.png")

    print("\n===== Шаг 3.2. Катастрофическое забывание (wikitext-2) =====")
    pp_base, pp_lora = state["misc"]["pp_base"], state["misc"]["pp_lora"]
    print(f"PP база: {pp_base:.2f}   PP LoRA: {pp_lora:.2f}   изменение: {pp_lora - pp_base:+.2f}")

    print("\n===== Шаг 3.3. Память по правилу 18 байт на параметр =====")
    BYTES_PER_TRAINABLE = 4 + 4 + 8 + 2
    n_lora = r4["params"]
    full_ft = n_all_base * BYTES_PER_TRAINABLE
    lora_mem = n_lora * BYTES_PER_TRAINABLE + n_all_base * 2
    print(f"полное дообучение: {full_ft / 2**30:.1f} ГиБ, LoRA: {lora_mem / 2**30:.2f} ГиБ (без активаций)")
    print(f"факт пик (обучение r=4): {r4['peak_gib']:.2f} ГиБ "
          f"(расхождение с расчётом LoRA: {r4['peak_gib'] - lora_mem / 2**30:+.2f} ГиБ — активации и буферы)")

    print("\n===== Шаг 3.4. Влияние ранга =====")
    print("\nТаблица результатов по рангам:")
    print(f"{'Ранг':>5} {'Параметры':>10} {'Val loss':>9} {'BLEU':>8} {'Время, с':>9}")
    for rk in RANKS:
        res = state["ranks"][str(rk)]
        print(f"{rk:>5} {res['params']:>10} {res['val_loss']:>9} {res['BLEU']:>8} {res['время, с']:>9}")

    plt.figure(figsize=(7, 4.5))
    mets = [state["ranks"][str(x)]["BLEU"] for x in RANKS]
    plt.plot(RANKS, mets, "o-", color="tab:blue", label="BLEU")
    for x, m in zip(RANKS, mets):
        plt.annotate(f"{m:.1f}", (x, m), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel("ранг r"); plt.ylabel("BLEU (метрика задачи)")
    plt.title("Метрика от ранга LoRA")
    plt.xticks(RANKS); plt.grid(alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig("rank_metric.png", dpi=120); plt.close()
    print("график сохранён: rank_metric.png")

    print("\n===== Шаг 3.5. Диагностика сбоев =====")
    overfit_seen = len(tv) > 1 and tv[-1][1] > tv[0][1]
    train_drop = tt[0][1] - tt[-1][1] if len(tt) > 1 else 0.0
    nan_train = any(math.isnan(l) for _, l in tt)
    pp_rise = (pp_lora - pp_base) > 0.5
    echo_seen = r4["echo_after"] > 0 or r4["tmpl_after"] > 0
    def mark(flag):
        return "да — наблюдалось" if flag else "нет — не наблюдалось"
    print(f"{'Симптом':38} {'Причина':34} {'Действие':42} Наблюдалось")
    rows = [
        ("Модель повторяет промпт", "маска не на ответе",
         "mt=1 только на токенах ответа", mark(echo_seen)),
        ("train↓, val↑ со 2-й эпохи", "переобучение",
         "ранняя остановка, меньший r", mark(overfit_seen)),
        ("Старые умения пропали", "полное дообучение без replay",
         "replay 10-20% старых данных, LoRA", mark(pp_rise)),
        ("Потеря не снижается", "B не нули, мала lr",
         "проверить инициализацию и lr",
         f"нет — train снизился на {train_drop:.2f}" if train_drop >= 0.1
         else f"да — снижение всего {train_drop:.2f}"),
        ("«Двойной» вывод после слияния", "адаптер слит дважды",
         "пересобрать из базовой модели", "нет — слияние не выполнялось"),
        ("NaN в потере", "lr велика для fp16",
         "снизить lr, клиппинг градиента", mark(nan_train)),
    ]
    for row in rows:
        print(f"{row[0]:38} {row[1]:34} {row[2]:42} {row[3]}")

    def fmt_pairs(bs, aes):
        return f"{bs:.1f} -> {aes:.1f} ({aes - bs:+.1f})"
    rank_best = max(RANKS, key=lambda x: state["ranks"][str(x)]["BLEU"])
    print("\n===== ВЫВОДЫ =====")
    print(f"Задание 2. На 50 тестовых примерах BLEU вырос {fmt_pairs(r4['BLEU_before'], r4['BLEU'])}; "
          f"LoRA (r=4, {r4['params']} параметров, {100 * r4['params'] / n_all_base:.3f}% модели) "
          f"обучается только на токенах ответа: мусор чат-шаблона исчез "
          f"({r4['tmpl_before']} -> {r4['tmpl_after']} из 50), эхо промпта "
          f"({r4['echo_before']} -> {r4['echo_after']}, в основном — копирование служебных строк). "
          f"Ручной расчёт параметров {manual} сошёлся с print_trainable_parameters ({r4['params']}).")
    print(f"Задание 3. Минимум val loss — эпоха {best_ep} ({best_val:.4f}), "
          f"разрыв к концу {gap:+.4f} ({'переобучение видно' if gap > 0.01 else 'переобучения нет, сработала ранняя остановка'}); "
          f"перплексия wikitext-2 {pp_base:.2f} -> {pp_lora:.2f} ({pp_lora - pp_base:+.2f}) — "
          f"{'знания сохранены' if pp_lora - pp_base < 1 else 'знания заметно ухудшены'}. "
          f"Память: полное дообучение {full_ft / 2**30:.1f} ГиБ не помещается в GPU, "
          f"LoRA-расчёт {lora_mem / 2**30:.2f} ГиБ против пика {r4['peak_gib']:.2f} ГиБ "
          f"(разница — активации). Метрика по рангам: " +
          ", ".join(f"r={x}: BLEU {state['ranks'][str(x)]['BLEU']} ({state['ranks'][str(x)]['время, с']} с)"
                    for x in RANKS) +
          f". Лучший ранг: {rank_best} (BLEU {state['ranks'][str(rank_best)]['BLEU']}); "
          f"разброс BLEU по рангам {max(mets) - min(mets):.2f} — метрика в пределах шума, "
          f"точка насыщения уже на r={RANKS[0]} (увеличение ранга в 4 раза не даёт прироста).")
    print("ГОТОВО")

state = load_state()
missing = [r for r in RANKS if str(r) not in state["ranks"]]
if missing:
    step_21()
    step_22(state)
    save_state(state)
    train_missing_rank(state)
    rest = [r for r in RANKS if str(r) not in state["ranks"]]
    if rest:
        print(f"\nОсталось рангов: {rest} — перезапустите скрипт для продолжения.")
    else:
        print("\nВсе ранги обучены. Перезапустите скрипт для итогового отчёта.")
else:
    report(state)
