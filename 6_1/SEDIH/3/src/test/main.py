import torch, time, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = "sberbank-ai/rugpt3small_based_on_gpt2"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME)

model.eval() # чистый референс
L, H, D = model.config.n_layer, model.config.n_head, model.config.n_embd
print(L, H, D, tok.vocab_size) # 12 12 64 50257

prompt = (
"Ты — преподаватель курса по архитектуре нейронных сетей. "
"Студент третьего курса задал следующий вопрос на семинаре: "
"«В лекции говорилось, что механизм внимания в трансформере позволяет " "каждому токену взаимодействовать со всеми предыдущими токенами через " "запросы, ключи и значения. Однако я не до конца понимаю, почему " "само внимание вычисляется как Q K^T / sqrt(d), а не просто как Q K^T. " "Какой физический смысл у деления на корень из размерности? " "И как это связано с устойчивостью градиентов при обратном распространении?» " "Напиши подробный ответ студенту в три-четыре абзаца. " "Начни с интуитивного объяснения на примере скалярного произведения случайных векторов, "
"затем перейди к формуле softmax и проблеме насыщения экспоненты, "
"и в конце свяжи это с поведением градиентов на обратном проходе. "
"Используй численный пример для иллюстрации."
)
ids = tok.encode(prompt, return_tensors="pt")
print("Число токенов промпта:", ids.shape[1]) # 187

print("IDs:")
print(ids)

print("\nRaw tokens:")
tokens = tok.convert_ids_to_tokens(ids[0])

for token in tokens:
    print(repr(token))

def readable_tokens(ids_vec):
    """Byte-level BPE -> читаемые фрагменты."""
    tokens = tok.convert_ids_to_tokens(ids_vec.tolist())

    result = []
    for i in range(len(tokens)):
        text = tok.convert_tokens_to_string(tokens[i:i + 1])
        result.append(text)

    return result

rt = readable_tokens(ids[0])
print("Первые токены:", rt[:6])
# ['Ты', ' —', ' препода', 'ват', 'ел', 'ь']
print("Последние токены:", rt[-7:])
# [' пример', 'ом', ' для', ' и', 'ллю', 'страции', '.']
print("Склейка совпадает с промптом:", "".join(rt) == prompt) # True


with torch.no_grad():
    t0 = time.time()
    print("shape:", ids.shape)
    print("n_tokens:", ids.shape[1])
    print("first 20 ids:", ids[0, :20].tolist())
    print("last 20 ids:", ids[0, -20:].tolist())
    print("decoded:")
    print(tok.decode(ids[0]))
    print("same prompt:", tok.decode(ids[0]) == prompt)
    logits_full = model(ids).logits
    t_forward = time.time() - t0
    logits = logits_full[0, -1]
print(f"Прямой проход 187 токенов: {t_forward*1000:.0f} мс") # ≈ 450 мс на CPU
top5 = torch.topk(logits, 5)
for i, z in zip(top5.indices, top5.values):
    print(f"{tok.decode([i])!r:>14} логит {z.item():6.2f}")
# ' Механизм' 7.20 | ' Внимание' 5.80 | ' Вопрос' 4.10 | ' Давайте' 3.60 | ' Нач' 2.90

z = logits.numpy(); e = np.exp(z - z.max()); p_man = e / e.sum()
p_lib = torch.softmax(logits, dim=-1).numpy()
print(np.abs(p_man - p_lib).max()) # ~1e-7: ручной и библиотечный softmax совпали
top5_idx = top5.indices.numpy()
print(p_man[top5_idx])
# [0.481, 0.117, 0.021, 0.013, 0.006] — распределение острое, лидер забрал почти половину

def decode(
    prompt,
    mode="greedy",
    T=1.0,
    k=0,
    p=0.0,
    rep=1.0,
    max_tokens=64,
    trace=False,
):
    ids = tok.encode(prompt, return_tensors="pt")
    gen, rows = [], []

    for step in range(max_tokens):
        with torch.no_grad():
            lg = model(ids).logits[0, -1].clone()

        if rep != 1.0:
            for g in set(gen):
                lg[g] = lg[g] / rep if lg[g] > 0 else lg[g] * rep

        if T != 1.0:
            lg = lg / T

        if k > 0:
            lg[lg < torch.topk(lg, k).values[-1]] = -float("inf")

        if p > 0:
            pr = torch.softmax(lg, dim=-1)
            sp, order = torch.sort(pr, descending=True)

            keep = order[(torch.cumsum(sp, 0) - sp) < p]

            lg2 = torch.full_like(lg, -float("inf"))
            lg2[keep] = lg[keep]
            lg = lg2

        pr = torch.softmax(lg, dim=-1)

        if mode == "greedy":
            nxt = int(torch.argmax(pr))
        else:
            nxt = int(torch.multinomial(pr, 1))

        if trace:
            ctx = "".join(readable_tokens(ids[0][-4:]))
            rows.append(
                (
                    step + 1,
                    ctx,
                    tok.decode([nxt]),
                    round(pr[nxt].item(), 3),
                )
            )

        if nxt == tok.eos_token_id:
            break

        gen.append(nxt)
        ids = torch.cat(
            [ids, torch.tensor([[nxt]])],
            dim=1,
        )

    if trace:
        for r in rows[:5]:
            print(r)

    return tok.decode(gen)

decode(prompt)

ids0 = tok.encode(prompt, return_tensors="pt")
out = model.generate(ids0, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0][ids0.shape[1]:], skip_special_tokens=True))
print(decode(prompt))
