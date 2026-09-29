#!/usr/bin/env python3
"""Решение варианта 16 задания лекции 3.

Пять шагов: softmax с температурой, фильтр top-k, жадный выбор токена,
штраф за повторы на втором шаге, метрики инференса и память KV-кэша.

Использование чисел — только для самопроверки. По условию задания код при сдаче
запрещён, таблица ответов заполняется вручную на калькуляторе.
"""

import math
import sys

TOKENS = ["A", "B", "C", "D", "E"]

# Фиксированная конфигурация модели для шага 5.
LAYERS = 32
KV_HEADS = 8
HEAD_DIM = 128
BATCH = 1

# Данные варианта
LOGITS_1 = [2.5, 2.4, 0.5, 0.4, 0.3]  # набор P5
TEMPERATURE = 0.7
TOP_K = 4

LOGITS_2 = [2.1, 1.6, 1.8, 2.4, 0.9]
RHO = 1.2
TTFT = 0.25
TPOT = 0.020
N_TOKENS = 128
CONTEXT = 4096
BYTES_PER_VALUE = 2


def softmax_with_temperature(logits, temperature):
    """Шаг 1. Деление на T, вычитание максимума, экспонента, нормировка."""
    scaled = [z / temperature for z in logits]
    shift = max(scaled)
    shifted = [s - shift for s in scaled]
    exps = [math.exp(s) for s in shifted]
    total = sum(exps)
    probs = [e / total for e in exps]
    return scaled, shift, shifted, exps, total, probs


def build_core_top_k(probs, k):
    """Шаг 2. Ядро из k наиболее вероятных токенов, порядок убывания."""
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    return order[:k], order


def sample(probs, core, u):
    """Шаг 3. Сэмплирование по кумулятивной функции на ядре."""
    total = sum(probs[i] for i in core)
    edges, cumulative = [], 0.0
    for i in core:
        cumulative += probs[i] / total
        edges.append((min(cumulative, 1.0), i))
    chosen = core[-1]  # страховка от ошибок округления на последнем интервале
    for edge, i in edges:
        if u < edge:
            chosen = i
            break
    return chosen, edges, total


def apply_repeat_penalty(logits, index, rho):
    """Шаг 4. Штраф ровно одному индексу: положительный логит делим, иначе умножаем."""
    modified = list(logits)
    value = logits[index]
    modified[index] = value / rho if value > 0 else value * rho
    return modified


def argmax_index(values):
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def fmt_aligned(value, width=10):
    return f"{value:>{width}.4f}"


def solve():
    print("Вариант 16")
    print(f"z1 = {LOGITS_1},  T = {TEMPERATURE},  top-k = {TOP_K},  выбор: жадно")
    print(f"z2 = {LOGITS_2},  rho = {RHO},  TTFT = {TTFT} с,  TPOT = {TPOT} с,  "
          f"N = {N_TOKENS},  S = {CONTEXT},  {BYTES_PER_VALUE} байт")
    print()

    # Шаг 1.
    scaled, shift, shifted, exps, total, probs = softmax_with_temperature(
        LOGITS_1, TEMPERATURE
    )
    print("Шаг 1. Softmax с температурой")
    header = f"{'Токен':<7}{'логит':>9}{'z/T':>10}{'z/T - max':>11}{'Экспонента':>12}{'Вероятность':>13}"
    print(header)
    print("-" * len(header))
    for i, token in enumerate(TOKENS):
        print(f"{token:<7}{LOGITS_1[i]:>9.1f}{scaled[i]:>10.4f}{shifted[i]:>11.4f}"
              f"{exps[i]:>12.4f}{probs[i]:>13.4f}")
    print(f"\nСумма экспонент {total:.4f}, сумма вероятностей {sum(probs):.4f} "
          f"(контроль 1.000, допуск 0.001)")
    print()

    # Шаг 2.
    core, ranked = build_core_top_k(probs, TOP_K)
    core_total = sum(probs[i] for i in core)
    weights = {i: probs[i] / core_total for i in core}
    print("Шаг 2. Фильтр top-k и перенормировка")
    print("  ранжирование: " + ", ".join(
        f"{TOKENS[i]} = {probs[i]:.4f}" for i in ranked))
    print(f"  берём первые {TOP_K}, порог отсечения = {probs[ranked[TOP_K - 1]]:.4f}")
    print(f"Ядро: {', '.join(TOKENS[i] for i in core)}, сумма ядра {core_total:.4f}, "
          f"отсечено {100 * (1 - core_total):.1f}% массы")
    for i in core:
        print(f"  {('p' + TOKENS[i]):<3} {fmt_aligned(probs[i], 7)} / "
              f"{core_total:.4f} = {weights[i]:.4f}")
    print()

    # Шаг 3.
    chosen = max(core, key=lambda i: weights[i])
    print("Шаг 3. Выбор токена")
    print("  жадный выбор: максимальный вес в ядре")
    print(f"Выбран токен {TOKENS[chosen]} с вероятностью {weights[chosen]:.4f}, "
          f"индекс {chosen}")
    print()

    # Шаг 4.
    modified = apply_repeat_penalty(LOGITS_2, chosen, RHO)
    with_penalty = argmax_index(modified)
    without_penalty = argmax_index(LOGITS_2)
    print("Шаг 4. Второй шаг со штрафом за повторы")
    raw = LOGITS_2[chosen]
    operation = "/" if raw > 0 else "*"
    print(f"  штраф применён к одному индексу: {chosen} ({TOKENS[chosen]}), "
          f"логит {raw} положителен -> {raw} {operation} {RHO} = {modified[chosen]:.4f}")
    for i, token in enumerate(TOKENS):
        mark = " <- штраф" if i == chosen else ""
        print(f"  {token}: было {LOGITS_2[i]:>6.2f}  стало {modified[i]:>6.4f}{mark}")
    print(f"  жадный выбор со штрафом: {TOKENS[with_penalty]} ({max(modified):.4f})")
    print(f"  жадный выбор без штрафа: {TOKENS[without_penalty]} ({max(LOGITS_2):.4f})")
    if with_penalty != without_penalty:
        print("  вывод: штраф развернул решение, предотвратив повтор-зацикливание")
    else:
        print(f"  вывод: решения не разошлись, штраф ничего не изменил — без штрафа "
              f"максимум и так был бы на {TOKENS[without_penalty]}, а не на "
              f"{TOKENS[chosen]}")
    print()

    # Шаг 5.
    latency = TTFT + TPOT * (N_TOKENS - 1)
    throughput = N_TOKENS / latency
    memory = 2 * LAYERS * KV_HEADS * HEAD_DIM * CONTEXT * BATCH * BYTES_PER_VALUE
    mib = memory / 1024 ** 2
    gib = memory / 1024 ** 3
    unit = f"{mib:.0f} МиБ" if gib < 1 else f"{mib:.0f} МиБ = {gib:.2f} ГиБ"
    print("Шаг 5. Метрики и память")
    print(f"  latency = TTFT + TPOT * (N - 1) = {TTFT} + {TPOT} * {N_TOKENS - 1} "
          f"= {latency:.2f} с")
    print(f"  throughput = N / latency = {N_TOKENS} / {latency:.2f} = {throughput:.1f} ток/с")
    print(f"  потолок по TPOT: N / TPOT = {N_TOKENS} / {TPOT} = {N_TOKENS / TPOT:.1f} "
          f"ток/с (разрыв — стоимость префикса)")
    print(f"  KV-кэш = 2 * L * H_kv * D_head * S * B * байт = 2 * {LAYERS} * {KV_HEADS} * "
          f"{HEAD_DIM} * {CONTEXT} * {BATCH} * {BYTES_PER_VALUE} = {memory:,} байт".replace(",", " "))
    print(f"  KV-кэш = {unit}")
    print()

    # Итоговая таблица сдачи.
    print("Итоговая таблица (шесть строк)")
    print(f"  1. Вероятности: [{', '.join(f'{p:.4f}' for p in probs)}], "
          f"сумма {sum(probs):.4f}")
    print(f"  2. Ядро {', '.join(TOKENS[i] for i in core)}; веса "
          f"{', '.join(f'{weights[i]:.4f}' for i in core)}")
    print(f"  3. Выбран {TOKENS[chosen]} с вероятностью {weights[chosen]:.4f} "
          f"(индекс {chosen})")
    print(f"  4. Модифицированные логиты [{', '.join(f'{m:.4f}' for m in modified)}]; "
          f"со штрафом {TOKENS[with_penalty]}, без штрафа {TOKENS[without_penalty]}")
    print(f"  5. Latency {latency:.2f} с, throughput {throughput:.1f} ток/с")
    print(f"  6. Память KV-кэша {memory:,} байт = {unit}".replace(",", " "))


def check_reference():
    """Сверка функций с разобранным примером из части 1 задания."""
    print("Сверка с эталоном из части 1")
    print("-" * 40)
    expected = {
        "probs": [0.8293, 0.1122, 0.0413, 0.0152, 0.0021],
        "core": [0, 1],
        "weights": [0.8808, 0.1192],
        "chosen": 1,
        "modified": [2.2, 2.0833, 0.9, 1.1, 0.4],
        "with_penalty": 0,
        "without_penalty": 1,
        "latency": 3.23,
        "throughput": 46.4,
        "memory": 268435456,
    }
    _, _, _, _, total, probs = softmax_with_temperature([2.0, 1.0, 0.5, 0.0, -1.0], 0.5)
    core = sorted(range(5), key=lambda i: probs[i], reverse=True)[:2]  # ядро top-p 0.9
    core_total = sum(probs[i] for i in core)
    weights = {i: probs[i] / core_total for i in core}
    chosen, _, _ = sample(probs, core, 0.90)
    modified = apply_repeat_penalty([2.2, 2.5, 0.9, 1.1, 0.4], chosen, 1.2)
    latency = 0.25 + 0.02 * (150 - 1)
    throughput = 150 / latency
    memory = 2 * LAYERS * KV_HEADS * HEAD_DIM * 2048 * BATCH * 2

    cases = [
        ("сумма экспонент", total, 1.2059, 1e-3),
        ("вероятности", probs, expected["probs"], 1e-4),
        ("ядро", core, expected["core"], 0),
        ("веса", [weights[i] for i in core], expected["weights"], 1e-4),
        ("выбранный индекс", chosen, expected["chosen"], 0),
        ("модифицированные логиты", modified, expected["modified"], 1e-4),
        ("выбор со штрафом", argmax_index(modified), expected["with_penalty"], 0),
        ("выбор без штрафа", argmax_index([2.2, 2.5, 0.9, 1.1, 0.4]),
         expected["without_penalty"], 0),
        ("latency", latency, expected["latency"], 1e-9),
        ("throughput", throughput, expected["throughput"], 0.05),
        ("память", memory, expected["memory"], 0),
    ]
    ok = True
    for name, got, want, tol in cases:
        if isinstance(want, list):
            good = len(got) == len(want) and all(
                abs(a - b) <= tol for a, b in zip(got, want)
            )
        else:
            good = abs(got - want) <= tol
        ok = ok and good
        mark = "OK " if good else "РАСХОЖДЕНИЕ"
        if isinstance(want, list):
            print(f"  {mark} {name}: {[round(v, 4) for v in got]}")
        else:
            print(f"  {mark} {name}: {got}")
    print()
    print("Эталон сходится" if ok else "Эталон не сошёлся")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_reference()
    else:
        solve()


if __name__ == "__main__":
    main()
