# Фраза "Садовник сажает дерево"
# Эмбеддинги:
# Слово - Вектор - Описание
# садовник - [1.0, 0.6, 0.9] - живое, активное, физ. объект
# сажает - [0.5, 0.7, 0.5] - одуш. действие, активное, процесс
# дерево - [0.3, −0.2, 0.9] - живое, пассивное, физ. объект
#
# Словарь: садовник=0, сажает=1, дерево=2, [EOS]=3
# Начальные веса LSTM (h = 2, d = 3):
#
# Wfx = 0.7 0.3 0.1
#       0.1 0.5 0.3,
#
# Wfh = 0.1  0
#       0  0.1,
#
# bf = 0
#
# Wix = 0.6 0.4 0.2
#       0.2 0.6 0.2,
#
# Wih = 0.1 0
#        0 0.1
#
# bi = 0
#
# Wcx = 0.8 0.2 0.1
#       0.1 0.6 0.2
#
# Wch = 0
# bc = 0
#
# Wox = 0.5 0.3 0.1
#       0.2 0.4 0.2
#
# Woh = 0.1 0
#        0 0.1
# bo = 0
#
# Why = 0.5 0.3
#       0.3 0.4
#       0.2 0.4
#       0.0 0.0
#
# by = 0

# Цели: шаг 1 (вход "садовник") -> предстазать "сажает" (индекс1); шаг 2->
# "дерево" (индекс2); шаг 3 -> [EOS] (индекс3)

import numpy as np

np.random.seed(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def tanh(x):
    return np.tanh(x)

def dtanh(y):
    return 1 - y**2

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# ===== ДАННЫЕ =====

embeddings = {
    'садовник': np.array([1.0, 0.6, 0.9]),
    'сажает': np.array([0.5, 0.7, 0.5]),
    'дерево': np.array([0.3, -0.2, 0.9]),
    'EOS': np.array([0.0, 0.0, 0.0])
}

word_to_idx = {'садовник': 0, 'сажает': 1, 'дерево': 2, 'EOS': 3}
idx_to_word = {v: k for k, v in word_to_idx.items()}

words = ['садовник', 'сажает', 'дерево']
targets = ['сажает', 'дерево', 'EOS']


# ===== ПАРАМЕТРЫ =====

d = 3
h_size = 2
vocab_size = 4
lr = 0.1


# случайная инициализация (лучше, чем фиксированная)
Wfx = np.array([[0.7, 0.3, 0.1],
                [0.1, 0.5, 0.3]])

Wfh = np.array([[0.1, 0],
                [0, 0.1]])
bf = np.zeros(2)


Wix = np.array([[0.6, 0.4, 0.2],
                [0.2, 0.6, 0.2]])

Wih = np.array([[0.1, 0],
                [0, 0.1]])
bi = np.zeros(2)


Wcx = np.array([[0.8, 0.2, 0.1],
                [0.1, 0.6, 0.2]])

Wch = np.zeros((2, 2))
bc = np.zeros(2)


Wox = np.array([[0.5, 0.3, 0.1],
                [0.2, 0.4, 0.2]])

Woh = np.array([[0.1, 0],
                [0, 0.1]])
bo = np.zeros(2)


Why = np.array([[0.5, 0.3],
                [0.3, 0.4],
                [0.2, 0.4],
                [0.0, 0.0]])

by = np.zeros(4)

for epoch in range(200):

    # ===== FORWARD =====
    xs, hs, cs = {}, {}, {}
    fs, is_, os, c_tildes = {}, {}, {}, {}
    ys, ps = {}, {}

    hs[-1] = np.zeros(h_size)
    cs[-1] = np.zeros(h_size)

    loss = 0

    for t in range(len(words)):
        x = embeddings[words[t]]
        xs[t] = x

        f = sigmoid(Wfx @ x + Wfh @ hs[t-1] + bf)
        i = sigmoid(Wix @ x + Wih @ hs[t-1] + bi)
        c_tilde = tanh(Wcx @ x + Wch @ hs[t-1] + bc)

        c = f * cs[t-1] + i * c_tilde
        o = sigmoid(Wox @ x + Woh @ hs[t-1] + bo)
        h = o * tanh(c)

        fs[t], is_[t], os[t], c_tildes[t] = f, i, o, c_tilde
        cs[t], hs[t] = c, h

        y = Why @ h + by
        p = softmax(y)

        ys[t], ps[t] = y, p

        target_idx = word_to_idx[targets[t]]
        loss += -np.log(p[target_idx] + 1e-9)

    # ===== BACKWARD (BPTT) =====

    dWfx = np.zeros_like(Wfx)
    dWfh = np.zeros_like(Wfh)
    dbf = np.zeros_like(bf)

    dWix = np.zeros_like(Wix)
    dWih = np.zeros_like(Wih)
    dbi = np.zeros_like(bi)

    dWcx = np.zeros_like(Wcx)
    dWch = np.zeros_like(Wch)
    dbc = np.zeros_like(bc)

    dWox = np.zeros_like(Wox)
    dWoh = np.zeros_like(Woh)
    dbo = np.zeros_like(bo)

    dWhy = np.zeros_like(Why)
    dby = np.zeros_like(by)

    dh_next = np.zeros(h_size)
    dc_next = np.zeros(h_size)

    for t in reversed(range(len(words))):

        target_idx = word_to_idx[targets[t]]

        dy = ps[t].copy()
        dy[target_idx] -= 1

        dWhy += np.outer(dy, hs[t])
        dby += dy

        dh = Why.T @ dy + dh_next

        do = dh * tanh(cs[t])
        do_raw = do * dsigmoid(os[t])

        dc = dh * os[t] * dtanh(tanh(cs[t])) + dc_next

        df = dc * cs[t-1]
        df_raw = df * dsigmoid(fs[t])

        di = dc * c_tildes[t]
        di_raw = di * dsigmoid(is_[t])

        dc_tilde = dc * is_[t]
        dc_tilde_raw = dc_tilde * dtanh(c_tildes[t])

        dWfx += np.outer(df_raw, xs[t])
        dWfh += np.outer(df_raw, hs[t-1])
        dbf += df_raw

        dWix += np.outer(di_raw, xs[t])
        dWih += np.outer(di_raw, hs[t-1])
        dbi += di_raw

        dWcx += np.outer(dc_tilde_raw, xs[t])
        dWch += np.outer(dc_tilde_raw, hs[t-1])
        dbc += dc_tilde_raw

        dWox += np.outer(do_raw, xs[t])
        dWoh += np.outer(do_raw, hs[t-1])
        dbo += do_raw

        dh_next = (
            Wfh.T @ df_raw +
            Wih.T @ di_raw +
            Wch.T @ dc_tilde_raw +
            Woh.T @ do_raw
        )

        dc_next = dc * fs[t]

    # ===== UPDATE =====
    for param, dparam in zip(
        [Wfx, Wfh, bf, Wix, Wih, bi, Wcx, Wch, bc, Wox, Woh, bo, Why, by],
        [dWfx, dWfh, dbf, dWix, dWih, dbi, dWcx, dWch, dbc, dWox, dWoh, dbo, dWhy, dby]
    ):
        param -= lr * dparam

    if epoch % 10 == 0:
        print(f"\ -----ЭПОХА {epoch}-----")
        for t in range(len(words)):
            x = embeddings[words[t]]

            f = sigmoid(Wfx @ x + Wfh @ h + bf)
            i = sigmoid(Wix @ x + Wih @ h + bi)
            c_tilde = tanh(Wcx @ x + Wch @ h + bc)
            c = f * c + i * c_tilde
            o = sigmoid(Wox @ x + Woh @ h + bo)
            h = o * tanh(c)


            print(f"\nСлово: {words[t]}")
            print(f"  f = [{f[0]:.3f}, {f[1]:.3f}]")
            print(f"  i = [{i[0]:.3f}, {i[1]:.3f}]")
            print(f"  c~ = [{c[0]:.3f}, {c[1]:.3f}]")
            print(f"  o = [{o[0]:.3f}, {o[1]:.3f}]")

            y = Why @ h + by
            p = softmax(y)

            pred = np.argmax(p)
            print(f"{words[t]} -> {idx_to_word[pred]} (p = { p })")
