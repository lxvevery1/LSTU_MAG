import numpy as np
import matplotlib.pyplot as plt

# ---------- INPUT ----------

input_matrix = np.array([
[0,0,0,0,0,0,0,0,0,0],
[0,0,1,1,1,1,1,1,0,0],
[0,1,0,0,2,2,0,0,1,0],
[0,1,0,0,2,2,0,0,1,0],
[0,1,0,2,2,2,2,0,1,0],
[0,1,0,2,2,2,2,0,1,0],
[0,1,0,0,2,2,0,0,1,0],
[0,1,0,0,2,2,0,0,1,0],
[0,0,1,1,1,1,1,1,0,0],
[0,0,0,0,0,0,0,0,0,0]
])

k1 = np.array([
[-1,-1,-1],
[-1, 8,-1],
[-1,-1,-1]
])

k2 = np.array([
[-1, 1,-1],
[ 1, 2, 1],
[-1, 1,-1]
])

k3 = np.array([
[ 1, 1, 1],
[ 1,-8, 1],
[ 1, 1, 1]
])

kernels = [k1, k2, k3]
names = ["Circle", "Cross", "Background"]

# ---------- DILATED CONV ----------

def pad_same_dilated(img, k, dilation):
    k_eff = k + (k - 1) * (dilation - 1)
    pad = k_eff // 2
    return np.pad(img, pad, mode='constant')

def conv2d_dilated(img, kernel, dilation=1):
    kh, kw = kernel.shape
    img = pad_same_dilated(img, kh, dilation)

    h, w = img.shape
    k_eff = kh + (kh - 1) * (dilation - 1)

    out_h = h - k_eff + 1
    out_w = w - k_eff + 1

    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            sum_val = 0
            for ki in range(kh):
                for kj in range(kw):
                    ii = i + ki * dilation
                    jj = j + kj * dilation
                    sum_val += img[ii, jj] * kernel[ki, kj]
            out[i, j] = sum_val

    return out

# ---------- ACTIVATIONS ----------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# ---------- VIS ----------

def show_triplet(title, a, b, c):
    fig, axes = plt.subplots(1, 3, figsize=(10,3))
    fig.suptitle(title)

    axes[0].imshow(a)
    axes[0].set_title("k1")

    axes[1].imshow(b)
    axes[1].set_title("k2")

    axes[2].imshow(c)
    axes[2].set_title("k3")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

def show_single(title, img):
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# ---------- MAIN ----------

for dilation in [1, 2]:
    print(f"\n===== DILATION = {dilation} =====")

    conv_maps = []
    sig_maps = []

    for k, name in zip(kernels, names):
        conv = conv2d_dilated(input_matrix, k, dilation)
        sig = sigmoid(conv)

        conv_maps.append(conv)
        sig_maps.append(sig)

        print(f"{name}: shape = {conv.shape}")

    # показать свертки
    show_triplet(f"Conv maps (dilation={dilation})",
                 conv_maps[0], conv_maps[1], conv_maps[2])

    # показать sigmoid
    show_triplet(f"Sigmoid maps (dilation={dilation})",
                 sig_maps[0], sig_maps[1], sig_maps[2])

    # ---------- COMBINATION ----------
    M = 0.5 * sig_maps[0] + 1.0 * sig_maps[1] - 0.3 * sig_maps[2]

    show_single(f"Combined M (dilation={dilation})", M)

    # ---------- TANH ----------
    M_tanh = tanh(M)

    show_single(f"Tanh(M) (dilation={dilation})", M_tanh)
