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
kernel_names = ["Circle", "Cross", "Background"]

# ---------- SAME PADDING ----------

def pad_same(img, kernel_size):
    pad = kernel_size // 2
    return np.pad(img, pad, mode='constant')

# ---------- CONV WITH STRIDE ----------

def conv2d_stride(img, kernel, stride):
    kh, kw = kernel.shape
    img = pad_same(img, kh)

    h, w = img.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = img[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(region * kernel)

    return out

# ---------- ACTIVATION ----------

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# ---------- POOLING ----------

def max_pool(x, size=2, stride=2):
    h, w = x.shape
    out_h = h // stride
    out_w = w // stride
    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = x[i*stride:i*stride+size, j*stride:j*stride+size]
            out[i, j] = np.max(region)

    return out

def avg_pool(x, size=2, stride=2):
    h, w = x.shape
    out_h = h // stride
    out_w = w // stride
    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = x[i*stride:i*stride+size, j*stride:j*stride+size]
            out[i, j] = np.mean(region)

    return out

# ---------- VISUALIZATION ----------

def show_all(title, conv, act, maxp, avgp):
    fig, axes = plt.subplots(1, 4, figsize=(12,3))
    fig.suptitle(title)

    axes[0].imshow(conv)
    axes[0].set_title("Conv")

    axes[1].imshow(act)
    axes[1].set_title("Leaky ReLU")

    axes[2].imshow(maxp)
    axes[2].set_title("MaxPool")

    axes[3].imshow(avgp)
    axes[3].set_title("AvgPool")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# ---------- MAIN ----------

for stride in [1, 2]:
    print(f"\n===== STRIDE = {stride} =====")

    for k, name in zip(kernels, kernel_names):
        conv = conv2d_stride(input_matrix, k, stride)
        act = leaky_relu(conv)
        maxp = max_pool(act)
        avgp = avg_pool(act)

        print(f"{name}:")
        print(f"Conv size: {conv.shape}")
        print(f"After MaxPool: {maxp.shape}")
        print(f"After AvgPool: {avgp.shape}\n")

        show_all(f"{name} | stride={stride}", conv, act, maxp, avgp)
