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

# ---------- CONV ----------

def conv2d(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1), dtype=int)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)

    return out

# ---------- PADDING ----------

def pad_same(img):
    return np.pad(img, 1, mode='constant')

def pad_reflect(img):
    return np.pad(img, 1, mode='reflect')

# ---------- RELU ----------

def relu(x):
    return np.maximum(0, x)

# ---------- VISUALIZATION ----------

def show_maps(title, before, after):
    fig, axes = plt.subplots(1, 2)
    fig.suptitle(title)

    axes[0].imshow(before)
    axes[0].set_title("Before ReLU")

    axes[1].imshow(after)
    axes[1].set_title("After ReLU")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# ---------- MAIN ----------

paddings = {
    "VALID": input_matrix,
    "SAME": pad_same(input_matrix),
    "REFLECT": pad_reflect(input_matrix)
}

for p_name, p_img in paddings.items():
    for k, k_name in zip(kernels, kernel_names):

        conv = conv2d(p_img, k)
        conv_relu = relu(conv)

        title = f"{p_name} - {k_name}"
        show_maps(title, conv, conv_relu)
