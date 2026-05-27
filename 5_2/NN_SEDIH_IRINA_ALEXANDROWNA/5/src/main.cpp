// последовательность с переходом через 0

#include <algorithm>
#include <iostream>

void print_arr(const float* arr, const int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << ' ';
    }
    putchar('\n');
}

// Одномерная свёртка.
// same=true: padding вычисляется так, чтобы output_size == sig_size (независимо от stride).
// out_size (если не nullptr) получает фактический размер выходного массива.
float* convolution(const float* sig, size_t sig_size, const float* core, size_t core_size,
                   int stride = 1, int padding = 0, bool same = false,
                   size_t* out_size = nullptr) {
    int pad_left = padding;
    int pad_right = padding;

    if (same) {
        // total_pad такой, чтобы (sig_size + total_pad - core_size) / stride + 1 == sig_size
        int total_pad = std::max(0, (int)((sig_size - 1) * stride + core_size - sig_size));
        pad_left = total_pad / 2;
        pad_right = total_pad - pad_left;
    }

    size_t padded_size = sig_size + pad_left + pad_right;
    float* padded = new float[padded_size]{};

    for (size_t i = 0; i < sig_size; i++)
        padded[i + pad_left] = sig[i];

    size_t output_size = (padded_size - core_size) / stride + 1;
    float* res = new float[output_size];

    for (size_t i = 0; i <= padded_size - core_size; i += stride) {
        float sum = 0;
        for (size_t j = 0; j < core_size; j++)
            sum += padded[i + j] * core[core_size - 1 - j];
        res[i / stride] = sum;
    }

    print_arr(res, output_size);

    if (out_size)
        *out_size = output_size;

    delete[] padded;
    return res;
}

// ─── Задание 1: базовая свёртка без паддинга, stride=1 ───
void task1() {
    float sig[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const float core[] = {0.5f, 1.0f, 0.5f};

    float* res = convolution(sig, 10, core, 3);
    putchar('\n');
    delete[] res;
}

// ─── Задание 2: same-паддинг, stride=2 ───
void task2() {
    float sig[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const float core[] = {-1.0f, 1.0f};

    float* res = convolution(sig, 10, core, 2, 2, 0, true);
    putchar('\n');
    delete[] res;
}

// ─── Задание 3: два канала, два ядра, поэлементное сложение ───
void task3() {
    const float core1[] = {1.0f, -1.0f};
    const float core2[] = {0.5f, 0.5f};
    float sig1[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    float sig2[] = {6, 12, 3, 9, 15, 6, 18, 9, 21, 12};

    size_t out_sz;

    std::cout << "convolution 1 (core1 on sig1):\n";
    float* res1 = convolution(sig1, 10, core1, 2, 1, 0, false, &out_sz);

    std::cout << "convolution 2 (core2 on sig2):\n";
    float* res2 = convolution(sig2, 10, core2, 2, 1, 0, false, &out_sz);

    float* sum12 = new float[out_sz];
    std::cout << "sum (core1*sig1 + core2*sig2):\n";
    for (size_t i = 0; i < out_sz; i++)
        sum12[i] = res1[i] + res2[i];
    print_arr(sum12, out_sz);

    // меняем ядра местами
    std::cout << "convolution 3 (core1 on sig2):\n";
    float* res3 = convolution(sig2, 10, core1, 2, 1, 0, false, &out_sz);

    std::cout << "convolution 4 (core2 on sig1):\n";
    float* res4 = convolution(sig1, 10, core2, 2, 1, 0, false, &out_sz);

    std::cout << "sum (core1*sig2 + core2*sig1):\n";
    for (size_t i = 0; i < out_sz; i++)
        sum12[i] = res3[i] + res4[i];
    print_arr(sum12, out_sz);

    putchar('\n');

    delete[] res1;
    delete[] res2;
    delete[] res3;
    delete[] res4;
    delete[] sum12;
}

// ─── ReLU ───
float relu(float x) { return std::max(0.0f, x); }
void relu(float* x, const int size) {
    for (int i = 0; i < size; i++)
        x[i] = relu(x[i]);
}

// ─── Задание 4: свёртка + ReLU, same-паддинг, stride=1 ───
void task4() {
    float sig[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const float core[] = {1.0f, -2.0f, 1.0f};

    size_t out_sz;
    float* res = convolution(sig, 10, core, 3, 1, 0, true, &out_sz);
    relu(res, out_sz);
    std::cout << "after ReLU: ";
    print_arr(res, out_sz);
    putchar('\n');

    delete[] res;
}

// ─── Задание 5: каскад из двух свёрток с ReLU ───
void task5() {
    const float core1[] = {0.5f, 1.0f, 0.5f};
    const float core2[] = {-1.0f, 1.0f};
    float sig[] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

    size_t out_sz;

    // Первая свёртка: same, stride=1
    std::cout << "step 1 (same conv + ReLU):\n";
    float* first = convolution(sig, 10, core1, 3, 1, 0, true, &out_sz);
    relu(first, out_sz);
    std::cout << "after ReLU: ";
    print_arr(first, out_sz);

    // Вторая свёртка: без паддинга, stride=2
    std::cout << "step 2 (no-pad conv, stride=2 + ReLU):\n";
    float* second = convolution(first, out_sz, core2, 2, 2, 0, false, &out_sz);
    relu(second, out_sz);
    std::cout << "after ReLU: ";
    print_arr(second, out_sz);

    putchar('\n');

    delete[] first;
    delete[] second;
}

// ─── main ───
int main() {
    std::cout << "task1 Базовая свёртка без паддинга, stride=1\n";
    task1();
    std::cout << "task2 Same-паддинг, stride=2\n";
    task2();
    std::cout << "task3 Два канала, два ядра, поэлементное сложение\n";
    task3();
    std::cout << "task4 Свёртка + ReLU, same-паддинг, stride=1\n";
    task4();
    std::cout << "task5 Каскад из двух свёрток с ReLU\n";
    task5();

    return 0;
}
