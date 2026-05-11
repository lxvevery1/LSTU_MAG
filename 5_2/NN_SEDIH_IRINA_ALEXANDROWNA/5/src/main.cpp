// последовательность с переходом через 0

#include <iostream>

float calc(const float* filter, const float* sig, const int size) {
    float result = 0;
    for (int i = 0; i < size; i++)
        result += sig[i] * filter[i];
    return result;
}

float task_test(const float* sig, const float* core) { return calc(sig, core, 3); }

void print_arr(const float* arr, const int size) {
    for (int i = 0; i < size; i++) {
        std::cout << arr[i] << ' ';
    }
    putchar('\n');
}

void apply_zeros(float* sig, const size_t size, const int padding) {
    for (int i = 0; i < padding; i++) {
        sig[i] = 0;
    }

    for (int i = size - padding; i < size; i++) {
        sig[i] = 0;
    }
}

float* convolution(float* sig, size_t sig_size, const float* core, size_t core_size,
                   int stride = 1, int padding = 0, bool same = false) {
    if (same)
        padding = ( core_size - 1 ) / 2; // игнор  padding

    size_t padded_size = sig_size + padding * 2;

    float* padded = new float[padded_size]{0};

    for (size_t i = 0; i < sig_size; i++)
        padded[i + padding] = sig[i];

    size_t output_size = (padded_size - core_size) / stride + 1;

    float* res = new float[output_size];

    for (size_t i = 0; i <= padded_size - core_size; i += stride) {
        float sum = 0;

        for (size_t j = 0; j < core_size; j++)
            sum += padded[i + j] * core[core_size - 1 - j];

        res[i / stride] = sum;
    }

    // если режим same — обрезаем центр
    if (same) {
        float* same_res = new float[sig_size];

        size_t start = (output_size - sig_size) / 2;

        for (size_t i = 0; i < sig_size; i++)
            same_res[i] = res[i + start];

        print_arr(same_res, sig_size);

        delete[] padded;
        delete[] res;

        return same_res;
    }

    print_arr(res, output_size);

    delete[] padded;
    return res;
}

void task1() {
    float* sig = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const float* core = new float[3]{0.5, 1, 0.5};

    convolution(sig, 10, core, 3, 1, 0);
    putchar('\n');

    delete[] sig;
    delete[] core;
}

void task2() {
    float* sig = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const float* core = new float[2]{-1, 1};

    // add zeros
    convolution(sig, 10, core, 2, 2, 0, true);
    putchar('\n');

    delete[] sig;
    delete[] core;
}

void task3() {
    const float* core1 = new float[2]{1, -1};
    const float* core2 = new float[2]{0.5, 0.5};
    float* sig1 = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    float* sig2 = new float[10]{6, 12, 3, 9, 15, 6, 18, 9, 21, 12};

    std::cout << "convolution 1: " << '\n';
    auto res1 = convolution(sig1, 10, core1, 2);
    std::cout << "convolution 2: " << '\n';
    auto res2 = convolution(sig2, 10, core2, 2);

    float* res = new float[10];
    std::cout << "result : " << '\n';
    for (int i = 0; i < 10; i++) {
        res[i] = res1[i] + res2[i];
    }

    auto res3 = convolution(sig2, 10, core1, 2);
    auto res4 = convolution(sig1, 10, core2, 2);
    std::cout << "result : " << '\n';
    for (int i = 0; i < 10; i++) {
        res[i] = res3[i] + res4[i];
    }

    print_arr(res, 10);

    putchar('\n');

    delete[] core1;
    delete[] core2;
    delete[] sig1;
    delete[] sig2;
    delete[] res;
}

float relu(float x) { return std::max(0.0f, x); }
void relu(float* x, const int size) {
    for (int i = 0; i < size; i++) {
        x[i] = relu(x[i]);
    }
}

void task4() {
    float* sig = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const float* core = new float[3]{1, -2, 1};

    auto res = convolution(sig, 10, core, 2, 1, 0);
    relu(res, 10);
    print_arr(res, 10);
    putchar('\n');

    delete[] sig;
    delete[] core;
}

void task5() {
    const float* core1 = new float[3]{0.5, 1, 0.5};
    const float* core2 = new float[2]{-1, 1};
    float* sig = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

    auto first = convolution(sig, 10, core1, 3);
    relu(first, 10);

    auto second = convolution(first, 10, core2, 2, 2, 0);
    relu(second, 10);

    putchar('\n');

    delete[] sig;
    delete[] core1;
    delete[] core2;
}

int main() {
    task1();
    task2();
    task3();
    task4();
    task5();

    return 0;
}
