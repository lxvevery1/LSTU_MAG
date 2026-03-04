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

float* convolution(float* sig, const size_t sig_size, const float* core,
                   const size_t core_size, const int stride = 1, const int padding = 1) {
    size_t output_size = (sig_size - core_size) / stride + 1;

    float* res = new float[output_size];

    for (size_t i = 0; i <= sig_size - core_size; i += stride) {
        float sum = 0;
        for (size_t j = 0; j < core_size; j++)
            sum += sig[i + j] * core[j];

        res[i / stride] = sum;
    }

    print_arr(res, output_size);
    return res;
}

void task1() {
    const float* sig = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    const float* core = new float[3]{0.5, 1, 0.5};

    float* sig_test = new float[10]{1, 2, 3, 4, 5, 5, 4, 3, 2, 1};
    const float* core_test = new float[3]{-1, 0, 1};

    float* layer = new float[10]{1, 0, 2, 0, 3, 0, 4, 0, 5, 0};
    const float* yadro = new float[3]{0.5, 1, 0.5};

    convolution(layer, 10, yadro, 3, 1, 0);
    putchar('\n');
}

void task2() {
    float* sig = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    float* sig_test = new float[10]{1, 2, 3, 4, 5, 5, 4, 3, 2, 1};

    float* layer = new float[10]{1, 0, 2, 0, 3, 0, 4, 0, 5, 0};
    const float* yadro = new float[2]{-1, 1};

    const float* core = new float[2]{-1, 1};
    const float* core_test = new float[3]{-1, 0, 1};
    convolution(layer, 10, yadro, 2, 2, 0);
    putchar('\n');
}

void task3() {
    const float* core1 = new float[2]{1, -1};
    const float* core2 = new float[2]{0.5, 0.5};
    float* sig1 = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};
    float* sig2 = new float[10]{6, 12, 3, 9, 15, 6, 18, 9, 21, 12};

    auto res1 = convolution(sig1, 10, core1, 2);
    auto res2 = convolution(sig2, 10, core2, 2);
    float* res = new float[10];
    for (int i = 0; i < 10; i++) {
        res[i] = res1[i] + res2[i];
    }
    print_arr(res, 10);
    putchar('\n');
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
}

void task5() {
    const float* core1 = new float[3]{0.5, 1, 0.5};
    const float* core2 = new float[2]{-1, 1};
    float* sig = new float[10]{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4};

    float* layer = new float[10]{1, 0, 2, 0, 3, 0, 4, 0, 5, 0};
    const float* yadro = new float[3]{-0.5, 1, 0.5};

    auto first = convolution(sig, 10, core1, 3);
    relu(first, 10);

    auto second = convolution(first, 10, core2, 2, 2, 0);
    relu(second, 10);

    putchar('\n');
}

int main() {
    task1();
    task2();
    task3();
    task4();
    task5();

    return 0;
}
