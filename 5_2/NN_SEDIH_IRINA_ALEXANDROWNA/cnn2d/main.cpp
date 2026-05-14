#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>

using Matrix = std::vector<std::vector<int>>;

// ---------- PRINT ----------

void print_pair(const Matrix& before, const Matrix& after, const std::string& name) {
    std::cout << "==== " << name << " ====\n";

    int h = before.size();
    int w = before[0].size();

    for (int i = 0; i < h; ++i) {
        // BEFORE
        for (int j = 0; j < w; ++j)
            std::cout << std::setw(4) << before[i][j];

        std::cout << "   |   ";

        // AFTER
        for (int j = 0; j < w; ++j)
            std::cout << std::setw(4) << after[i][j];

        std::cout << "\n";
    }

    std::cout << "\n";
}

// ---------- PADDING ----------

Matrix pad_same_zero(const Matrix& input, int pad) {
    int h = input.size();
    int w = input[0].size();

    Matrix out(h + 2*pad, std::vector<int>(w + 2*pad, 0));

    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            out[i+pad][j+pad] = input[i][j];

    return out;
}

Matrix pad_reflect(const Matrix& input, int pad) {
    int h = input.size();
    int w = input[0].size();

    Matrix out(h + 2*pad, std::vector<int>(w + 2*pad));

    for (int i = 0; i < h + 2*pad; ++i) {
        for (int j = 0; j < w + 2*pad; ++j) {
            int ii = i - pad;
            int jj = j - pad;

            if (ii < 0) ii = -ii - 1;
            if (jj < 0) jj = -jj - 1;
            if (ii >= h) ii = 2*h - ii - 1;
            if (jj >= w) jj = 2*w - jj - 1;

            out[i][j] = input[ii][jj];
        }
    }

    return out;
}

// ---------- CONV ----------

Matrix conv2d(const Matrix& input, const Matrix& kernel) {
    int h = input.size();
    int w = input[0].size();
    int kh = kernel.size();
    int kw = kernel[0].size();

    int out_h = h - kh + 1;
    int out_w = w - kw + 1;

    Matrix out(out_h, std::vector<int>(out_w, 0));

    for (int i = 0; i < out_h; ++i) {
        for (int j = 0; j < out_w; ++j) {
            int sum = 0;
            for (int ki = 0; ki < kh; ++ki)
                for (int kj = 0; kj < kw; ++kj)
                    sum += input[i+ki][j+kj] * kernel[ki][kj];

            out[i][j] = sum;
        }
    }

    return out;
}

// ---------- RELU ----------

Matrix relu_copy(const Matrix& input) {
    Matrix out = input;

    for (auto& row : out)
        for (auto& x : row)
            x = std::max(0, x);

    return out;
}

// ---------- MAIN ----------

int main() {

    Matrix input = {
        {0,0,0,0,0,0,0,0,0,0},
        {0,0,1,1,1,1,1,1,0,0},
        {0,1,0,0,2,2,0,0,1,0},
        {0,1,0,0,2,2,0,0,1,0},
        {0,1,0,2,2,2,2,0,1,0},
        {0,1,0,2,2,2,2,0,1,0},
        {0,1,0,0,2,2,0,0,1,0},
        {0,1,0,0,2,2,0,0,1,0},
        {0,0,1,1,1,1,1,1,0,0},
        {0,0,0,0,0,0,0,0,0,0}
    };

    Matrix k1 = {
        {-1,-1,-1},
        {-1, 8,-1},
        {-1,-1,-1}
    };

    Matrix k2 = {
        {-1, 1,-1},
        { 1, 2, 1},
        {-1, 1,-1}
    };

    Matrix k3 = {
        { 1, 1, 1},
        { 1,-8, 1},
        { 1, 1, 1}
    };

    int pad = 1;

    // ---------- VALID ----------
    auto v1 = conv2d(input, k1);
    auto v2 = conv2d(input, k2);
    auto v3 = conv2d(input, k3);

    // ---------- SAME ----------
    auto s_input = pad_same_zero(input, pad);
    auto s1 = conv2d(s_input, k1);
    auto s2 = conv2d(s_input, k2);
    auto s3 = conv2d(s_input, k3);

    // ---------- REFLECT ----------
    auto r_input = pad_reflect(input, pad);
    auto r1 = conv2d(r_input, k1);
    auto r2 = conv2d(r_input, k2);
    auto r3 = conv2d(r_input, k3);

    // ---------- RELU ----------
    print_pair(v1, relu_copy(v1), "VALID - K1");
    print_pair(v2, relu_copy(v2), "VALID - K2");
    print_pair(v3, relu_copy(v3), "VALID - K3");

    print_pair(s1, relu_copy(s1), "SAME - K1");
    print_pair(s2, relu_copy(s2), "SAME - K2");
    print_pair(s3, relu_copy(s3), "SAME - K3");

    print_pair(r1, relu_copy(r1), "REFLECT - K1");
    print_pair(r2, relu_copy(r2), "REFLECT - K2");
    print_pair(r3, relu_copy(r3), "REFLECT - K3");

    return 0;
}
