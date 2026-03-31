/*

   1. Сверточный слой применяет набор обучаемых фильтров к входному изображению
   2. Функция активации - ReLU, Leaky ReLU, ELU, Sigmoid, Tanh
   3. Пулинг (Subsampling) - Max Pooling, Average Pooling
   4. Полносвязные слои - для итоговой классификации
   5. Функция потерь - Cross-Entropy (для классификации)
   6. Оптимизаторы - SGD, Momentum, NAG, RMSprop, Adam, Nadam, AdamW
   7. Регуляризация - Dropout, Batch Normalization, L2 регуляризация


*/
#include <iostream>
#include "Tensor.hpp"
#include "ConvLayer.hpp"

void example00() {
    TensorSize inputSize{3, 5, 5};
    ConvLayer conv(inputSize, 2, 3, 1, 2);

    // ===== INPUT =====
    Tensor X(inputSize);

    // channel 0
    double c0[5][5] = {
        {1,2,0,1,0},
        {2,0,0,0,1},
        {1,2,2,0,2},
        {2,2,2,0,1},
        {2,0,1,0,1}
    };

    // channel 1
    double c1[5][5] = {
        {1,2,2,1,2},
        {0,2,2,0,2},
        {1,2,2,1,1},
        {2,2,0,1,0},
        {2,2,1,0,0}
    };

    // channel 2
    double c2[5][5] = {
        {0,2,0,1,1},
        {2,0,2,1,1},
        {2,0,1,2,1},
        {0,1,0,1,2},
        {1,0,1,2,1}
    };

    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++) {
            X(0,i,j) = c0[i][j];
            X(1,i,j) = c1[i][j];
            X(2,i,j) = c2[i][j];
        }

    // ===== FILTER 0 =====
    double w0[3][3][3] = {
        {
            {-1,1,1},
            {-1,1,1},
            {0,0,1}
        },
        {
            {0,-1,1},
            {-1,0,-1},
            {1,0,0}
        },
        {
            {1,-1,0},
            {-1,0,-1},
            {0,1,-1}
        }
    };

    // ===== FILTER 1 =====
    double w1[3][3][3] = {
        {
            {0,0,-1},
            {1,0,0},
            {0,-1,0}
        },
        {
            {1,-1,0},
            {-1,1,0},
            {-1,1,1}
        },
        {
            {1,0,1},
            {1,-1,1},
            {-1,0,0}
        }
    };

    // записываем веса
    for (int d = 0; d < 3; d++)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++) {
                conv.SetWeight(0, d, i, j, w0[d][i][j]);
                conv.SetWeight(1, d, i, j, w1[d][i][j]);
            }
        }
    }


    // bias
    conv.SetBias(0, 1);
    conv.SetBias(1, 0);

    // ===== FORWARD =====
    Tensor Y = conv.Forward(X);

    std::cout << "Output:\n" << Y << std::endl;
}

void example01() {
    TensorSize inputSize{1, 4, 4};
    ConvLayer conv(inputSize, 1, 3, 0, 1);

    Tensor X(inputSize);

    double c0[4][4] = {
        {4,5,8,7},
        {1,8,8,8},
        {3,6,6,4},
        {6,5,7,8},
    };

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            X(0,i,j) = c0[i][j];

    double w0[3][3] = {
        {1,4,1},
        {1,4,3},
        {3,3,1}
    };

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            conv.SetWeight(0, 0, i, j, w0[i][j]);

    conv.SetBias(0, 0);

    Tensor Y = conv.Forward(X);

    std::cout << "Forward:\n" << Y << std::endl;
}

int main() {
    example00();
    example01();
    return 0;
}
