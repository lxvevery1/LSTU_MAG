/*
 * Просчитать одну (две) итерацию цикла обучения по delta-правилу однослойной
 * аналоговой однородной нейронной сети, состоящей из 3 нейронов и имеющей
 * сигмоидальную функцию активации (k = 0.8). Синаптические веса и обучающую
 * выборку задать случайным образом (не нули).
 */

#include <cmath>
#include <iostream>

double sigm(double x) {
    double s = x / (1 + (x));
    return s;
}

int main() {

    // 1 шаг, задаим матрицу весов случайным образом
    double wm[3][3] = {{0.2, 0.4, 0.6}, {0.5, 0.1, 0.3}, {0.7, 0.8, 0.2}};

    // 2 шаг: вектор X и D
    double x[3] = {1, 0.5, 0};
    double D[3] = {0, 0.4, 1};

    // 3 шаг: вычисление исходных значений нейронной сети
    double Y[3];
    double S[3];

    const double k = 0.8;

    std::cout << "S: " << '\n';
    for (int i = 0; i < 3; i++) {
        S[i] = x[0] * wm[0][i] + x[1] * wm[1][i] + x[2] * wm[2][i];
        std::cout << S[i] << '\n';
    }

    std::cout << "Y: " << '\n';
    for (int i = 0; i < 3; i++) {
        Y[i] = sigm(k * S[i]);
        std::cout << Y[i] << '\n';
    }

    // 4 шаг:
    double EPS[3];

    std::cout << "EPSs: " << '\n';
    for (int i = 0; i < 3; i++) {
        EPS[i] = D[i] - Y[i];
        std::cout << EPS[i] << '\n';
    }

    // 5 шаг: задаем n-коэффициент обучения от 0 до 1 и изменяем веса в соотв.
    // с методом обратного распространения ошибки
    const double n = 0.8;

    // thetha - для выходного слоя. Корректирует вес

    std::cout << "weights: " << '\n';
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            wm[j][i] += n * EPS[i] * x[j];
            std::cout << wm[j][i] << '\n';
        }
    }

    // 6 шаг: вычислим среднеквадратичную ошибку
    double eps = std::pow(EPS[0], 2) + std::pow(EPS[1], 2) + std::pow(EPS[2], 2);

    std::cout << "EPS_ full: " << '\n';
    std::cout << eps << '\n';

    // ПЕРЕСЧЕТ

    // пересчёт S
    std::cout << "S: " << '\n';
    for (int i = 0; i < 3; i++) {
        S[i] = x[0] * wm[0][i] + x[1] * wm[1][i] + x[2] * wm[2][i];
        std::cout << S[i] << '\n';
    }

    // пересчёт Y
    std::cout << "Y: " << '\n';
    for (int i = 0; i < 3; i++) {
        Y[i] = sigm(k * S[i]);
        std::cout << Y[i] << '\n';
    }

    // пересчёт EPS
    std::cout << "EPS: " << '\n';
    for (int i = 0; i < 3; i++) {
        EPS[i] = D[i] - Y[i];
        std::cout << EPS[i] << '\n';
    }

    std::cout << "weights: " << '\n';
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            wm[j][i] += n * EPS[i] * x[j];
            std::cout << wm[j][i] << '\n';
        }
    }

    // новая ошибка
    std::cout << "EPS_ full: " << '\n';
    double eps_new = std::pow(EPS[0], 2) + std::pow(EPS[1], 2) + std::pow(EPS[2], 2);
    std::cout << eps_new << '\n';
    return 0;
}
