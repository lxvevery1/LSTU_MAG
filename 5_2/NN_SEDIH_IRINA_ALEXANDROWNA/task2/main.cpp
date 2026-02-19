/* Разделить на классы, построив разделяющую линию.
 *
 * Для заданной матрицы входных данных P и заданнго вектора выходных значений T (из таблицы)
 * разработать простой персептрон, решающий задачу классификации 2-х классов. Решение найти с
 * помощью графического построения границы решения и вычисления весов и смещения вручную.
 * Протестировать с помощью компьютера полученного решение для всех входных векторов (столбцов
 * матрицы P)
 *
 * Вариант 17 (17%15=2):
 * Матрица P -> [0 1 1 0]
 *              [0 0 1 1]
 *
 * Вектор T ->  [0 0 1 1]
 *
 * */

#include <iostream>

double perceptron(double x1, double x2, double w1, double w2, double b) {
    double p = x1 * w1 + x2 * w2 + b;
    return p;
}

// корректировка весов
bool errorCheck(double* errors, size_t errors_size) {
    for (int i = 0; i < errors_size; i++) {
        if (errors[i] != 0) {
            return false;
        }
    }

    return true;
}

int main() {
    double P[2][4] = {{0, 1, 1, 0}, {0, 0, 1, 1}};
    double T[4] = {0, 0, 1, 1};

    double w1 = 0.5;
    double w2 = 0.5;
    double b = 0.5;

    double y = 0;

    double error = 0;
    double errors[4] = {1, 1, 1, 1};
    const size_t epoch = 2000;

    const double lern = 0.001;

    for (size_t j = 0; j < epoch; j++) {
        for (size_t e = 0; e < 4; e++) {
            errors[e] = 1;
        }
        for (size_t i = 0; i < 4; i++) {
            double x1 = P[0][i];
            double x2 = P[1][i];

            y = perceptron(x1, x2, w1, w2, b);

            int a = (y >= 0) ? 1 : 0;
            error = T[i] - a;

            w1 += lern * error * x1;
            w2 += lern * error * x2;
            b += lern * error;

            errors[i] = error;

            std::cout << "Input x1=" << x1 << ", x2=" << x2 << '\n';
            std::cout << "y=" << y << '\n';
            std::cout << "output=" << a << '\n';
            std::cout << "Error=" << error << '\n';
            std::cout << '\n';

            if (errorCheck(errors, 4)) {
                std::cout << "Exit at epoch=" << j << '\n';
                return 0;
            }
        }

        std::cout << "w1=" << w1 << '\n';
        std::cout << "w2=" << w2 << '\n';
        std::cout << "b=" << b << '\n';
    }

    return 0;
}
