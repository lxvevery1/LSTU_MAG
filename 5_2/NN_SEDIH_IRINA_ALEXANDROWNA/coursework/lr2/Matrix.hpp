#pragma once

#include <vector>

class Tensor; //forward declaration

class Matrix {
    int rows; // число строк
    int columns; // число столбцов
    std::vector<std::vector<double>> values; // значения

public:
    Matrix(int rows, int columns); // конструктор из заданных размеров

    double& operator()(int i, int j); // индексация
    double operator()(int i, int j) const; // индексация

    static Tensor ToTensor(const Matrix& m);
};
