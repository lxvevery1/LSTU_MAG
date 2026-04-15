#include "Matrix.hpp"

// конструктор из заданных размеров
Matrix::Matrix(int rows, int columns) {
    this->rows = rows; // сохраняем число строк
    this->columns = columns; // сохраняем число столбцов

    values = std::vector<std::vector<double>>(rows, std::vector<double>(columns, 0)); // создаём векторы для значений матрицы
}

// индексация
double& Matrix::operator()(int i, int j) {
    return values[i][j];
}

// индексация
double Matrix::operator()(int i, int j) const {
    return values[i][j];
}
