#include "Tensor.hpp"

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

// конвертация в тензор
Tensor Matrix::ToTensor(const Matrix& m) {
    TensorSize size;
    size.depth = 1;
    size.height = m.rows;
    size.width = m.columns;

    Tensor t(size);

    int idx = 0;
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.columns; j++) {
            t.AtFlat(idx++) = m(i, j);
        }
    }

    return t;
}
