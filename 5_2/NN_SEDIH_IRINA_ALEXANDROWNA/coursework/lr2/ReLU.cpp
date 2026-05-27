#include "ReLU.hpp"

// создание слоя
ReLULayer::ReLULayer(TensorSize size) {
    this->size = size; // сохраняем размер
}

// прямое распространение
Tensor ReLULayer::Forward(const Tensor &X) {
    Tensor output(size); // создаём выходной тензор

    // проходимся по всем значениям входного тензора
    for (int i = 0; i < size.height; i++)
        for (int j = 0; j < size.width; j++)
            for (int k = 0; k < size.depth; k++)
                output(k, i, j) = X(k, i, j) > 0 ? X(k, i, j) : 0; // вычисляем значение функции активации

    return output; // возвращаем выходной тензор
}

Tensor ReLULayer::Backward(const Tensor &dout, const Tensor &X) {
    Tensor dX(size); // создаём тензор градиентов

    // проходимся по всем значениям тензора градиентов
    for (int i = 0; i < size.height; i++)
        for (int j = 0; j < size.width; j++)
            for (int k = 0; k < size.depth; k++)
                dX(k, i, j) = dout(k, i, j) * (X(k, i, j) > 0 ? 1 : 0); // умножаем градиенты следующего слоя на производную функции активации

    return dX; // возвращаем тензор градиентов
}
