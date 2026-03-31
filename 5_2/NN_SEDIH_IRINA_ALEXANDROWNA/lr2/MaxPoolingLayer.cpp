#include"Tensor.hpp"
#include"MaxPoolingLayer.hpp"

MaxPoolingLayer::MaxPoolingLayer(TensorSize size, int scale) : mask(size) {
    // запоминаем входной размер
    inputSize.width = size.width;
    inputSize.height = size.height;
    inputSize.depth = size.depth;

    // вычисляем выходной размер
    outputSize.width = size.width / scale;
    outputSize.height = size.height / scale;
    outputSize.depth = size.depth;

    this->scale = scale; // запоминаем коэффициент уменьшения
}

Tensor MaxPoolingLayer::Forward(const Tensor &X) {
    Tensor output(outputSize); // создаём выходной тензор

    // проходимся по каждому из каналов
    for (int d = 0; d < inputSize.depth; d++) {
        for (int i = 0; i < inputSize.height; i += scale) {
            for (int j = 0; j < inputSize.width; j += scale) {
                double max = X(d, i, j); // начальное значение максимума - значение первой клетки подматрицы

                // проходимся по подматрице и ищем максимум и его координаты
                for (int y = i; y < i + scale; y++) {
                    for (int x = j; x < j + scale; x++) {
                        double value = X(d, y, x); // получаем значение входного тензора

                        // если очередное значение больше максимального
                        if (value > max)
                            max = value; // обновляем максимум
                    }
                }

                output(d, i / scale, j / scale) = max; // записываем в выходной тензор найденный максимум
            }
        }
    }

    return output; // возвращаем выходной тензор
}


// обратное распространение
Tensor MaxPoolingLayer::Backward(const Tensor &dout, const Tensor &X) {
    Tensor dX(inputSize); // создаём тензор для градиентов

    for (int d = 0; d < inputSize.depth; d++)
        for (int i = 0; i < inputSize.height; i++)
            for (int j = 0; j < inputSize.width; j++)
                dX(d, i, j) = dout(d, i / scale, j / scale) * mask(d, i, j); // умножаем градиенты на маску

    return dX; // возвращаем посчитанные градиенты
}
