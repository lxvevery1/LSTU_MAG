#include"ConvLayer.hpp"
#include <iostream>

ConvLayer::ConvLayer(TensorSize size, int fc, int fs, int P, int S) : distribution(0.0, sqrt(2.0 / (fs * fs * size.depth))) {
    inputSize.width = size.width;
    inputSize.height = size.height;
    inputSize.depth = size.depth;


    outputSize.width = (size.width - fs + 2 * P) / S + 1;
    outputSize.height = (size.height - fs + 2 * P) / S + 1;
    outputSize.depth = fc;

    std::cout << "Output size: "
        << outputSize.width << " "
        << outputSize.height << " "
        << outputSize.depth << std::endl;

    std::cout << "INPUT SIZE: "
        << size.width << " "
        << size.height << " "
        << size.depth << std::endl;
    this->P = P;
    this->S = S;

    this->fc = fc;
    this->fs = fs;
    this->fd = size.depth;

    W =std::vector<Tensor>(fc, Tensor(fs, fs, fd));
    dW = std::vector<Tensor>(fc, Tensor(fs, fs, fd));

    b = std::vector<double>(fc, 0);
    db = std::vector<double>(fc, 0);

    InitWeights();
}

void ConvLayer::InitWeights() {
    for (int index = 0; index < fc; index++) {
        for (int i = 0; i < fs; i++)
            for (int j = 0; j < fs; j++)
                for (int k = 0; k < fd; k++)
                    W[index](k,i,j) = distribution(generator);

        b[index] = 0.01;
    }
}

Tensor ConvLayer::Forward(const Tensor &X) {
    Tensor output(outputSize);

    for (int f = 0; f < fc; f++) {
        for (int y = 0; y < outputSize.height; y++) {
            for (int x = 0; x < outputSize.width; x++) {
                double sum = b[f];

                for (int i = 0; i < fs; i++)
                {
                    for (int j = 0; j < fs; j++) {
                        int i0 = S * y + i - P;
                        int j0 = S * x + j - P;

                    if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
                        continue;

                    for (int c = 0; c < fd; c++)
                        sum += X (c, i0, j0) * W[f](c,i,j);
                    }
                }
                output(f, y, x) = sum;
            }
        }
    }
    return output;
}

Tensor ConvLayer::Backward(const Tensor &dout, const Tensor &X) {
    TensorSize size; // размер дельт

    // расчитываем размер для дельт
    size.height = S * (outputSize.height - 1) + 1;
    size.width = S * (outputSize.width - 1) + 1;
    size.depth = outputSize.depth;

    Tensor deltas(size); // создаём тензор для дельт

    // расчитываем значения дельт
    for (int d = 0; d < size.depth; d++)
        for (int i = 0; i < outputSize.height; i++)
            for (int j = 0; j < outputSize.width; j++)
                deltas(d, i * S, j * S) = dout(d, i, j);

    // расчитываем градиенты весов фильтров и смещений
    for (int f = 0; f < fc; f++) {
        for (int y = 0; y < size.height; y++) {
            for (int x = 0; x < size.width; x++) {
                double delta = deltas(f, y, x); // запоминаем значение градиента

                for (int i = 0; i < fs; i++) {
                    for (int j = 0; j < fs; j++) {
                        int i0 = i + y - P;
                        int j0 = j + x - P;

                        // игнорируем выходящие за границы элементы
                        if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
                            continue;

                        // наращиваем градиент фильтра
                        for (int c = 0; c < fd; c++)
                            dW[f](c, i, j) += delta * X(c, i0, j0);
                    }
                }

                db[f] += delta; // наращиваем градиент смещения
            }
        }
    }

    int pad = fs - 1 - P; // заменяем величину дополнения
    Tensor dX(inputSize); // создаём тензор градиентов по входу

    // расчитываем значения градиента
    for (int y = 0; y < inputSize.height; y++) {
        for (int x = 0; x < inputSize.width; x++) {
            for (int c = 0; c < fd; c++) {
                double sum = 0; // сумма для градиента

                // идём по всем весовым коэффициентам фильтров
                for (int i = 0; i < fs; i++) {
                    for (int j = 0; j < fs; j++) {
                        int i0 = y + i - pad;
                        int j0 = x + j - pad;

                        // игнорируем выходящие за границы элементы
                        if (i0 < 0 || i0 >= size.height || j0 < 0 || j0 >= size.width)
                            continue;

                        // суммируем по всем фильтрам
                        for (int f = 0; f < fc; f++)
                            sum += W[f](c, fs - 1 - i, fs - 1 - j) * deltas(f, i0, j0); // добавляем произведение повёрнутых фильтров на дельты
                    }
                }

                dX(c, y, x) = sum; // записываем результат в тензор градиента
            }
        }
    }

    return dX; // возвращаем тензор градиентов
}

// обновление весовых коэффициентов
void ConvLayer::UpdateWeights(double learningRate) {
    for (int index = 0; index < fc; index++) {
        for (int i = 0; i < fs; i++) {
            for (int j = 0; j < fs; j++) {
                for (int d = 0; d < fd; d++) {
                    W[index](d, i, j) -= learningRate * dW[index](d, i, j); // вычитаем градиент, умноженный на скорость обучения
                    dW[index](d, i, j) = 0; // обнуляем градиент фильтра
                }
            }
        }

        b[index] -= learningRate * db[index]; // вычитаем градиент, умноженный на скорость обучения
        db[index] = 0; // обнуляем градиент веса смещения
    }
}

void ConvLayer::SetWeight(int index, int i, int j, int k, double weight) {
    W[index](i, j, k) = weight;
}

void ConvLayer::SetBias(int index, double bias) {
    b[index] = bias;
}
