#include "FullyConnectedLayer.hpp"
#include "Optimizer.hpp"
#include "Tensor.hpp"

FullyConnectedLayer::FullyConnectedLayer(TensorSize size, int outputs, const
        std::string& activationType) : W(outputs, size.height * size.width *
            size.depth), dW(outputs, size.height * size.width * size.depth),
        df(1, 1, outputs), distribution(0.0, sqrt(2.0 / (size.height *
                        size.width * size.depth))) {
    // запоминаем входной размер
    inputSize.width = size.width;
    inputSize.height = size.height;
    inputSize.depth = size.depth;

    // вычисляем выходной размер
    outputSize.width = 1;
    outputSize.height = 1;
    outputSize.depth = outputs;

    this->inputs = size.height * size.width * size.depth; // запоминаем число входных нейронов
    this->outputs = outputs; // запоминаем число выходных нейронов

    this->activationType = GetActivationType(activationType); // получаем активационную функцию

    b = std::vector<double>(outputs); // создаём вектор смещений
    db = std::vector<double>(outputs); // создаём вектор градиентов по весам смещения

    InitWeights(); // инициализируем весовые коэффициенты
}

// получение типа активационной функции по строке
FullyConnectedLayer::ActivationType FullyConnectedLayer::GetActivationType(const std::string& activationType) const {
    if (activationType == "sigmoid")
        return ActivationType::Sigmoid;

    if (activationType == "tanh")
        return ActivationType::Tanh;

    if (activationType == "relu")
        return ActivationType::ReLU;

    if (activationType == "leakyrelu")
        return ActivationType::LeakyReLU;

    if (activationType == "elu")
        return ActivationType::ELU;

    if (activationType == "none" || activationType == "")
        return ActivationType::None;

    throw std::runtime_error("Invalid activation function");
}


// инициализация весовых коэффициентов
void FullyConnectedLayer::InitWeights() {
    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++)
            W(i, j) = distribution(generator); // генерируем очередное случайное число

        b[i] = 0.01; // все смещения делаем равными 0.01
    }
}


// применение активационной функции с вычислением значений её производной
void FullyConnectedLayer::Activate(Tensor &output) {
    for (int i = 0; i < outputs; i++) {
        double val = output.AtFlat(i);

        if (activationType == ActivationType::None) {
            df.AtFlat(i) = 1;
        }
        else if (activationType == ActivationType::Sigmoid) {
            val = 1 / (1 + exp(-val));
            output.AtFlat(i) = val;
            df.AtFlat(i) = val * (1 - val);
        }
        else if (activationType == ActivationType::ReLU) {
            if (val > 0) {
                df.AtFlat(i) = 1;
            } else {
                output.AtFlat(i) = 0;
                df.AtFlat(i) = 0;
            }
        }
        else if (activationType == ActivationType::ELU) {
            if (val > 0) {
                df.AtFlat(i) = 1;
            } else {
                val = exp(val) - 1;
                output.AtFlat(i) = val;
                df.AtFlat(i) = val + 1;
            }
        }
    }
}

Tensor FullyConnectedLayer::Forward(const Tensor &X) {
    Tensor output(outputSize);

    for (int i = 0; i < outputs; i++) {
        double sum = b[i];

        for (int j = 0; j < inputs; j++) {
            sum += W(i, j) * X.AtFlat(j);
        }

        output.AtFlat(i) = sum;
    }

    Activate(output);

    return output;
}

// обратное распространение
Tensor FullyConnectedLayer::Backward(const Tensor &dout, const Tensor &X) {
    for (int i = 0; i < outputs; i++)
        df.AtFlat(i) *= dout.AtFlat(i);

    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < inputs; j++)
            dW(i, j) = df.AtFlat(i) * X.AtFlat(j);

        db[i] = df.AtFlat(i);
    }

    Tensor dX(inputSize);

    for (int j = 0; j < inputs; j++) {
        double sum = 0;

        for (int i = 0; i < outputs; i++)
            sum += W(i, j) * df.AtFlat(i);

        dX.AtFlat(j) = sum;
    }

    return dX;
}


// обновление весовых коэффициентов
void FullyConnectedLayer::UpdateWeights(Optimizer& opt)
{
    Tensor Wt = Matrix::ToTensor(W);
    Tensor dWt = Matrix::ToTensor(dW);

    opt.Update(Wt, dWt);

    W = Tensor::ToMatrix(Wt);
    dW = Tensor::ToMatrix(dWt);

    for (int i = 0; i < outputs; i++) {
        opt.UpdateBias(b[i], db[i]);
        db[i] = 0;
    }
}

// установка веса матрицы
void FullyConnectedLayer::SetWeight(int i, int j, double weight) {
    W(i, j) = weight;
}

// установка веса смешения
void FullyConnectedLayer::SetBias(int i, double bias) {
    b[i] = bias;
}


int FullyConnectedLayer::GetWeightsSize() const {
    return inputs * outputs;
}
