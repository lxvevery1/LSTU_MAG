#include "Matrix.hpp"
#include "Optimizer.hpp"
#include "Tensor.hpp"
#include <random>

class FullyConnectedLayer {
    // тип активационной функции
    enum class ActivationType {
        None, // без активации
        Sigmoid, // сигмоидальная функция
        Tanh, // гиперболический тангенс
        ReLU, // выпрямитель
        LeakyReLU, // выпрямитель с утечкой
        ELU // экспоненциальный выпрямитель
    };

    TensorSize inputSize; // входой размер
    TensorSize outputSize; // выходной размер

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    int inputs; // число входных нейронов
    int outputs; // число выходных нейронов

    ActivationType activationType; // тип активационной функции
    Tensor df; // тензор производных функции активации

    Matrix W; // матрица весовых коэффициентов
    Matrix dW; // матрица градиентов весовых коэффициентов

    std::vector<double> b; // смещения
    std::vector<double> db; // градиенты смещений

    ActivationType GetActivationType(const std::string& activationType) const; // получение типа активационной функции по строке

    void InitWeights(); // инициализация весовых коэффициентов
    void Activate(Tensor &output); // применение активационной функции

public:
    FullyConnectedLayer(TensorSize size, int outputs, const std::string& activationType = "none"); // создание слоя

    Tensor Forward(const Tensor &X); // прямое распространение
    Tensor Backward(const Tensor &dout, const Tensor &X); // обратное распространение

    void UpdateWeights(Optimizer& opt); // обновление весовых коэффициентов
    int GetWeightsSize() const;
    void SetWeight(int i, int j, double weight); // обновление весовых коэффициентов
    void SetBias(int i, double bias); // обновление весовых коэффициентов
};
