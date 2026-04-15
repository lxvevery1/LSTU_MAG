#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "Tensor.hpp"

class Optimizer {
public:
    double lr;

    explicit Optimizer(double lr);
    virtual void Update(Tensor& W, Tensor& dW) = 0;
    virtual ~Optimizer() = default;

    // Опционально: геттер/сеттер для изменения lr во время обучения
    double GetLearningRate() const;
    void SetLearningRate(double new_lr);
    virtual void UpdateBias(double& b, double db);
};

class SGD : public Optimizer {
public:
    explicit SGD(double lr);  // просто передаем lr в базовый класс
    void Update(Tensor& W, Tensor& dW) override;
};

class Momentum : public Optimizer {
    double beta;
    Tensor v;

public:
    Momentum(double lr, double beta, TensorSize size);
    void Update(Tensor& W, Tensor& dW) override;
};

class Adam : public Optimizer {
    double b1, b2, eps;
    Tensor m, v;

public:
    Adam(double lr, TensorSize size);
    void Update(Tensor& W, Tensor& dW) override;
};

#endif // OPTIMIZER_HPP
