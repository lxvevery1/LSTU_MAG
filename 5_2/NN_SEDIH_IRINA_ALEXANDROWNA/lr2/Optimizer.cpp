#include "Optimizer.hpp"
#include <cmath>

// Базовый класс Optimizer
Optimizer::Optimizer(double lr) : lr(lr) {}

double Optimizer::GetLearningRate() const {
    return lr;
}

void Optimizer::UpdateBias(double& b, double db)
{
    b -= lr * db;
}

void Optimizer::SetLearningRate(double new_lr) {
    lr = new_lr;
}

// SGD implementation
SGD::SGD(double lr) : Optimizer(lr) {}

void SGD::Update(Tensor& W, Tensor& dW) {
    for (int i = 0; i < W.GetSize().Magnitude(); i++) {
        W.AtFlat(i) -= lr * dW.AtFlat(i);  // используем lr из базового класса
        dW.AtFlat(i) = 0;
    }
}

Momentum::Momentum(double lr, double beta, int size)
    : Optimizer(lr), beta(beta), v(size, 1, 1)
{
}

void Momentum::Update(Tensor& W, Tensor& dW)
{
    for (int i = 0; i < W.GetSize().Magnitude(); i++) {
        v.AtFlat(i) = beta * v.AtFlat(i) + lr * dW.AtFlat(i);
        W.AtFlat(i) -= v.AtFlat(i);
        dW.AtFlat(i) = 0;
    }
}

// Adam implementation
Adam::Adam(double lr, int size)
    : Optimizer(lr), b1(0.9), b2(0.999), eps(1e-8), m(size), v(size) {}

void Adam::Update(Tensor& W, Tensor& dW) {
    for (int i = 0; i < W.GetSize().Magnitude(); i++) {
        double g = dW.AtFlat(i);

        m.AtFlat(i) = b1 * m.AtFlat(i) + (1 - b1) * g;
        v.AtFlat(i) = b2 * v.AtFlat(i) + (1 - b2) * g * g;

        W.AtFlat(i) -= lr * m.AtFlat(i) / (std::sqrt(v.AtFlat(i)) + eps);  // используем lr

        dW.AtFlat(i) = 0;
    }
}
