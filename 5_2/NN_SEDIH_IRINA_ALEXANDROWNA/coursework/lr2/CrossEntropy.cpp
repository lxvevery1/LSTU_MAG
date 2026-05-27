#include "CrossEntropy.hpp"
#include <cmath>

double CrossEntropy::CrossEntropyLoss(const Tensor& probs, int label) {
    double p = probs.AtFlat(label);

    // защита от log(0)
    if (p < 1e-12)
        p = 1e-12;

    return -log(p);
}

Tensor CrossEntropy::CrossEntropyBackward(const Tensor& probs, int label) {
    Tensor grad(probs.GetSize());

    for (int i = 0; i < probs.GetSize().depth; i++) {
        grad.AtFlat(i) = probs.AtFlat(i);
    }

    // p - y
    grad.AtFlat(label) -= 1.0;

    return grad;
}
