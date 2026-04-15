#include "Tensor.hpp"

class CrossEntropy {
public:
    static double CrossEntropyLoss(const Tensor& probs, int label);
    static Tensor CrossEntropyBackward(const Tensor& probs, int label);
};
