#include "Tensor.hpp"

class ReLULayer {
    TensorSize size; // размер слоя

public:
    ReLULayer(TensorSize size); // создание слоя

    Tensor Forward(const Tensor &X); // прямое распространение
    Tensor Backward(const Tensor &dout, const Tensor &X); // обратное распространение
};
