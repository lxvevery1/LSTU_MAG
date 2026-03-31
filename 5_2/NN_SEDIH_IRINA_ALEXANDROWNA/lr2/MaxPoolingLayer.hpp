#include"Tensor.hpp"

class MaxPoolingLayer {
    TensorSize inputSize; // размер входа
    TensorSize outputSize; // размер выхода

    int scale; // во сколько раз уменьшается размерность
    Tensor mask;

public:
    MaxPoolingLayer(TensorSize size, int scale = 2); // создание слоя

    Tensor Forward(const Tensor &X); // прямое распространение
    Tensor Backward(const Tensor &dout, const Tensor &X); // обратное распространение
};
