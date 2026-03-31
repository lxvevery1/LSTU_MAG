#include"Tensor.hpp"
#include <ostream>

void Tensor::Init(int width, int height, int depth) {
    size.width = width;
    size.height = height;
    size.depth = depth;

    dw = depth * width;

    values = std::vector<double>(width * height * depth, 0);
}

Tensor::Tensor(int width, int height, int depth) {
    Init(width, height, depth);
}

Tensor::Tensor(const TensorSize& size) {
    Init(size.width, size.height, size.depth);
}

double& Tensor::operator()(int d, int i, int j) {
    return values[i * dw + j * size.depth + d];
}

double Tensor::operator()(int d, int i, int j) const {
    return values[i * dw + j * size.depth + d];
}

TensorSize Tensor::GetSize() const {
    return size;
}

std::ostream& operator<<(std::ostream& os, const Tensor &tensor) {
    for (int d = 0; d < tensor.size.depth; d++) {
        for (int i = 0; i < tensor.size.height; i++) {
            for (int j = 0; j < tensor.size.width; j++) {
                os << tensor.values[i * tensor.dw + j * tensor.size.depth + d] << " ";
            }
            os << std::endl;
        }
    }
    return os;
}
