#include"Tensor.hpp"
#include"Matrix.hpp"
#include <cmath>
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

double& Tensor::AtFlat(int idx) {
    return values[idx];
}

double Tensor::AtFlat(int idx) const {
    return values[idx];
}

Tensor Tensor::Unflatten(TensorSize size) {
    Tensor out(size);

    for (int d = 0; d < size.depth; d++)
        for (int i = 0; i < size.height; i++)
            for (int j = 0; j < size.width; j++) {
                int idx = d * size.height * size.width
                        + i * size.width
                        + j;

                out(d,i,j) = this->AtFlat(idx);
            }

    return out;
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

Matrix Tensor::ToMatrix(const Tensor& t)
{
    TensorSize size = t.GetSize();

    Matrix m(size.height, size.width);

    int idx = 0;
    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            m(i, j) = t.AtFlat(idx++);
        }
    }

    return m;
}
