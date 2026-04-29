#pragma once

#include "Matrix.hpp"
#include <ostream>
#include <vector>

struct TensorSize {
    int depth;
    int height;
    int width;

    int Magnitude()
    {
        return depth * height * width;
    }
};


class Tensor {
    TensorSize size;
    std::vector<double> values;
    int dw;
    void Init(int depth, int height, int width);

    public:
    Tensor(int depth, int height, int width);
    Tensor(int flatSize) : size{flatSize, 1, 1}, values(flatSize) {};
    Tensor(const TensorSize &size);
    double& operator()(int d, int i, int j);
    double operator()(int d, int i, int j) const;
    TensorSize GetSize() const;
    double& AtFlat(int idx);
    double AtFlat(int idx) const;
    Tensor Unflatten(TensorSize size);

    static Matrix ToMatrix(const Tensor& t);

    friend std::ostream& operator<<(std::ostream& os, const Tensor &tensor);
};
