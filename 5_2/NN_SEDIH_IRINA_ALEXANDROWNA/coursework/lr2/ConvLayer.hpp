#include <random>
#include "Optimizer.hpp"
#include"Tensor.hpp"

class ConvLayer {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    TensorSize inputSize;
    TensorSize outputSize;

    std::vector<Tensor> W; // filters
    std::vector<double> b; // offset

    std::vector<Tensor> dW; // filter gradient
    std::vector<double> db;

    int P; // zeroes
    int S; // step

    int fc; // filter count
    int fs; // filter size
    int fd; // filter depth

    bool depthwise; // depthwise mode: filter f only sees channel f

    void InitWeights();

    public:
    ConvLayer(TensorSize size, int fc, int fs, int P, int S, bool depthwise = false);

    Tensor Forward(const Tensor &X);
    Tensor Backward(const Tensor &dout, const Tensor &X);

    void UpdateWeights(Optimizer& opt);
    void SetWeight(int index, int i, int j, int k, double weight);
    int GetWeightsSize() const;
    int GetParams() const;  // effective parameter count
    void SetBias(int index, double bias);
};
