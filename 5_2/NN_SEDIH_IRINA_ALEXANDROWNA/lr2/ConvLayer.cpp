#include"ConvLayer.hpp"

ConvLayer::ConvLayer(TensorSize size, int fc, int fs, int P, int S) : distribution(0.0, sqrt(2.0 / (fs * fs * size.depth))) {
    inputSize.width = size.width;
    inputSize.height = size.height;
    inputSize.depth = size.depth;


    outputSize.width = (size.width - fs + 2 * P) / S + 1;
    outputSize.height = (size.height - fs + 2 * P) / S + 1;
    outputSize.depth = fc;

    this->P = P;
    this->S = S;

    this->fc = fc;
    this->fs = fs;
    this->fd = size.depth;

    W =std::vector<Tensor>(fc, Tensor(fs, fs, fd));
    dW = std::vector<Tensor>(fc, Tensor(fs, fs, fd));

    b = std::vector<double>(fc, 0);
    db = std::vector<double>(fc, 0);

    InitWeights();
}

void ConvLayer::InitWeights() {
    for (int index = 0; index < fc; index++) {
        for (int i = 0; i < fs; i++)
            for (int j = 0; j < fs; j++)
                for (int k = 0; k < fd; k++)
                    W[index](k,i,j) = distribution(generator);

        b[index] = 0.01;
    }
}

Tensor ConvLayer::Forward(const Tensor &X) {
    Tensor output(outputSize);

    for (int f = 0; f < fc; f++) {
        for (int y = 0; y < outputSize.height; y++) {
            for (int x = 0; x < outputSize.width; x++) {
                double sum = b[f];

                for (int i = 0; i < fs; i++)
                {
                    for (int j = 0; j < fs; j++) {
                        int i0 = S * y + i - P;
                        int j0 = S * x + j - P;

                    if (i0 < 0 || i0 >= inputSize.height || j0 < 0 || j0 >= inputSize.width)
                        continue;

                    for (int c = 0; c < fd; c++)
                        sum += X (c, i0, j0) * W[f](c,i,j);
                    }
                }
                output(f, y, x) = sum;
            }
        }
    }
    return output;
}
