/*

   1. Сверточный слой применяет набор обучаемых фильтров к входному изображению
   2. Функция активации - ReLU, Leaky ReLU, ELU, Sigmoid, Tanh
   3. Пулинг (Subsampling) - Max Pooling, Average Pooling
   4. Полносвязные слои - для итоговой классификации
   5. Функция потерь - Cross-Entropy (для классификации)
   6. Оптимизаторы - SGD, Momentum, NAG, RMSprop, Adam, Nadam, AdamW
   7. Регуляризация - Dropout, Batch Normalization, L2 регуляризация

   Вариант 17: Двухканальный вход
   Архитектура:
   - Вход 1 (оригинал, 28x28x1) -> Conv2D (1->8, 3x3, padding: 1) -> ReLU
   - Вход 2 (градиенты Sobel, 28x28x1) -> Conv2D (1->8, 3x3, padding: 1) -> ReLU
   - Concat по каналам (16 каналов) -> Conv2D (16->32, 3x3, padding: 1) -> ReLU
   -> MaxPool2D (2x2) -> Flatten -> Dense (14*14*32 = 6272->128) -> ReLU ->
   Dense (128->10) -> Softmax

   Оптимизаторы для сравнения: SGD, Momentum SGD, Adam
   Параметры эксперимента:
   - Сравнить с одноканальной архитектурой (только оригинал)
   - Исследовать влияние различных предобработок второго канала (Sobel, Canny, Laplacian)
   - Построить графики важности разных каналов (буду использовать pbplots.h)
   - Сравнить точность на тестовой выборке
   - Сделать вывод о том, как многоканальность влияет на сходимость разных оптимизаторов

*/

#include <algorithm>
#include <iostream>
#include "Optimizer.hpp"
#include "Tensor.hpp"
#include "ConvLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "ReLU.hpp"
#include "FullyConnectedLayer.hpp"
#include "CrossEntropy.hpp"
#include "Data.hpp"
#include "./pbplots/pbPlots.hpp"
#include "./pbplots/supportLib.hpp"

enum OPTIMIZER {
    Adam,
    SGD,
    MomentumSGD
};

Tensor Concat(const Tensor& A, const Tensor& B) {
    TensorSize sizeA = A.GetSize();
    TensorSize sizeB = B.GetSize();

    if (sizeA.height != sizeB.height || sizeA.width != sizeB.width)
        throw std::runtime_error("Concat size mismatch");

    TensorSize outSize;
    outSize.depth = sizeA.depth + sizeB.depth;
    outSize.height = sizeA.height;
    outSize.width = sizeA.width;

    Tensor out(outSize);

    // copy A
    for (int d = 0; d < sizeA.depth; d++)
        for (int i = 0; i < sizeA.height; i++)
            for (int j = 0; j < sizeA.width; j++)
                out(d, i, j) = A(d, i, j);

    // copy B
    for (int d = 0; d < sizeB.depth; d++)
        for (int i = 0; i < sizeB.height; i++)
            for (int j = 0; j < sizeB.width; j++)
                out(d + sizeA.depth, i, j) = B(d, i, j);

    return out;
}

Tensor Flatten(const Tensor& X) {
    TensorSize size = X.GetSize();

    int total = size.depth * size.height * size.width;

    Tensor out(1, 1, total);

    for (int d = 0; d < size.depth; d++)
        for (int i = 0; i < size.height; i++)
            for (int j = 0; j < size.width; j++) {
                int idx = d * size.height * size.width + i * size.width + j;
                out.AtFlat(idx) = X(d, i, j);
            }

    return out;
}

Tensor Softmax(const Tensor& X) {
    TensorSize size = X.GetSize();
    Tensor out(size);

    double maxVal = X.AtFlat(0);
    for (int i = 1; i < size.depth; i++)
        if (X.AtFlat(i) > maxVal)
            maxVal = X.AtFlat(i);

    double sum = 0;
    for (int i = 0; i < size.depth; i++) {
        double e = exp(X.AtFlat(i) - maxVal);
        out.AtFlat(i) = e;
        sum += e;
    }

    for (int i = 0; i < size.depth; i++)
        out.AtFlat(i) /= sum;

    return out;
}

Tensor Sobel(const Tensor& X) {
    TensorSize size = X.GetSize();
    Tensor out(size);

    int K[3][3] = {
        {-1,0,1},
        {-2,0,2},
        {-1,0,1}
    };

    for (int i = 1; i < size.height - 1; i++) {
        for (int j = 1; j < size.width - 1; j++) {
            double sum = 0;

            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    sum += X(0, i + ki - 1, j + kj - 1) * K[ki][kj];

            out(0, i, j) = sum;
        }
    }

    return out;
}

// ===== NETWORK =====
ConvLayer *conv1;
ConvLayer *conv2;
ConvLayer *conv3;

ReLULayer *relu1;
ReLULayer *relu2;
ReLULayer *relu3;

MaxPoolingLayer *pool;

FullyConnectedLayer *fc1;
FullyConnectedLayer *fc2;

TensorSize inputSize{1,28,28};

std::vector<double> channelImportance1;
std::vector<double> channelImportance2;
std::vector<double> channelImportance3;

std::vector<Tensor> images;
std::vector<int> labels;

void Init1() {
    images = Data::LoadImages("../data/train-images.idx3-ubyte");
    std::cout << "Images: " << images.size() << std::endl;
    labels = Data::LoadLabels("../data/train-labels.idx1-ubyte");
    std::cout << "Labels: " << labels.size() << std::endl;
    // ===== NETWORK =====
    conv1 = new ConvLayer(inputSize, 8, 3, 1, 1);
    conv2 = new ConvLayer(inputSize, 8, 3, 1, 1);

    relu1 = new ReLULayer({8,28,28});
    relu2 = new ReLULayer({8,28,28});

    conv3 = new ConvLayer({8,28,28}, 32, 3, 1, 1);
    relu3 = new ReLULayer({32,28,28});

    pool = new MaxPoolingLayer({32,28,28}, 2);

    fc1 = new FullyConnectedLayer({32,14,14}, 128, "relu");
    fc2 = new FullyConnectedLayer({1,1,128}, 10, "none");

    channelImportance1.resize(8, 0.0);
    channelImportance2.resize(8, 0.0);
    channelImportance3.resize(32, 0.0);
}


void Init2() {
    // ===== NETWORK =====
    conv1 = new ConvLayer(inputSize, 8, 3, 1, 1);
    conv2 = new ConvLayer(inputSize, 8, 3, 1, 1);

    relu1 = new ReLULayer({8,28,28});
    relu2 = new ReLULayer({8,28,28});

    conv3 = new ConvLayer({16,28,28}, 32, 3, 1, 1);
    relu3 = new ReLULayer({32,28,28});

    pool = new MaxPoolingLayer({32,28,28}, 2);

    fc1 = new FullyConnectedLayer({32,14,14}, 128, "relu");
    fc2 = new FullyConnectedLayer({1,1,128}, 10, "none");

    channelImportance1.resize(8, 0.0);
    channelImportance2.resize(8, 0.0);
    channelImportance3.resize(32, 0.0);
}

double Learn1(const Tensor& X, int label, OPTIMIZER opt)
{
    // ===== FORWARD =====
    Tensor Y1  = conv1->Forward(X);
    Tensor Y1r = relu1->Forward(Y1);

    Tensor Y3  = conv3->Forward(Y1r);
    Tensor Y3r = relu3->Forward(Y3);

    Tensor Ypool = pool->Forward(Y3r);

    Tensor Yflat = Flatten(Ypool);

    Tensor Yfc1 = fc1->Forward(Yflat);
    Tensor Yfc2 = fc2->Forward(Yfc1);

    Tensor Ysoft = Softmax(Yfc2);

    // ===== LOSS =====
    double loss = CrossEntropy::CrossEntropyLoss(Ysoft, label);

    // ===== BACKWARD =====
    Tensor dY = CrossEntropy::CrossEntropyBackward(Ysoft, label);

    Tensor d_fc2 = fc2->Backward(dY, Yfc1);
    Tensor d_fc1 = fc1->Backward(d_fc2, Yflat);

    Tensor d_pool = d_fc1.Unflatten(Ypool.GetSize());

    Tensor d_conv3r = pool->Backward(d_pool, Y3r);
    Tensor d_conv3  = relu3->Backward(d_conv3r, Y3);

    Tensor d_Y1r = conv3->Backward(d_conv3, Y1r);
    Tensor d_Y1  = relu1->Backward(d_Y1r, Y1);

    conv1->Backward(d_Y1, X);

    // ===== UPDATE =====
    double lr = 0.001;
    double beta = 0.9;

    switch (opt)
    {
        case MomentumSGD:
        {
            static Momentum opt_fc1(lr, beta, fc1->GetWeightsSize());
            static Momentum opt_fc2(lr, beta, fc2->GetWeightsSize());

            static Momentum opt_c1(lr, beta, conv1->GetWeightsSize());
            static Momentum opt_c3(lr, beta, conv3->GetWeightsSize());

            fc1->UpdateWeights(opt_fc1);
            fc2->UpdateWeights(opt_fc2);
            conv1->UpdateWeights(opt_c1);
            conv3->UpdateWeights(opt_c3);
            break;
        }

        case Adam:
        {
            static class Adam opt_fc1_adam(lr, fc1->GetWeightsSize());
            static class Adam opt_fc2_adam(lr, fc2->GetWeightsSize());

            static class Adam opt_c1_adam(lr, conv1->GetWeightsSize());
            static class Adam opt_c3_adam(lr, conv3->GetWeightsSize());

            fc1->UpdateWeights(opt_fc1_adam);
            fc2->UpdateWeights(opt_fc2_adam);
            conv1->UpdateWeights(opt_c1_adam);
            conv3->UpdateWeights(opt_c3_adam);
            break;
        }

        case SGD:
        {
            static class SGD opt_fc1_sgd(lr);
            static class SGD opt_fc2_sgd(lr);

            static class SGD opt_c1_sgd(lr);
            static class SGD opt_c3_sgd(lr);

            fc1->UpdateWeights(opt_fc1_sgd);
            fc2->UpdateWeights(opt_fc2_sgd);
            conv1->UpdateWeights(opt_c1_sgd);
            conv3->UpdateWeights(opt_c3_sgd);
            break;
        }

        default:
            std::cerr << "Unknown optimizer\n";
            break;
    }

    return loss;
}

double Learn2(const Tensor& X, int label, OPTIMIZER opt)
{
    // ===== Sobel =====
    Tensor X2 = Sobel(X);

    // ===== FORWARD =====
    Tensor Y1 = conv1->Forward(X);
    Tensor Y1r = relu1->Forward(Y1);

    Tensor Y2 = conv2->Forward(X2);
    Tensor Y2r = relu2->Forward(Y2);

    Tensor Ycat = Concat(Y1r, Y2r);

    Tensor Y3 = conv3->Forward(Ycat);
    Tensor Y3r = relu3->Forward(Y3);

    Tensor Ypool = pool->Forward(Y3r);

    Tensor Yflat = Flatten(Ypool);

    Tensor Yfc1 = fc1->Forward(Yflat);
    Tensor Yfc2 = fc2->Forward(Yfc1);

    Tensor Ysoft = Softmax(Yfc2);

    // ===== LOSS =====
    double loss = CrossEntropy::CrossEntropyLoss(Ysoft, label);

    // ===== BACKWARD =====
    Tensor dY = CrossEntropy::CrossEntropyBackward(Ysoft, label);

    Tensor d_fc2 = fc2->Backward(dY, Yfc1);
    Tensor d_fc1 = fc1->Backward(d_fc2, Yflat);

    Tensor d_pool = d_fc1.Unflatten(Ypool.GetSize());

    Tensor d_conv3r = pool->Backward(d_pool, Y3r);
    Tensor d_conv3 = relu3->Backward(d_conv3r, Y3);

    Tensor d_cat = conv3->Backward(d_conv3, Ycat);

    // split
    Tensor dY1(Y1r.GetSize());
    Tensor dY2(Y2r.GetSize());

    int offset = Y1r.GetSize().depth;

    for (int d = 0; d < Y1r.GetSize().depth; d++)
        for (int i = 0; i < Y1r.GetSize().height; i++)
            for (int j = 0; j < Y1r.GetSize().width; j++)
                dY1(d,i,j) = d_cat(d,i,j);

    for (int d = 0; d < Y2r.GetSize().depth; d++)
        for (int i = 0; i < Y2r.GetSize().height; i++)
            for (int j = 0; j < Y2r.GetSize().width; j++)
                dY2(d,i,j) = d_cat(d + offset,i,j);

    Tensor d1 = relu1->Backward(dY1, Y1);
    conv1->Backward(d1, X);

    Tensor d2 = relu2->Backward(dY2, Y2);
    conv2->Backward(d2, X2);

    // ===== UPDATE =====
    double lr = 0.001;
    double beta = 0.9;

    switch (opt)
    {
        case MomentumSGD:

            static Momentum opt_fc1(lr, beta, fc1->GetWeightsSize());
            static Momentum opt_fc2(lr, beta, fc2->GetWeightsSize());

            static Momentum opt_c1(lr, beta, conv1->GetWeightsSize());
            static Momentum opt_c2(lr, beta, conv2->GetWeightsSize());
            static Momentum opt_c3(lr, beta, conv3->GetWeightsSize());
            fc1->UpdateWeights(opt_fc1);
            fc2->UpdateWeights(opt_fc2);
            conv1->UpdateWeights(opt_c1);
            conv2->UpdateWeights(opt_c2);
            conv3->UpdateWeights(opt_c3);
            break;

        case Adam:
            static class Adam opt_fc1_adam(lr, fc1->GetWeightsSize());
            static class Adam opt_fc2_adam(lr, fc2->GetWeightsSize());

            static class Adam opt_c1_adam(lr, conv1->GetWeightsSize());
            static class Adam opt_c2_adam(lr, conv2->GetWeightsSize());
            static class Adam opt_c3_adam(lr, conv3->GetWeightsSize());
            fc1->UpdateWeights(opt_fc1_adam);
            fc2->UpdateWeights(opt_fc2_adam);
            conv1->UpdateWeights(opt_c1_adam);
            conv2->UpdateWeights(opt_c2_adam);
            conv3->UpdateWeights(opt_c3_adam);
            break;

        case SGD:
            static class SGD opt_fc1_sgd(lr);
            static class SGD opt_fc2_sgd(lr);

            static class SGD opt_c1_sgd(lr);
            static class SGD opt_c2_sgd(lr);
            static class SGD opt_c3_sgd(lr);

            fc1->UpdateWeights(opt_fc1_sgd);
            fc2->UpdateWeights(opt_fc2_sgd);
            conv1->UpdateWeights(opt_c1_sgd);
            conv2->UpdateWeights(opt_c2_sgd);
            conv3->UpdateWeights(opt_c3_sgd);

            break;

        default:
            std::cerr << "I dunno this optimizer\n";
            break;
    }

    return loss;
}

inline const char* ToString(OPTIMIZER opt)
{
    switch (opt)
    {
        case MomentumSGD:   return "MomentumSGD";
        case SGD:   return "SGD";
        case Adam: return "Adam";
        default:      return "[Unknown Optimizer]";
    }
}

std::vector<double> losses1;
std::vector<double> losses2;

const OPTIMIZER opt = MomentumSGD;

void plot(const std::vector<double>& losses1,
          const std::vector<double>& losses2)
{
#pragma region Plot

    RGBABitmapImageReference* imageReference = CreateRGBABitmapImageReference();

    std::vector<double> epochs1, epochs2;
    std::vector<double> scaled1, scaled2;

    auto normalize = [](const std::vector<double>& in, std::vector<double>& out)
    {
        double maxLoss = *std::max_element(in.begin(), in.end());
        double minLoss = *std::min_element(in.begin(), in.end());

        double range = maxLoss - minLoss;
        if (range < 1e-9) range = 1.0;

        for (double l : in) {
            double norm = (l - minLoss) / range;
            out.push_back(norm * 0.1);
        }
    };

    // ===== epochs =====
    for (int i = 0; i < losses1.size(); i++)
        epochs1.push_back(i);

    for (int i = 0; i < losses2.size(); i++)
        epochs2.push_back(i);

    // ===== normalize losses =====
    normalize(losses1, scaled1);
    normalize(losses2, scaled2);

    // ===== series 1 (1-channel) =====
    ScatterPlotSeries* series1 = GetDefaultScatterPlotSeriesSettings();
    series1->xs = &epochs1;
    series1->ys = &scaled1;
    series1->lineType = toVector(L"solid");
    series1->color = CreateRGBColor(0, 0, 1); // blue

    // ===== series 2 (2-channel) =====
    ScatterPlotSeries* series2 = GetDefaultScatterPlotSeriesSettings();
    series2->xs = &epochs2;
    series2->ys = &scaled2;
    series2->lineType = toVector(L"solid");
    series2->color = CreateRGBColor(1, 0, 0); // red

    // ===== settings =====
    ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
    settings->width = 900;
    settings->height = 500;
    settings->title = toVector(L"Single vs Dual channel loss comparison");
    settings->xLabel = toVector(L"Epoch");
    settings->yLabel = toVector(L"Loss");

    settings->scatterPlotSeries->push_back(series1);
    settings->scatterPlotSeries->push_back(series2);

    DrawScatterPlotFromSettings(imageReference, settings, NULL);

    std::vector<double>* pngData = ConvertToPNG(imageReference->image);
    WriteToFile(pngData, "single_vs_dual_" + std::string(ToString(opt)) + ".png");

    std::cout << "\nPlot generated\n";

    DeleteImage(imageReference->image);
}
#pragma endregion

int main() {
    Init1();

    const int datasetSize = 10;
    const int EPOCH = 10;

    for (int epoch = 0; epoch < EPOCH; epoch++) {
        double totalLoss = 0;

        for (int i = 0; i < datasetSize; i++)
        {
            totalLoss += Learn1(images[i], labels[i], opt);
        }

        double avgLoss = totalLoss / datasetSize;
        losses1.push_back(avgLoss);

        std::cout << ToString(opt) << " Epoch (one layer) " << epoch << " loss: " << avgLoss << std::endl;
    }

    Init2();

    for (int epoch = 0; epoch < EPOCH; epoch++) {
        double totalLoss = 0;

        for (int i = 0; i < datasetSize; i++)
        {
            totalLoss += Learn2(images[i], labels[i], opt);
        }

        double avgLoss = totalLoss / datasetSize;
        losses2.push_back(avgLoss);

        std::cout << ToString(opt) << "Epoch (double layer)" << epoch << " loss: " << avgLoss << std::endl;
    }

    plot(losses1, losses2);

    return 0;
}
