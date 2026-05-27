/*
   CIFAR-10 CNN: Standard vs Depthwise Separable Convolutions

   Architecture:
     Input(3, 32, 32)
     -> Conv(3->16, 3x3, pad=1) / [Depthwise(3->3, 3x3) + Pointwise(3->16, 1x1)]
     -> ReLU
     -> Conv(16->32, 3x3, pad=1) / [Depthwise(16->16, 3x3) + Pointwise(16->32, 1x1)]
     -> ReLU
     -> MaxPool(2x2) -> 32x16x16
     -> Flatten -> 8192
     -> FC(8192 -> 128) -> ReLU
     -> FC(128 -> 10) -> Softmax

   Comparison: parameter count, training time, test accuracy.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ConvLayer.hpp"
#include "CrossEntropy.hpp"
#include "Data.hpp"
#include "FullyConnectedLayer.hpp"
#include "MaxPoolingLayer.hpp"
#include "Optimizer.hpp"
#include "ReLU.hpp"
#include "Tensor.hpp"

// ===== Configuration =====
const int MAX_TRAIN_SAMPLES = 500; // subset of CIFAR-10 training set
const int MAX_TEST_SAMPLES = 200;  // subset of test set
const int EPOCHS = 15;
const double LEARNING_RATE = 0.1;
const int NUM_CLASSES = 10;

// ===== Helper functions =====

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
        double e = std::exp(X.AtFlat(i) - maxVal);
        out.AtFlat(i) = e;
        sum += e;
    }
    for (int i = 0; i < size.depth; i++)
        out.AtFlat(i) /= sum;
    return out;
}

int argmax(const Tensor& X) {
    int best = 0;
    double best_val = X.AtFlat(0);
    for (int i = 1; i < X.GetSize().depth; i++) {
        if (X.AtFlat(i) > best_val) {
            best_val = X.AtFlat(i);
            best = i;
        }
    }
    return best;
}

// ===== Network struct =====
// Holds all layers for one architecture variant.

struct Network {
    // Conv layers
    ConvLayer* conv1; // standard: 3->16.  separable: depthwise 3->3
    ConvLayer* conv2; // standard: 16->32. separable: depthwise 16->16. nullptr when separable.
    ConvLayer* conv1_pw; // nullptr when standard. separable: pointwise 3->16
    ConvLayer* conv2_pw; // nullptr when standard. separable: pointwise 16->32

    ReLULayer* relu1;
    ReLULayer* relu2;
    MaxPoolingLayer* pool;
    FullyConnectedLayer* fc1;
    FullyConnectedLayer* fc2;

    bool separable;

    // ----- Forward -----
    Tensor forward(const Tensor& X) const {
        if (separable) {
            // Depthwise + pointwise block 1
            Tensor d1 = conv1->Forward(X);     // depthwise: 3->3
            Tensor p1 = conv1_pw->Forward(d1); // pointwise: 3->16
            Tensor a1 = relu1->Forward(p1);

            // Depthwise + pointwise block 2
            Tensor d2 = conv2->Forward(a1);    // depthwise: 16->16
            Tensor p2 = conv2_pw->Forward(d2); // pointwise: 16->32
            Tensor a2 = relu2->Forward(p2);

            Tensor pooled = pool->Forward(a2);
            Tensor flat = Flatten(pooled);
            Tensor h1 = fc1->Forward(flat);
            Tensor logits = fc2->Forward(h1);
            return Softmax(logits);
        } else {
            Tensor c1 = conv1->Forward(X); // standard: 3->16
            Tensor a1 = relu1->Forward(c1);
            Tensor c2 = conv2->Forward(a1); // standard: 16->32
            Tensor a2 = relu2->Forward(c2);

            Tensor pooled = pool->Forward(a2);
            Tensor flat = Flatten(pooled);
            Tensor h1 = fc1->Forward(flat);
            Tensor logits = fc2->Forward(h1);
            return Softmax(logits);
        }
    }

    // ----- Parameter count -----
    int count_params() const {
        int total = 0;
        if (separable) {
            total += conv1->GetParams();    // depthwise 3->3
            total += conv1_pw->GetParams(); // pointwise 3->16
            total += conv2->GetParams();    // depthwise 16->16
            total += conv2_pw->GetParams(); // pointwise 16->32
        } else {
            total += conv1->GetParams(); // 3->16
            total += conv2->GetParams(); // 16->32
        }
        total += fc1->GetWeightsSize() +
                 static_cast<int>(fc1->GetWeightsSize() / (8192.0 / 128)); // weights + biases
        total += fc2->GetWeightsSize() + 10;                               // weights + biases
        return total;
    }

    // ----- Cleanup -----
    void destroy() {
        delete conv1;
        if (separable) {
            delete conv1_pw;
            delete conv2; // this is the depthwise 16->16
            delete conv2_pw;
        } else {
            delete conv2; // standard 16->32
        }
        delete relu1;
        delete relu2;
        delete pool;
        delete fc1;
        delete fc2;
    }
};

// ===== Network constructors =====

Network create_standard() {
    Network net;
    net.separable = false;

    TensorSize input3x32 = {3, 32, 32};
    TensorSize size16 = {16, 32, 32};

    net.conv1 = new ConvLayer(input3x32, 16, 3, 1, 1); // 3->16
    net.conv2 = new ConvLayer(size16, 32, 3, 1, 1);    // 16->32
    net.conv1_pw = nullptr;
    net.conv2_pw = nullptr;

    net.relu1 = new ReLULayer({16, 32, 32});
    net.relu2 = new ReLULayer({32, 32, 32});
    net.pool = new MaxPoolingLayer({32, 32, 32}, 2); // -> 32x16x16
    net.fc1 = new FullyConnectedLayer({32, 16, 16}, 128, "relu");
    net.fc2 = new FullyConnectedLayer({1, 1, 128}, NUM_CLASSES, "none");

    return net;
}

Network create_separable() {
    Network net;
    net.separable = true;

    TensorSize input3x32 = {3, 32, 32};
    TensorSize size3 = {3, 32, 32};   // after depthwise
    TensorSize size16 = {16, 32, 32}; // after pointwise 1

    // Block 1: depthwise 3->3 + pointwise 3->16
    net.conv1 = new ConvLayer(input3x32, 3, 3, 1, 1, true); // depthwise
    net.conv1_pw = new ConvLayer(size3, 16, 1, 0, 1);       // pointwise 1x1

    // Block 2: depthwise 16->16 + pointwise 16->32
    net.conv2 = new ConvLayer(size16, 16, 3, 1, 1, true); // depthwise
    net.conv2_pw = new ConvLayer(size16, 32, 1, 0, 1);    // pointwise 1x1

    net.relu1 = new ReLULayer({16, 32, 32});
    net.relu2 = new ReLULayer({32, 32, 32});
    net.pool = new MaxPoolingLayer({32, 32, 32}, 2);
    net.fc1 = new FullyConnectedLayer({32, 16, 16}, 128, "relu");
    net.fc2 = new FullyConnectedLayer({1, 1, 128}, NUM_CLASSES, "none");

    return net;
}

// ===== Accuracy evaluation =====

double evaluate(const Network& net, const std::vector<Tensor>& images,
                const std::vector<int>& labels) {
    int correct = 0;
    int total = static_cast<int>(images.size());

    for (int i = 0; i < total; i++) {
        Tensor probs = net.forward(images[i]);
        if (argmax(probs) == labels[i])
            correct++;
    }
    return 100.0 * correct / total;
}

// ===== Main =====

int main() {
    std::cout << "=== CIFAR-10 CNN: Standard vs Depthwise Separable ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    // ----- Load CIFAR-10 data -----
    std::cout << "\nLoading CIFAR-10..." << std::endl;

    std::vector<Tensor> train_images, test_images;
    std::vector<int> train_labels, test_labels;

    const std::string data_dir = "../data/cifar-10-batches-py/";

    // Load training batches
    for (int batch = 1; batch <= 5; batch++) {
        std::string path = data_dir + "data_batch_" + std::to_string(batch);
        auto imgs = Data::LoadCIFAR10Images(path);
        auto lbls = Data::LoadCIFAR10Labels(path);

        int to_take = std::min((int)imgs.size(), MAX_TRAIN_SAMPLES - (int)train_images.size());
        if (to_take <= 0)
            break;

        train_images.insert(train_images.end(), imgs.begin(), imgs.begin() + to_take);
        train_labels.insert(train_labels.end(), lbls.begin(), lbls.begin() + to_take);
    }

    // Load test batch
    {
        auto imgs = Data::LoadCIFAR10Images(data_dir + "test_batch");
        auto lbls = Data::LoadCIFAR10Labels(data_dir + "test_batch");

        int to_take = std::min((int)imgs.size(), MAX_TEST_SAMPLES);
        test_images.insert(test_images.end(), imgs.begin(), imgs.begin() + to_take);
        test_labels.insert(test_labels.end(), lbls.begin(), lbls.begin() + to_take);
    }

    std::cout << "Training samples: " << train_images.size() << std::endl;
    std::cout << "Test samples:     " << test_images.size() << std::endl;
    std::cout << "Epochs:           " << EPOCHS << std::endl;
    std::cout << "Learning rate:    " << LEARNING_RATE << std::endl;

    // ===== Train standard model =====
    std::cout << "\n--- Training Standard CNN ---" << std::endl;

    auto net_std = create_standard();
    int N = static_cast<int>(train_images.size());

    // Per-layer Adam optimizers for standard model
    Adam opt_std_conv1(LEARNING_RATE, net_std.conv1->GetWeightsSize());
    Adam opt_std_conv2(LEARNING_RATE, net_std.conv2->GetWeightsSize());
    Adam opt_std_fc1(LEARNING_RATE, net_std.fc1->GetWeightsSize());
    Adam opt_std_fc2(LEARNING_RATE, net_std.fc2->GetWeightsSize());

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    auto t0_std = std::chrono::high_resolution_clock::now();
    double best_acc_std = 0.0;

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double total_loss = 0.0;

        for (int idx : indices) {
            const Tensor& X = train_images[idx];
            int label = train_labels[idx];

            // Forward
            Tensor c1 = net_std.conv1->Forward(X);
            Tensor a1 = net_std.relu1->Forward(c1);
            Tensor c2 = net_std.conv2->Forward(a1);
            Tensor a2 = net_std.relu2->Forward(c2);
            Tensor pooled = net_std.pool->Forward(a2);
            Tensor flat = Flatten(pooled);
            Tensor h1 = net_std.fc1->Forward(flat);
            Tensor logits = net_std.fc2->Forward(h1);
            Tensor probs = Softmax(logits);

            double loss = CrossEntropy::CrossEntropyLoss(probs, label);
            total_loss += loss;

            // Backward
            Tensor dY = CrossEntropy::CrossEntropyBackward(probs, label);
            Tensor g_fc2 = net_std.fc2->Backward(dY, h1);
            Tensor g_fc1 = net_std.fc1->Backward(g_fc2, flat);
            Tensor g_unflat = g_fc1.Unflatten(pooled.GetSize());
            Tensor g_pool = net_std.pool->Backward(g_unflat, a2);
            Tensor g_relu2 = net_std.relu2->Backward(g_pool, c2);
            Tensor g_conv2 = net_std.conv2->Backward(g_relu2, a1);
            Tensor g_relu1 = net_std.relu1->Backward(g_conv2, c1);
            /* g_conv1 = */ net_std.conv1->Backward(g_relu1, X);

            // Update
            net_std.fc2->UpdateWeights(opt_std_fc2);
            net_std.fc1->UpdateWeights(opt_std_fc1);
            net_std.conv2->UpdateWeights(opt_std_conv2);
            net_std.conv1->UpdateWeights(opt_std_conv1);
        }

        double avg_loss = total_loss / N;
        double test_acc = evaluate(net_std, test_images, test_labels);
        if (test_acc > best_acc_std)
            best_acc_std = test_acc;

        std::cout << "Epoch " << std::setw(2) << epoch << "/" << EPOCHS
                  << "  loss=" << std::setw(8) << avg_loss << "  test_acc=" << std::setw(6)
                  << std::setprecision(2) << test_acc << "%"
                  << "  best=" << best_acc_std << "%" << std::endl;
        std::cout << std::setprecision(4);
    }

    auto t1_std = std::chrono::high_resolution_clock::now();
    double time_std = std::chrono::duration<double>(t1_std - t0_std).count();
    double final_acc_std = evaluate(net_std, test_images, test_labels);
    int params_std = net_std.count_params();

    std::cout << "Train time: " << std::setprecision(1) << time_std << " s"
              << ", Final test accuracy: " << std::setprecision(2) << final_acc_std << "%"
              << std::endl;
    std::cout << std::setprecision(4);

    // ===== Train separable model =====
    std::cout << "\n--- Training Depthwise Separable CNN ---" << std::endl;

    auto net_sep = create_separable();

    // Per-layer Adam optimizers for separable model
    Adam opt_sep_c1(LEARNING_RATE, net_sep.conv1->GetWeightsSize());      // depthwise 3->3
    Adam opt_sep_c1pw(LEARNING_RATE, net_sep.conv1_pw->GetWeightsSize()); // pointwise 3->16
    Adam opt_sep_c2(LEARNING_RATE, net_sep.conv2->GetWeightsSize());      // depthwise 16->16
    Adam opt_sep_c2pw(LEARNING_RATE, net_sep.conv2_pw->GetWeightsSize()); // pointwise 16->32
    Adam opt_sep_fc1(LEARNING_RATE, net_sep.fc1->GetWeightsSize());
    Adam opt_sep_fc2(LEARNING_RATE, net_sep.fc2->GetWeightsSize());

    std::iota(indices.begin(), indices.end(), 0);

    auto t0_sep = std::chrono::high_resolution_clock::now();
    double best_acc_sep = 0.0;

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);
        double total_loss = 0.0;

        for (int idx : indices) {
            const Tensor& X = train_images[idx];
            int label = train_labels[idx];

            // Forward — separable
            Tensor d1 = net_sep.conv1->Forward(X);     // depthwise 3->3
            Tensor p1 = net_sep.conv1_pw->Forward(d1); // pointwise 3->16
            Tensor a1 = net_sep.relu1->Forward(p1);
            Tensor d2 = net_sep.conv2->Forward(a1);    // depthwise 16->16
            Tensor p2 = net_sep.conv2_pw->Forward(d2); // pointwise 16->32
            Tensor a2 = net_sep.relu2->Forward(p2);
            Tensor pooled = net_sep.pool->Forward(a2);
            Tensor flat = Flatten(pooled);
            Tensor h1 = net_sep.fc1->Forward(flat);
            Tensor logits = net_sep.fc2->Forward(h1);
            Tensor probs = Softmax(logits);

            double loss = CrossEntropy::CrossEntropyLoss(probs, label);
            total_loss += loss;

            // Backward — separable
            Tensor dY = CrossEntropy::CrossEntropyBackward(probs, label);
            Tensor g_fc2 = net_sep.fc2->Backward(dY, h1);
            Tensor g_fc1 = net_sep.fc1->Backward(g_fc2, flat);
            Tensor g_unflat = g_fc1.Unflatten(pooled.GetSize());
            Tensor g_pool = net_sep.pool->Backward(g_unflat, a2);
            Tensor g_relu2 = net_sep.relu2->Backward(g_pool, p2);
            Tensor g_pw2 = net_sep.conv2_pw->Backward(g_relu2, d2);
            Tensor g_dw2 = net_sep.conv2->Backward(g_pw2, a1);
            Tensor g_relu1 = net_sep.relu1->Backward(g_dw2, p1);
            Tensor g_pw1 = net_sep.conv1_pw->Backward(g_relu1, d1);
            /* g_dw1 = */ net_sep.conv1->Backward(g_pw1, X);

            // Update — separable
            net_sep.fc2->UpdateWeights(opt_sep_fc2);
            net_sep.fc1->UpdateWeights(opt_sep_fc1);
            net_sep.conv2_pw->UpdateWeights(opt_sep_c2pw);
            net_sep.conv2->UpdateWeights(opt_sep_c2);
            net_sep.conv1_pw->UpdateWeights(opt_sep_c1pw);
            net_sep.conv1->UpdateWeights(opt_sep_c1);
        }

        double avg_loss = total_loss / N;
        double test_acc = evaluate(net_sep, test_images, test_labels);
        if (test_acc > best_acc_sep)
            best_acc_sep = test_acc;

        std::cout << "Epoch " << std::setw(2) << epoch << "/" << EPOCHS
                  << "  loss=" << std::setw(8) << avg_loss << "  test_acc=" << std::setw(6)
                  << std::setprecision(2) << test_acc << "%"
                  << "  best=" << best_acc_sep << "%" << std::endl;
        std::cout << std::setprecision(4);
    }

    auto t1_sep = std::chrono::high_resolution_clock::now();
    double time_sep = std::chrono::duration<double>(t1_sep - t0_sep).count();
    double final_acc_sep = evaluate(net_sep, test_images, test_labels);
    int params_sep = net_sep.count_params();

    std::cout << "Train time: " << std::setprecision(1) << time_sep << " s"
              << ", Final test accuracy: " << std::setprecision(2) << final_acc_sep << "%"
              << std::endl;
    std::cout << std::setprecision(4);

    // ===== Comparison =====
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Model                    Params      Time (s)    Test acc (%)" << std::endl;
    std::cout << "----------------------------------------------------------------"
              << std::endl;
    std::cout << "Standard                 " << std::setw(10) << params_std << "  "
              << std::setw(9) << std::setprecision(1) << time_std << "  " << std::setw(10)
              << std::setprecision(2) << final_acc_std << std::endl;
    std::cout << "Depthwise Separable      " << std::setw(10) << params_sep << "  "
              << std::setw(9) << std::setprecision(1) << time_sep << "  " << std::setw(10)
              << std::setprecision(2) << final_acc_sep << std::endl;
    std::cout << std::endl;

    double param_ratio = (double)params_std / params_sep;
    double speedup = time_std / time_sep;
    double acc_delta = final_acc_std - final_acc_sep;

    std::cout << "Separable vs standard:" << std::endl;
    std::cout << "  Params:          " << std::setprecision(2) << param_ratio << "x fewer"
              << std::endl;
    std::cout << "  Training speed:  " << speedup << "x faster" << std::endl;
    std::cout << "  Accuracy delta:  " << std::showpos << acc_delta << " pp" << std::noshowpos
              << std::endl;

    // Cleanup
    net_std.destroy();
    net_sep.destroy();

    return 0;
}
