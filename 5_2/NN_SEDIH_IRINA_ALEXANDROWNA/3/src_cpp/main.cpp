#include <cmath>
#include <cstdint>
#include <iostream>

const uint32_t INPUT_COUNT = 2;
const uint32_t HIDDEN_LAYER_COUNT = 3;
const uint32_t OUTPUT_COUNT = 4;

// z(i) = W(i)X + B(i)
double* hidden_layer(const double W[HIDDEN_LAYER_COUNT][INPUT_COUNT], const double* X,
                     const double* B) {
    double* z = new double[HIDDEN_LAYER_COUNT];

    for (int i = 0; i < HIDDEN_LAYER_COUNT; i++) {
        z[i] = 0.0;

        for (int j = 0; j < INPUT_COUNT; j++) {
            z[i] += W[i][j] * X[j];
        }

        z[i] += B[i];
    }

    return z;
}

// z(i) = W(i)H + B(i)
double* output_layer(const double W[OUTPUT_COUNT][HIDDEN_LAYER_COUNT], const double* H,
                     const double* B) {
    double* z = new double[OUTPUT_COUNT];

    for (int i = 0; i < OUTPUT_COUNT; i++) {
        z[i] = 0.0;

        for (int j = 0; j < HIDDEN_LAYER_COUNT; j++) {
            z[i] += W[i][j] * H[j];
        }

        z[i] += B[i];
    }

    return z;
}

double* activation(double* x, uint32_t n) {
    double* H = new double[n];

    for (int i = 0; i < n; i++) {
        H[i] = x[i] > 0.0 ? 1.0 : 0.0;
    }

    return H;
}

double* activation_sigmoid(double* x, uint32_t n) {
    double* H = new double[n];

    for (int i = 0; i < n; i++)
        H[i] = 1.0 / (1.0 + std::exp(-x[i]));

    return H;
}

double sigmoid(double t, double y) { return (t - y) * y * (1 - y); }

int main() {
    const double X[INPUT_COUNT] = {0.6, 0.8};
    double B[HIDDEN_LAYER_COUNT] = {-0.1, -0.2, -0.3};
    double W[HIDDEN_LAYER_COUNT][INPUT_COUNT] = {{0.5, -0.2}, {-0.3, 0.7}, {0.2, 0.1}};
    double B2[OUTPUT_COUNT] = {0.0, 0.1, -0.1, -0.2};

    /// 1
    double* z1 = hidden_layer(W, X, B);

    std::cout << "printed hidden layers(Z1): " << '\n';
    for (int i = 0; i < HIDDEN_LAYER_COUNT; i++) {
        std::cout << z1[i] << '\n';
    }

    double* H = activation_sigmoid(z1, HIDDEN_LAYER_COUNT);

    std::cout << "printed activations(H): " << '\n';
    for (int i = 0; i < HIDDEN_LAYER_COUNT; i++) {
        std::cout << H[i] << '\n';
    }

    /// 2
    // same, but with calculated H instead of X
    double W2[OUTPUT_COUNT][HIDDEN_LAYER_COUNT] = {
        {0.1, -0.1, 0.2}, {-0.2, -0.3, -0.1}, {0.3, -0.2, 0.4}, {0.0, 0.2, -0.3}};

    double* z2 = output_layer(W2, H, B2);

    std::cout << "printed hidden layers(Z2): " << '\n';
    for (int i = 0; i < OUTPUT_COUNT; i++) {
        std::cout << z2[i] << '\n';
    }

    double* Y = activation_sigmoid(z2, OUTPUT_COUNT);

    std::cout << "printed activations(Y): " << '\n';
    for (int i = 0; i < OUTPUT_COUNT; i++) {
        std::cout << Y[i] << '\n';
    }

    /// 3 - error
    const double T[OUTPUT_COUNT] = {0, 0, 1, 0};
    const double nyu = 0.1;

    double delta1[OUTPUT_COUNT];

    for (int i = 0; i < OUTPUT_COUNT; i++)
        delta1[i] = sigmoid(T[i], Y[i]);

    // output weights update
    {
        for (int i = 0; i < OUTPUT_COUNT; i++) {
            for (int j = 0; j < HIDDEN_LAYER_COUNT; j++)
                W2[i][j] += nyu * delta1[i] * H[j];

            B2[i] += nyu * delta1[i];
        }
    }

    // delta hidden
    {
        double delta2[HIDDEN_LAYER_COUNT];

        for (int j = 0; j < HIDDEN_LAYER_COUNT; j++) {
            delta2[j] = 0.0;

            for (int i = 0; i < OUTPUT_COUNT; i++)
                delta2[j] += delta1[i] * W2[i][j];
        }

        // hidden weights update
        //
        for (int j = 0; j < HIDDEN_LAYER_COUNT; j++) {
            for (int k = 0; k < INPUT_COUNT; k++) {
                W[j][k] += nyu * delta2[j] * X[k];
            }

            B[j] += nyu * delta2[j];
        }
    }

    // print
    //
    {
        std::cout << "\nAfter weight update\n";

        std::cout << "Updated W2:\n";
        for (int i = 0; i < OUTPUT_COUNT; i++) {
            for (int j = 0; j < HIDDEN_LAYER_COUNT; j++)
                std::cout << W2[i][j] << " ";
            std::cout << '\n';
        }

        std::cout << "\nUpdated B2:\n";
        for (int i = 0; i < OUTPUT_COUNT; i++)
            std::cout << B2[i] << '\n';

        std::cout << "\nUpdated W1:\n";
        for (int i = 0; i < HIDDEN_LAYER_COUNT; i++) {
            for (int j = 0; j < INPUT_COUNT; j++)
                std::cout << W[i][j] << " ";
            std::cout << '\n';
        }

        std::cout << "\nUpdated B1:\n";
        for (int i = 0; i < HIDDEN_LAYER_COUNT; i++)
            std::cout << B[i] << '\n';
    }
    return 0;
}
