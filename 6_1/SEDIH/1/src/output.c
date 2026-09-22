#include "output.h"

// Output projection to vocabulary (D_MODEL -> VOCAB)
const double W_out[D_MODEL][VOCAB] = {{0.20, -0.10, 0.30, 0.10, -0.20, 0.15},
                                      {0.10, 0.25, -0.15, 0.30, 0.10, -0.20},
                                      {-0.20, 0.10, 0.20, -0.10, 0.25, 0.30},
                                      {0.15, 0.20, 0.10, 0.25, -0.15, 0.20}};

const double b_out[VOCAB] = {0.10, -0.05, 0.15, 0.00, 0.05, -0.10};

// Project decoder output to vocabulary logits (TARGET_WORDS x VOCAB)
void decoderLogits(const double x_out_dec[TARGET_WORDS][D_MODEL],
                   const double W[D_MODEL][VOCAB], const double b[VOCAB],
                   double logits[TARGET_WORDS][VOCAB]) {
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < VOCAB; j++) {
            logits[i][j] = b[j];

            for (int k = 0; k < D_MODEL; k++)
                logits[i][j] += x_out_dec[i][k] * W[k][j];
        }
    }
}

// Softmax over each row of logits
void softmaxLogits(const double logits[TARGET_WORDS][VOCAB],
                   double probs[TARGET_WORDS][VOCAB]) {
    for (int i = 0; i < TARGET_WORDS; i++) {
        double sum = 0.0;

        for (int j = 0; j < VOCAB; j++) {
            probs[i][j] = exp(logits[i][j]);
            sum += probs[i][j];
        }

        for (int j = 0; j < VOCAB; j++)
            probs[i][j] /= sum;
    }
}

// Cross-entropy loss for one position/target token
double crossEntropyLoss(const double probs[VOCAB], int target) {
    return -log(probs[target]);
}
