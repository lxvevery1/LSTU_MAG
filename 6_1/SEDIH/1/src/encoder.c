#include "encoder.h"

// Encoder: Head 1 weights
const double W_Q1[D_MODEL][D_K] = {
    {0.1, 0.3}, {0.2, 0.0}, {0.0, 0.4}, {0.3, -0.1}};

const double W_K1[D_MODEL][D_K] = {
    {0.0, 0.2}, {0.3, -0.1}, {0.1, 0.3}, {0.2, 0.0}};

const double W_V1[D_MODEL][D_K] = {
    {0.2, 0.1}, {0.1, 0.2}, {0.3, 0.0}, {0.0, 0.3}};

// Encoder: Head 2 weights
const double W_Q2[D_MODEL][D_K] = {
    {0.2, 0.1}, {0.0, 0.3}, {0.3, 0.0}, {0.1, 0.2}};

const double W_K2[D_MODEL][D_K] = {
    {0.1, 0.0}, {0.2, 0.3}, {0.0, 0.2}, {0.3, -0.1}};

const double W_V2[D_MODEL][D_K] = {
    {0.0, 0.2}, {0.3, 0.1}, {0.1, 0.3}, {0.2, 0.0}};

// Output projection W_O (4x4): Identity matrix.
double W_O[D_MODEL][D_MODEL] = {{1.0, 0.0, 0.0, 0.0},
                                {0.0, 1.0, 0.0, 0.0},
                                {0.0, 0.0, 1.0, 0.0},
                                {0.0, 0.0, 0.0, 1.0}};

// Decoder: cross-attention weights (same values as encoder)
const double W_Q_cross1[D_MODEL][D_K] = {
    {0.1, 0.3}, {0.2, 0.0}, {0.0, 0.4}, {0.3, -0.1}};

const double W_K_cross1[D_MODEL][D_K] = {
    {0.0, 0.2}, {0.3, -0.1}, {0.1, 0.3}, {0.2, 0.0}};

const double W_V_cross1[D_MODEL][D_K] = {
    {0.2, 0.1}, {0.1, 0.2}, {0.3, 0.0}, {0.0, 0.3}};

const double W_Q_cross2[D_MODEL][D_K] = {
    {0.2, 0.1}, {0.0, 0.3}, {0.3, 0.0}, {0.1, 0.2}};

const double W_K_cross2[D_MODEL][D_K] = {
    {0.1, 0.0}, {0.2, 0.3}, {0.0, 0.2}, {0.3, -0.1}};

const double W_V_cross2[D_MODEL][D_K] = {
    {0.0, 0.2}, {0.3, 0.1}, {0.1, 0.3}, {0.2, 0.0}};

// FF
const double W_FF1[D_MODEL][D_FF] = {{0.20, -0.10, 0.30, 0.10, -0.20, 0.15},
                                     {0.10, 0.25, -0.15, 0.30, 0.10, -0.20},
                                     {-0.20, 0.10, 0.20, -0.10, 0.25, 0.30},
                                     {0.15, 0.20, 0.10, 0.25, -0.15, 0.20}};

const double W_FF2[D_FF][D_MODEL] = {
    {0.20, 0.10, -0.10, 0.15}, {-0.15, 0.20, 0.10, 0.25},
    {0.10, -0.20, 0.25, 0.10}, {0.25, 0.15, 0.20, -0.10},
    {0.10, 0.25, -0.15, 0.20}, {-0.20, 0.10, 0.15, 0.25}};

double b1[D_FF] = {0.10, -0.05, 0.15, 0.00, 0.05, -0.10};
double b2[D_MODEL] = {0.05, 0.10, -0.05, 0.00};

void matrixMultiply(const double a[SRC_WORDS][D_MODEL],
                    const double b[D_MODEL][D_K],
                    double result[SRC_WORDS][D_K]) {
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            result[i][j] = 0.0;

            for (int k = 0; k < D_MODEL; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void encoderHeadQKV(const double xsrc[SRC_WORDS][D_MODEL],
                    const double W_Q[D_MODEL][D_K],
                    const double W_K[D_MODEL][D_K],
                    const double W_V[D_MODEL][D_K], double Q[SRC_WORDS][D_K],
                    double K[SRC_WORDS][D_K], double V[SRC_WORDS][D_K]) {
    matrixMultiply(xsrc, W_Q, Q);
    matrixMultiply(xsrc, W_K, K);
    matrixMultiply(xsrc, W_V, V);
}

void calculateScores(const double Q[SRC_WORDS][D_K],
                     const double K[SRC_WORDS][D_K],
                     double scores[SRC_WORDS][SRC_WORDS]) {
    double scale = sqrt((double)D_K);

    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < SRC_WORDS; j++) {
            scores[i][j] = 0.0;

            for (int k = 0; k < D_K; k++) {
                scores[i][j] += Q[i][k] * K[j][k];
            }

            scores[i][j] /= scale;
        }
    }
}

void softmaxRows(const double scores[SRC_WORDS][SRC_WORDS],
                 double attention[SRC_WORDS][SRC_WORDS]) {
    for (int i = 0; i < SRC_WORDS; i++) {
        double sum = 0.0;

        for (int j = 0; j < SRC_WORDS; j++) {
            attention[i][j] = exp(scores[i][j]);
            sum += attention[i][j];
        }

        for (int j = 0; j < SRC_WORDS; j++) {
            attention[i][j] /= sum;
        }
    }
}

void calculateHead(const double attention[SRC_WORDS][SRC_WORDS],
                   const double V[SRC_WORDS][D_K],
                   double head[SRC_WORDS][D_K]) {
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            head[i][j] = 0.0;

            for (int k = 0; k < SRC_WORDS; k++) {
                head[i][j] += attention[i][k] * V[k][j];
            }
        }
    }
}

void attentionHead(const double xsrc[SRC_WORDS][D_MODEL],
                   const double W_Q[D_MODEL][D_K],
                   const double W_K[D_MODEL][D_K],
                   const double W_V[D_MODEL][D_K],
                   double head[SRC_WORDS][D_K]) {
    double Q[SRC_WORDS][D_K];
    double K[SRC_WORDS][D_K];
    double V[SRC_WORDS][D_K];

    double scores[SRC_WORDS][SRC_WORDS];
    double attention[SRC_WORDS][SRC_WORDS];

    encoderHeadQKV(xsrc, W_Q, W_K, W_V, Q, K, V);

    calculateScores(Q, K, scores);

    softmaxRows(scores, attention);

    calculateHead(attention, V, head);
}

void concatenateHeads(const double head1[SRC_WORDS][D_K],
                      const double head2[SRC_WORDS][D_K],
                      double concat[SRC_WORDS][D_MODEL]) {
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            concat[i][j] = head1[i][j];
            concat[i][j + D_K] = head2[i][j];
        }
    }
}

void attn(const double xsrc[SRC_WORDS][D_MODEL],
          const double Z[SRC_WORDS][D_MODEL],
          double xattn[SRC_WORDS][D_MODEL]) {
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            xattn[i][j] = xsrc[i][j] + Z[i][j];
        }
    }
}

void ffn(const double x_attn[SRC_WORDS][D_MODEL],
         const double W1[D_MODEL][D_FF], const double b1[D_FF],
         const double W2[D_FF][D_MODEL], const double b2[D_MODEL],
         double x_out_enc[SRC_WORDS][D_MODEL]) {
    double hidden[SRC_WORDS][D_FF];

    // 4 -> 6
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_FF; j++) {
            hidden[i][j] = b1[j];

            for (int k = 0; k < D_MODEL; k++)
                hidden[i][j] += x_attn[i][k] * W1[k][j];
        }
    }

    // ReLU
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_FF; j++) {
            if (hidden[i][j] < 0.0)
                hidden[i][j] = 0.0;
        }
    }

    // 6 -> 4
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            x_out_enc[i][j] = b2[j];

            for (int k = 0; k < D_FF; k++)
                x_out_enc[i][j] += hidden[i][k] * W2[k][j];
        }
    }
}
