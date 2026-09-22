#include <math.h>
#include <stdio.h>

#define D_MODEL 4
#define D_H 2
#define D_K 2
#define POSITIONS 6
#define HEADS 2
#define SRC_WORDS 4
#define TARGET_WORDS 6
#define D_FF 6

#define PRINT_MATRIX(name, matrix, rows, cols)                                 \
    do {                                                                       \
        printf("%s (%dx%d):\n", name, rows, cols);                             \
        for (int i = 0; i < rows; i++) {                                       \
            for (int j = 0; j < cols; j++) {                                   \
                printf("%8.4f", matrix[i][j]);                                 \
            }                                                                  \
            printf("\n");                                                      \
        }                                                                      \
        printf("\n");                                                          \
    } while (0)

// 1st TASK
void posEncoding(double pe[POSITIONS][D_MODEL]) {
    for (int pos = 0; pos < POSITIONS; pos++) {
        for (int i = 0; i < D_MODEL / 2; i++) {
            double denominator = pow(10000.0, (2.0 * i) / D_MODEL);

            // чет
            pe[pos][2 * i] = sin(pos / denominator);
            // нечет
            pe[pos][2 * i + 1] = cos(pos / denominator);
        }
    }

    for (int pos = 0; pos < POSITIONS; pos++) {
        printf("PE(%d) = [ ", pos);
        for (int j = 0; j < D_MODEL; j++) {
            printf("%f", pe[pos][j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

// 2nd task
double src[SRC_WORDS][D_MODEL] = {{0.6, 0.9, -0.5, 0.2},  // children
                                  {1.0, 0.4, 0.5, 0.8},   // play
                                  {0.8, -0.1, -1.0, 0.5}, // games
                                  {0.5, 0.5, 0.5, 0.5}};  // eos

void source(const double pe[POSITIONS][D_MODEL],
            double xsrc[SRC_WORDS][D_MODEL]) {
    printf("\nxsrc = [ ");
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            xsrc[i][j] = src[i][j] + pe[i][j];
            printf("%f", xsrc[i][j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

// 3rd task
double xtgt[TARGET_WORDS][D_MODEL] = {{0.0, 0.0, 0.0, 0.0},   // sos
                                      {0.5, 0.7, -0.3, 0.1},  // Дети
                                      {1.2, 0.5, 0.6, 0.9},   // играют
                                      {0.2, 0.2, 0.2, 0.2},   // в
                                      {0.9, -0.1, -1.0, 0.5}, // игры
                                      {1.0, 1.0, 1.0, 1.0}};  // eos

void target(const double pe[POSITIONS][D_MODEL],
            double xtgt[TARGET_WORDS][D_MODEL]) {
    printf("\nxtgt = [ ");
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            xtgt[i][j] = xtgt[i][j] + pe[i][j];
            printf("%f", xtgt[i][j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

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

// decoder uses same weights as encoder
void matrixMultiplyGeneric(const double *a, int rows, int colsA,
                           const double *b, int colsB, double *result) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < colsB; j++) {
            result[i * colsB + j] = 0.0;

            for (int k = 0; k < colsA; k++) {
                result[i * colsB + j] += a[i * colsA + k] * b[k * colsB + j];
            }
        }
    }
}

void decoderProjection(const double x_tgt[TARGET_WORDS][D_MODEL],
                       const double W[D_MODEL][D_K],
                       double result[TARGET_WORDS][D_K]) {
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            result[i][j] = 0.0;

            for (int k = 0; k < D_MODEL; k++)
                result[i][j] += x_tgt[i][k] * W[k][j];
        }
    }
}

void decoderSelfAttentionQKV(const double x_tgt[TARGET_WORDS][D_MODEL],
                             const double W_Q[D_MODEL][D_K],
                             const double W_K[D_MODEL][D_K],
                             const double W_V[D_MODEL][D_K],
                             double Q[TARGET_WORDS][D_K],
                             double K[TARGET_WORDS][D_K],
                             double V[TARGET_WORDS][D_K]) {
    decoderProjection(x_tgt, W_Q, Q);
    decoderProjection(x_tgt, W_K, K);
    decoderProjection(x_tgt, W_V, V);
}

void decoderScores(const double Q[TARGET_WORDS][D_K],
                   const double K[TARGET_WORDS][D_K],
                   double scores[TARGET_WORDS][TARGET_WORDS]) {
    double scale = sqrt((double)D_K);

    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < TARGET_WORDS; j++) {
            if (j > i) {
                scores[i][j] = -INFINITY;
                continue;
            }

            scores[i][j] = 0.0;

            for (int k = 0; k < D_K; k++)
                scores[i][j] += Q[i][k] * K[j][k];

            scores[i][j] /= scale;
        }
    }
}

void decoderSoftmax(const double scores[TARGET_WORDS][TARGET_WORDS],
                    double attention[TARGET_WORDS][TARGET_WORDS]) {
    for (int i = 0; i < TARGET_WORDS; i++) {
        double sum = 0.0;

        for (int j = 0; j < TARGET_WORDS; j++) {
            attention[i][j] = exp(scores[i][j]);
            sum += attention[i][j];
        }

        for (int j = 0; j < TARGET_WORDS; j++)
            attention[i][j] /= sum;
    }
}

void decoderHead(const double attention[TARGET_WORDS][TARGET_WORDS],
                 const double V[TARGET_WORDS][D_K],
                 double head[TARGET_WORDS][D_K]) {
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            head[i][j] = 0.0;

            for (int k = 0; k < TARGET_WORDS; k++) {
                head[i][j] += attention[i][k] * V[k][j];
            }
        }
    }
}

void decoderAttentionHead(const double x_tgt[TARGET_WORDS][D_MODEL],
                          const double W_Q[D_MODEL][D_K],
                          const double W_K[D_MODEL][D_K],
                          const double W_V[D_MODEL][D_K],
                          double head[TARGET_WORDS][D_K]) {
    double Q[TARGET_WORDS][D_K];
    double K[TARGET_WORDS][D_K];
    double V[TARGET_WORDS][D_K];

    double scores[TARGET_WORDS][TARGET_WORDS];
    double attention[TARGET_WORDS][TARGET_WORDS];

    decoderSelfAttentionQKV(x_tgt, W_Q, W_K, W_V, Q, K, V);

    decoderScores(Q, K, scores);

    decoderSoftmax(scores, attention);

    decoderHead(attention, V, head);
}

void concatenateDecoderHeads(const double head1[TARGET_WORDS][D_K],
                             const double head2[TARGET_WORDS][D_K],
                             double concat[TARGET_WORDS][D_MODEL]) {
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            concat[i][j] = head1[i][j];
            concat[i][j + D_K] = head2[i][j];
        }
    }
}

int main() {
    double pe[POSITIONS][D_MODEL];   // 4x6
    double xsrc[SRC_WORDS][D_MODEL]; // 4x4
    // 1
    posEncoding(pe);
    // 2
    source(pe, xsrc);
    // 3
    target(pe, xtgt);
    // 4: encoder
    double head1[SRC_WORDS][D_K];
    double head2[SRC_WORDS][D_K];
    attentionHead(xsrc, W_Q1, W_K1, W_V1, head1);
    attentionHead(xsrc, W_Q2, W_K2, W_V2, head2);

    PRINT_MATRIX("head1", head1, SRC_WORDS, D_K);
    PRINT_MATRIX("head2", head2, SRC_WORDS, D_K);

    double concat[SRC_WORDS][D_MODEL]; // 4x4
    concatenateHeads(head1, head2, concat);
    PRINT_MATRIX("concat", concat, SRC_WORDS, D_MODEL);

    // Z = Concat * W_O = Concat * I4 = Concat
    double x_attn[SRC_WORDS][D_MODEL];
    attn(xsrc, concat, x_attn);
    PRINT_MATRIX("x_attn", x_attn, SRC_WORDS, D_MODEL);

    double hidden[SRC_WORDS][D_FF];
    double x_out_enc[SRC_WORDS][D_MODEL];
    ffn(x_attn, W_FF1, b1, W_FF2, b2, x_out_enc);
    PRINT_MATRIX("ffn", x_out_enc, SRC_WORDS, D_MODEL);

    // 5: decoder
    double head_dec1[TARGET_WORDS][D_K];
    double head_dec2[TARGET_WORDS][D_K];

    decoderAttentionHead(xtgt, W_Q1, W_K1, W_V1, head_dec1);
    decoderAttentionHead(xtgt, W_Q2, W_K2, W_V2, head_dec2);

    PRINT_MATRIX("decoder head 1", head_dec1, TARGET_WORDS, D_K);
    PRINT_MATRIX("decoder head 2", head_dec2, TARGET_WORDS, D_K);

    double concat_dec[TARGET_WORDS][D_MODEL];

    concatenateDecoderHeads(head_dec1, head_dec2, concat_dec);

    PRINT_MATRIX("decoder concat", concat_dec, TARGET_WORDS, D_MODEL);

    double x_dec_self[TARGET_WORDS][D_MODEL];

    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            x_dec_self[i][j] = xtgt[i][j] + concat_dec[i][j];
        }
    }

    double x_cross_attn[TARGET_WORDS][D_MODEL];

    double head_cross1[TARGET_WORDS][D_K];
    double head_cross2[TARGET_WORDS][D_K];

    crossAttentionHead(x_dec_self, x_out_enc, W_Q_cross1, W_K_cross1,
                       W_V_cross1, head_cross1);

    crossAttentionHead(x_dec_self, x_out_enc, W_Q_cross2, W_K_cross2,
                       W_V_cross2, head_cross2);

    double concat_cross[TARGET_WORDS][D_MODEL];

    concatenateDecoderHeads(head_cross1, head_cross2, concat_cross);
    return 0;
}
