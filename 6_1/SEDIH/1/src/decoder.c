#include "decoder.h"

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

// Decoder FFN: same weights and biases as encoder, but TARGET_WORDS rows.
void decoderFFN(const double x_attn[TARGET_WORDS][D_MODEL],
                const double W1[D_MODEL][D_FF], const double b1[D_FF],
                const double W2[D_FF][D_MODEL], const double b2[D_MODEL],
                double x_out_dec[TARGET_WORDS][D_MODEL]) {
    double hidden[TARGET_WORDS][D_FF];

    // 4 -> 6
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_FF; j++) {
            hidden[i][j] = b1[j];

            for (int k = 0; k < D_MODEL; k++)
                hidden[i][j] += x_attn[i][k] * W1[k][j];
        }
    }

    // ReLU
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_FF; j++) {
            if (hidden[i][j] < 0.0)
                hidden[i][j] = 0.0;
        }
    }

    // 6 -> 4
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_MODEL; j++) {
            x_out_dec[i][j] = b2[j];

            for (int k = 0; k < D_FF; k++)
                x_out_dec[i][j] += hidden[i][k] * W2[k][j];
        }
    }
}

// Cross-attention: Q from decoder, K and V from encoder output.
void crossAttentionHead(const double x_dec[TARGET_WORDS][D_MODEL],
                        const double x_enc[SRC_WORDS][D_MODEL],
                        const double W_Q[D_MODEL][D_K],
                        const double W_K[D_MODEL][D_K],
                        const double W_V[D_MODEL][D_K],
                        double head[TARGET_WORDS][D_K]) {
    double Q[TARGET_WORDS][D_K];
    double K[SRC_WORDS][D_K];
    double V[SRC_WORDS][D_K];

    double scores[TARGET_WORDS][SRC_WORDS];
    double attention[TARGET_WORDS][SRC_WORDS];

    // Q из декодера
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            Q[i][j] = 0.0;

            for (int k = 0; k < D_MODEL; k++)
                Q[i][j] += x_dec[i][k] * W_Q[k][j];
        }
    }

    // K и V из выхода энкодера
    for (int i = 0; i < SRC_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            K[i][j] = 0.0;
            V[i][j] = 0.0;

            for (int k = 0; k < D_MODEL; k++) {
                K[i][j] += x_enc[i][k] * W_K[k][j];
                V[i][j] += x_enc[i][k] * W_V[k][j];
            }
        }
    }

    double scale = sqrt((double)D_K);

    // scores (без маски - виден весь вход энкодера)
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < SRC_WORDS; j++) {
            scores[i][j] = 0.0;

            for (int k = 0; k < D_K; k++)
                scores[i][j] += Q[i][k] * K[j][k];

            scores[i][j] /= scale;
        }
    }

    // softmax
    for (int i = 0; i < TARGET_WORDS; i++) {
        double sum = 0.0;

        for (int j = 0; j < SRC_WORDS; j++) {
            attention[i][j] = exp(scores[i][j]);
            sum += attention[i][j];
        }

        for (int j = 0; j < SRC_WORDS; j++)
            attention[i][j] /= sum;
    }

    // head = attention * V
    for (int i = 0; i < TARGET_WORDS; i++) {
        for (int j = 0; j < D_K; j++) {
            head[i][j] = 0.0;

            for (int k = 0; k < SRC_WORDS; k++)
                head[i][j] += attention[i][k] * V[k][j];
        }
    }
}
