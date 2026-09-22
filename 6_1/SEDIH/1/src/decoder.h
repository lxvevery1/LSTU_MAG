#ifndef DECODER_H
#define DECODER_H

#include "common.h"
#include "encoder.h"

void decoderProjection(const double x_tgt[TARGET_WORDS][D_MODEL],
                       const double W[D_MODEL][D_K],
                       double result[TARGET_WORDS][D_K]);

void decoderSelfAttentionQKV(const double x_tgt[TARGET_WORDS][D_MODEL],
                             const double W_Q[D_MODEL][D_K],
                             const double W_K[D_MODEL][D_K],
                             const double W_V[D_MODEL][D_K],
                             double Q[TARGET_WORDS][D_K],
                             double K[TARGET_WORDS][D_K],
                             double V[TARGET_WORDS][D_K]);

void decoderScores(const double Q[TARGET_WORDS][D_K],
                   const double K[TARGET_WORDS][D_K],
                   double scores[TARGET_WORDS][TARGET_WORDS]);

void decoderSoftmax(const double scores[TARGET_WORDS][TARGET_WORDS],
                    double attention[TARGET_WORDS][TARGET_WORDS]);

void decoderHead(const double attention[TARGET_WORDS][TARGET_WORDS],
                 const double V[TARGET_WORDS][D_K],
                 double head[TARGET_WORDS][D_K]);

void decoderAttentionHead(const double x_tgt[TARGET_WORDS][D_MODEL],
                          const double W_Q[D_MODEL][D_K],
                          const double W_K[D_MODEL][D_K],
                          const double W_V[D_MODEL][D_K],
                          double head[TARGET_WORDS][D_K]);

void concatenateDecoderHeads(const double head1[TARGET_WORDS][D_K],
                             const double head2[TARGET_WORDS][D_K],
                             double concat[TARGET_WORDS][D_MODEL]);

void decoderFFN(const double x_attn[TARGET_WORDS][D_MODEL],
                const double W1[D_MODEL][D_FF], const double b1[D_FF],
                const double W2[D_FF][D_MODEL], const double b2[D_MODEL],
                double x_out_dec[TARGET_WORDS][D_MODEL]);

void crossAttentionHead(const double x_dec[TARGET_WORDS][D_MODEL],
                        const double x_enc[SRC_WORDS][D_MODEL],
                        const double W_Q[D_MODEL][D_K],
                        const double W_K[D_MODEL][D_K],
                        const double W_V[D_MODEL][D_K],
                        double head[TARGET_WORDS][D_K]);

#endif
