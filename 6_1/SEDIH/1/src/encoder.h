#ifndef ENCODER_H
#define ENCODER_H

#include "common.h"

extern const double W_Q1[D_MODEL][D_K];
extern const double W_K1[D_MODEL][D_K];
extern const double W_V1[D_MODEL][D_K];

extern const double W_Q2[D_MODEL][D_K];
extern const double W_K2[D_MODEL][D_K];
extern const double W_V2[D_MODEL][D_K];

extern double W_O[D_MODEL][D_MODEL];

extern const double W_FF1[D_MODEL][D_FF];
extern const double W_FF2[D_FF][D_MODEL];
extern double b1[D_FF];
extern double b2[D_MODEL];

extern const double W_Q_cross1[D_MODEL][D_K];
extern const double W_K_cross1[D_MODEL][D_K];
extern const double W_V_cross1[D_MODEL][D_K];

extern const double W_Q_cross2[D_MODEL][D_K];
extern const double W_K_cross2[D_MODEL][D_K];
extern const double W_V_cross2[D_MODEL][D_K];

void matrixMultiply(const double a[SRC_WORDS][D_MODEL],
                    const double b[D_MODEL][D_K],
                    double result[SRC_WORDS][D_K]);

void encoderHeadQKV(const double xsrc[SRC_WORDS][D_MODEL],
                    const double W_Q[D_MODEL][D_K],
                    const double W_K[D_MODEL][D_K],
                    const double W_V[D_MODEL][D_K], double Q[SRC_WORDS][D_K],
                    double K[SRC_WORDS][D_K], double V[SRC_WORDS][D_K]);

void calculateScores(const double Q[SRC_WORDS][D_K],
                     const double K[SRC_WORDS][D_K],
                     double scores[SRC_WORDS][SRC_WORDS]);

void softmaxRows(const double scores[SRC_WORDS][SRC_WORDS],
                 double attention[SRC_WORDS][SRC_WORDS]);

void calculateHead(const double attention[SRC_WORDS][SRC_WORDS],
                   const double V[SRC_WORDS][D_K],
                   double head[SRC_WORDS][D_K]);

void attentionHead(const double xsrc[SRC_WORDS][D_MODEL],
                   const double W_Q[D_MODEL][D_K],
                   const double W_K[D_MODEL][D_K],
                   const double W_V[D_MODEL][D_K],
                   double head[SRC_WORDS][D_K]);

void concatenateHeads(const double head1[SRC_WORDS][D_K],
                      const double head2[SRC_WORDS][D_K],
                      double concat[SRC_WORDS][D_MODEL]);

void attn(const double xsrc[SRC_WORDS][D_MODEL],
          const double Z[SRC_WORDS][D_MODEL],
          double xattn[SRC_WORDS][D_MODEL]);

void ffn(const double x_attn[SRC_WORDS][D_MODEL],
         const double W1[D_MODEL][D_FF], const double b1[D_FF],
         const double W2[D_FF][D_MODEL], const double b2[D_MODEL],
         double x_out_enc[SRC_WORDS][D_MODEL]);

#endif
