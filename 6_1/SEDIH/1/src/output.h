#ifndef OUTPUT_H
#define OUTPUT_H

#include "common.h"

extern const double W_out[D_MODEL][VOCAB];
extern const double b_out[VOCAB];

void decoderLogits(const double x_out_dec[TARGET_WORDS][D_MODEL],
                   const double W[D_MODEL][VOCAB], const double b[VOCAB],
                   double logits[TARGET_WORDS][VOCAB]);

void softmaxLogits(const double logits[TARGET_WORDS][VOCAB],
                   double probs[TARGET_WORDS][VOCAB]);

double crossEntropyLoss(const double probs[VOCAB], int target);

#endif
