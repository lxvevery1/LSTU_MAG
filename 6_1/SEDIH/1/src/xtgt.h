#ifndef XTGT_H
#define XTGT_H

#include "common.h"

extern double xtgt[TARGET_WORDS][D_MODEL];

void target(const double pe[POSITIONS][D_MODEL],
            double xtgt[TARGET_WORDS][D_MODEL]);

#endif
