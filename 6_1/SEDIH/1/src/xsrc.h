#ifndef XSRC_H
#define XSRC_H

#include "common.h"

void posEncoding(double pe[POSITIONS][D_MODEL]);
void source(const double pe[POSITIONS][D_MODEL],
            double xsrc[SRC_WORDS][D_MODEL]);

#endif
