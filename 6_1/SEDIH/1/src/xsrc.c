#include "xsrc.h"

double src[SRC_WORDS][D_MODEL] = {{0.6, 0.9, -0.5, 0.2},  // children
                                  {1.0, 0.4, 0.5, 0.8},   // play
                                  {0.8, -0.1, -1.0, 0.5}, // games
                                  {0.5, 0.5, 0.5, 0.5}};  // eos

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
