#include "xtgt.h"

double xtgt[TARGET_WORDS][D_MODEL] = {{0.0, 0.0, 0.0, 0.0},   // sos
                                      {0.5, 0.7, -0.3, 0.1},  // Дети
                                      {1.2, 0.5, 0.6, 0.9},   // играют
                                      {0.2, 0.2, 0.2, 0.2},   // в
                                      {0.9, -0.1, -1.0, 0.5}, // игры
                                      {1.0, 1.0, 1.0, 1.0}};  // eos

// 3rd task
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
