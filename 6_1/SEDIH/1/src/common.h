#ifndef COMMON_H
#define COMMON_H

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
#define VOCAB 6

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

#endif
