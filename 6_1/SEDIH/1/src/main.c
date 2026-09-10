#include<stdio.h>
#include<math.h>

#define D_MODEL 4
#define POSITIONS 6
#define D_K 2
#define HEADS 2
#define SRC_WORDS 3
#define TARGET_WORDS 6


// 1st TASK
void posEncoding(double pe[POSITIONS][D_MODEL])
{
    for (int pos = 0; pos < POSITIONS; pos++)
    {
        for (int i = 0; i < D_MODEL / 2; i++)
        {
            double denominator = pow(10000.0, (2.0 * i) / D_MODEL);

            // чет
            pe[pos][2 * i] = sin(pos / denominator);
            // нечет
            pe[pos][2 * i + 1] = cos(pos / denominator);
        }
    }

    for (int pos = 0; pos < POSITIONS; pos++)
    {
        printf("PE(%d) = [ ", pos);
        for (int j = 0; j < D_MODEL; j++)
        {
            printf("%f", pe[pos][j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

// 2nd task
double src[SRC_WORDS][D_MODEL] = {{0.6, 0.9, -0.5, 0.2},  // children
    {1.0, 0.4, 0.5, 0.8},                   // play
    {0.8, -0.1, -1.0, 0.5}};                // games

void source(const double pe[POSITIONS][D_MODEL], double xsrc[SRC_WORDS][D_MODEL])
{
    printf("\nxsrc = [ ");
    for (int i = 0; i < SRC_WORDS; i++)
    {
        for (int j = 0; j < D_MODEL; j++)
        {
            xsrc[i][j] = src[i][j] + pe[i][j];
            printf("%f", xsrc[i][j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

// 3rd task
double tgt[TARGET_WORDS][D_MODEL] =
        {{0.0, 0.0, 0.0, 0.0},  // sos
        {0.5, 0.7, -0.3, 0.1},  // Дети
        {1.2, 0.5, 0.6, 0.9},   // играют
        {0.2, 0.2, 0.2, 0.2},   // в
        {0.9, -0.1, -1.0, 0.5}, // игры
        {1.0, 1.0, 1.0, 1.0}};  // eos

void target(const double pe[POSITIONS][D_MODEL], double xtgt[TARGET_WORDS][D_MODEL])
{
    printf("\nxtgt = [ ");
    for (int i = 0; i < TARGET_WORDS; i++)
    {
        for (int j = 0; j < D_MODEL; j++)
        {
            xtgt[i][j] = tgt[i][j] + pe[i][j];
            printf("%f", xtgt[i][j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

// 4th task
// Encoder: Head 1 weights (4x2).
double W_Q1[D_MODEL][D_K] =
{
    { 0.1,  0.2},
    { 0.3, -0.1},
    { 0.0,  0.4},
    {-0.2,  0.1}
};

double W_K1[D_MODEL][D_K] =
{
    { 0.2,  0.1},
    {-0.1,  0.3},
    { 0.4,  0.0},
    { 0.1, -0.2}
};

double W_V1[D_MODEL][D_K] =
{
    {0.1, -0.1},
    {0.2,  0.3},
    {0.0,  0.1},
    {0.3,  0.0}
};


// Encoder: Head 2 weights (4x2).

double W_Q2[D_MODEL][D_K] =
{
    { 0.0, 0.3},
    { 0.2, 0.1},
    {-0.1, 0.2},
    { 0.3, 0.0}
};

double W_K2[D_MODEL][D_K] =
{
    {0.1,  0.2},
    {0.3, -0.2},
    {0.0,  0.1},
    {0.2,  0.3}
};

double W_V2[D_MODEL][D_K] =
{
    { 0.2, 0.0},
    { 0.1, 0.2},
    {-0.1, 0.3},
    { 0.0, 0.1}
};


// Output projection W_O (4x4): Identity matrix.


double W_O[D_MODEL][D_MODEL] =
{
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {0.0, 0.0, 1.0, 0.0},
    {0.0, 0.0, 0.0, 1.0}
};


void encoder()
{

}

int main()
{
    double pe[POSITIONS][D_MODEL];
    // по заданию должна получиться матрица 3x4
    // поэтому EOS не в счет
    double xsrc[SRC_WORDS][D_MODEL];

    posEncoding(pe);
    source(pe, xsrc);
    target(pe, tgt);

    return 0;
}
