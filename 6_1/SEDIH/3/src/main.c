#include <math.h>
#include <stdio.h>

#define N 5

const char *tokens[N] = {"A", "B", "C", "D", "E"};

int main(void) {
  double z1[N] = {2.5, 2.4, 0.5, 0.4, 0.3};
  double T = 0.7;
  int k = 4;

  double z2[N] = {2.1, 1.6, 1.8, 2.4, 0.9};
  double rho = 1.2;

  // Latency
  double ttft = 0.25;
  double tpot = 0.020;
  int n = 128;
  int context = 4096;

  /* 1. Softmax */
  double p[N];
  double max = z1[0];

  for (int i = 1; i < N; i++)
    if (z1[i] > max)
      max = z1[i];

  double sum = 0;

  for (int i = 0; i < N; i++) {
    p[i] = exp((z1[i] - max) / T);
    sum += p[i];
  }

  for (int i = 0; i < N; i++)
    p[i] /= sum;

  printf("z1 = [2.5, 2.4, 0.5, 0.4, 0.3], T = 0.7, top-k = 4\n");
  printf("z2 = [2.1, 1.6, 1.8, 2.4, 0.9], rho = 1.2\n\n");

  printf("1. Softmax\n");

  for (int i = 0; i < N; i++)
    printf("%s = %.4f\n", tokens[i], p[i]);

  /* 2. Top-k */
  int order[N] = {0, 1, 2, 3, 4};

  for (int i = 0; i < N - 1; i++) {
    for (int j = i + 1; j < N; j++) {
      if (p[order[j]] > p[order[i]]) {
        int temp = order[i];
        order[i] = order[j];
        order[j] = temp;
      }
    }
  }

  double core_sum = 0;

  for (int i = 0; i < k; i++)
    core_sum += p[order[i]];

  printf("\n2. Top-k (k = 4)\n");
  printf("Ядро: ");

  for (int i = 0; i < k; i++)
    printf("%s%s", i ? ", " : "", tokens[order[i]]);

  printf("\nСумма ядра: %.4f\n", core_sum);
  printf("После нормализации:\n");

  for (int i = 0; i < k; i++)
    printf("%s = %.4f\n", tokens[order[i]], p[order[i]] / core_sum);

  /* 3. Greedy */
  int chosen = order[0];

  printf("\n3. Выбор токена\n");
  printf("Метод: жадный\n");
  printf("Выбран: %s (%.4f)\n", tokens[chosen], p[chosen] / core_sum);

  /* 4. Repetition penalty */

  double z2_penalty[N];
  double p2[N];
  double p2_penalty[N];

  for (int i = 0; i < N; i++) {
    z2_penalty[i] = z1[i];
    p2[i] = 0.0;
    p2_penalty[i] = 0.0;
  }

  /* Apply repetition penalty only to the Top-k candidates */
  for (int i = 0; i < k; i++) {
    int idx = order[i];

    if (idx == chosen)
      z2_penalty[idx] /= rho;
  }

  /* Softmax without penalty, only over Top-k */
  double max_z2 = z1[order[0]];

  for (int i = 1; i < k; i++) {
    int idx = order[i];

    if (z1[idx] > max_z2)
      max_z2 = z1[idx];
  }

  double sum_z2 = 0.0;

  for (int i = 0; i < k; i++) {
    int idx = order[i];

    p2[idx] = exp(z1[idx] - max_z2);
    sum_z2 += p2[idx];
  }

  for (int i = 0; i < k; i++) {
    int idx = order[i];
    p2[idx] /= sum_z2;
  }

  /* Softmax after penalty, only over Top-k */
  double max_penalty = z2_penalty[order[0]];

  for (int i = 1; i < k; i++) {
    int idx = order[i];

    if (z2_penalty[idx] > max_penalty)
      max_penalty = z2_penalty[idx];
  }

  double sum_penalty = 0.0;

  for (int i = 0; i < k; i++) {
    int idx = order[i];

    p2_penalty[idx] = exp(z2_penalty[idx] - max_penalty);
    sum_penalty += p2_penalty[idx];
  }

  for (int i = 0; i < k; i++) {
    int idx = order[i];
    p2_penalty[idx] /= sum_penalty;
  }

  /* Select by probability */
  int without_penalty = order[0];
  int with_penalty = order[0];

  for (int i = 1; i < k; i++) {
    int idx = order[i];

    if (p2[idx] > p2[without_penalty])
      without_penalty = idx;

    if (p2_penalty[idx] > p2_penalty[with_penalty])
      with_penalty = idx;
  }

  printf("\n4. Штраф за повтор (rho = %.1f)\n", rho);
  printf("%s: %.2f -> %.2f\n", tokens[chosen], z1[chosen], z2_penalty[chosen]);

  printf("\nВероятности без штрафа:\n");

  for (int i = 0; i < k; i++) {
    int idx = order[i];
    printf("%s = %.4f\n", tokens[idx], p2[idx]);
  }

  printf("\nВероятности со штрафом:\n");

  for (int i = 0; i < k; i++) {
    int idx = order[i];
    printf("%s = %.4f\n", tokens[idx], p2_penalty[idx]);
  }

  printf("\nВыбор со штрафом: %s (%.4f)\n", tokens[with_penalty],
         p2_penalty[with_penalty]);

  printf("Выбор без штрафа: %s (%.4f)\n", tokens[without_penalty],
         p2[without_penalty]);

  /* 5. Metrics */
  double latency = ttft + tpot * (n - 1);
  double throughput = n / latency;

  double kv_bytes = 2.0 * 32 * 8 * 128 * context * 2;

  double kv_mib = kv_bytes / (1024.0 * 1024.0);
  double kv_gib = kv_bytes / (1024.0 * 1024.0 * 1024.0);

  printf("\n5. Метрики\n");
  printf("Latency: %.2f с\n", latency);
  printf("Throughput: %.1f ток/с\n", throughput);
  printf("KV-cache: %.0f МиБ (%.2f ГиБ)\n", kv_mib, kv_gib);

  return 0;
}
