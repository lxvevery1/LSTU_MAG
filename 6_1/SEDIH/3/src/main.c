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

  double ttft = 0.25;
  double tpot = 0.020;
  int n = 128;
  int context = 4096;

  /* 1. Softmax */
  double p[N];
  double max = z1[0];

  for (int i = 1; i < N; i++)
    if (z1[i] / T > max / T)
      max = z1[i];

  double sum = 0;

  for (int i = 0; i < N; i++) {
    p[i] = exp(z1[i] / T - max / T);
    sum += p[i];
  }

  for (int i = 0; i < N; i++)
    p[i] /= sum;

  printf("Вариант 16\n");
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

  for (int i = 0; i < N - 1; i++) {
    printf("%d", order[i]);
  }

  putchar('\n');
  putchar('\n');
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

  /* 4. Penalty */
  double z2_penalty[N];

  for (int i = 0; i < N; i++) {
    z2_penalty[i] = z2[i];

    if (i == chosen)
      z2_penalty[i] /= rho;
  }

  int with_penalty = 0;
  int without_penalty = 0;

  for (int i = 1; i < N; i++) {
    if (z2_penalty[i] > z2_penalty[with_penalty])
      with_penalty = i;

    if (z2[i] > z2[without_penalty])
      without_penalty = i;
  }

  printf("\n4. Штраф за повтор (rho = %.1f)\n", rho);
  printf("%s: %.2f -> %.2f\n", tokens[chosen], z2[chosen], z2_penalty[chosen]);

  printf("Выбор со штрафом: %s\n", tokens[with_penalty]);
  printf("Выбор без штрафа: %s\n", tokens[without_penalty]);

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
