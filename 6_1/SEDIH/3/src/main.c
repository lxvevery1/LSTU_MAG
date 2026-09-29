/*
 * main.c — решение практического задания к лекции 3
 *          «Inference и декодирование языковых трансформеров».
 *
 * Вариант 16 (таблицы А и Б части 3 файла task.md). Пять шагов задания:
 *   1) softmax с температурой T через трюк вычитания максимума;
 *   2) фильтр top-k, состав ядра и перенормировка весов;
 *   3) жадный выбор токена (в варианте 16 сэмплирование не требуется);
 *   4) штраф за повторы rho ровно на одном индексе — выбранном в шаге 3,
 *      жадный выбор со штрафом и без него, вывод о развороте решения;
 *   5) latency / throughput по TTFT, TPOT, N и память KV-кэша.
 *
 */

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define N_TOKENS 5 /* размер словаря-кандидата: A, B, C, D, E */
#define LAYERS 32
#define KV_HEADS 8
#define HEAD_DIM 128
#define BATCH 1

/* Словарь-кандидат: пять имён, чтобы в таблицах печатались буквы, а не индексы.
 */
static const char *const TOKENS[N_TOKENS] = {"A", "B", "C", "D", "E"};

/* Тип фильтра шага 2. */
typedef enum { FILTER_TOP_P, FILTER_TOP_K } filter_kind_t;

/* Тип выбора токена в шаге 3. */
typedef enum { PICK_SAMPLE, PICK_GREEDY } pick_kind_t;

/*
 * Первый шаг: следующая позиция, T = 0.7, фильтр top-k = 4, выбор жадный.
 */
static const double LOGITS_1[N_TOKENS] = {2.5, 2.4, 0.5, 0.4, 0.3};

/* Данные одного варианта задания: шаги 1-3 плюс шаги 4-5 из таблицы Б. */
typedef struct {
  double temperature;   /* T */
  filter_kind_t filter; /* top-p или top-k */
  double top_p;         /* порог top-p (не используется в варианте 16) */
  int top_k;            /* размер ядра top-k */
  pick_kind_t pick;     /* сэмплирование или жадный выбор */
  double u;             /* случайное число для рулетки */
  double z2[N_TOKENS];  /* логиты второго шага */
  double rho;           /* штраф за повторы */
  double ttft;          /* время до первого токена, с */
  double tpot;          /* время на токен, с */
  int n_tokens;         /* N */
  int context;          /* S */
  int bytes_per_value;  /* байт на число (2 = fp16, 4 = fp32) */
} variant_t;

/* Вариант 16: строки 16 таблиц А и Б части 3 задания. */
static const variant_t VARIANT_16 = {.temperature = 0.7,
                                     .filter = FILTER_TOP_K,
                                     .top_p =
                                         0.0, /* top-k: порог top-p не нужен */
                                     .top_k = 4,
                                     .pick = PICK_GREEDY,
                                     .u = 0.0, /* жадный выбор: u не нужно */
                                     .z2 = {2.1, 1.6, 1.8, 2.4, 0.9},
                                     .rho = 1.2,
                                     .ttft = 0.25,
                                     .tpot = 0.020,
                                     .n_tokens = 128,
                                     .context = 4096,
                                     .bytes_per_value = 2};

/* Результат решения: всё, что нужно напечатать. */
typedef struct {
  double scaled[N_TOKENS];  /* z / T */
  double max_scaled;        /* максимум z/T — вычитаемое в трюке стабилизации */
  double shifted[N_TOKENS]; /* z/T - max */
  double exps[N_TOKENS];    /* exp(z/T - max) */
  double exp_sum;           /* сумма экспонент */
  double probs[N_TOKENS];   /* softmax: вероятности до фильтра */
  double prob_sum;          /* контроль суммы (должно быть 1) */

  int order[N_TOKENS]; /* индексы токенов по убыванию вероятности */
  int core[N_TOKENS];  /* индексы ядра в порядке убывания */
  int core_size;
  double
      core_sum; /* сумма вероятностей ядра (отброшенная масса = 1 - core_sum) */
  double weights[N_TOKENS]; /* вероятности ядра, перенормированные в сумму 1 */

  int chosen;         /* индекс токена, выбранного в шаге 3 */
  double chosen_prob; /* его вероятность */

  double z2_mod[N_TOKENS]; /* логиты второго шага со штрафом */
  int pick_penalty;        /* жадный выбор со штрафом */
  int pick_plain;          /* жадный выбор без штрафа */

  double latency;        /* TTFT + TPOT * (N - 1) */
  double throughput;     /* N / latency */
  double ceiling;        /* N / TPOT — теоретический потолок */
  double kv_bytes;       /* объём KV-кэша */
  double kv_mib, kv_gib; /* то же в МиБ и ГиБ */
} answer_t;

/* ------------------------------------------------------------------------- */
/* Шаг 1. Softmax с температурой.                                            */
/* ------------------------------------------------------------------------- */

/*
 * softmax с температурой T через трюк вычитания максимума:
 *   p_i = exp(z_i / T) / sum_j exp(z_j / T).
 * Вычитание максимума не меняет результат (все экспоненты умножаются на один
 * и тот же exp(-max)), зато гарантирует, что аргумент экспоненты <= 0 и
 * exp() не переполнится. Именно так считается эталон из части 1 задания.
 */
static void softmax_temperature(const double *z, double temperature,
                                answer_t *a) {
  a->max_scaled = -INFINITY;
  for (int i = 0; i < N_TOKENS; ++i) {
    a->scaled[i] = z[i] / temperature; /* T < 1 делает распределение острее */
    if (a->scaled[i] > a->max_scaled) {
      a->max_scaled = a->scaled[i];
    }
  }

  a->exp_sum = 0.0;
  for (int i = 0; i < N_TOKENS; ++i) {
    a->shifted[i] = a->scaled[i] - a->max_scaled; /* трюк вычитания максимума */
    a->exps[i] = exp(a->shifted[i]);
    a->exp_sum += a->exps[i];
  }

  a->prob_sum = 0.0;
  for (int i = 0; i < N_TOKENS; ++i) {
    a->probs[i] = a->exps[i] / a->exp_sum;
    a->prob_sum += a->probs[i];
  }
}

/*
 * Сортировка индексов по убыванию вероятности (сортировка вставками,
 * устойчивая: при равных вероятностях порядок словаря не меняется —
 * так же ведёт себя torch.sort в test/main.py).
 */
static void sort_desc_by_prob(const double *probs, int *order) {
  for (int i = 0; i < N_TOKENS; ++i) {
    order[i] = i;
  }
  for (int i = 1; i < N_TOKENS; ++i) {
    int key = order[i];
    int j = i - 1;
    while (j >= 0 && probs[order[j]] < probs[key]) {
      order[j + 1] = order[j];
      --j;
    }
    order[j + 1] = key;
  }
}

/* ------------------------------------------------------------------------- */
/* Шаг 2. Фильтр top-p / top-k и перенормировка.                              */
/* ------------------------------------------------------------------------- */

static void build_core(answer_t *a, const variant_t *v) {
  sort_desc_by_prob(a->probs, a->order);
  a->core_size = 0;
  a->core_sum = 0.0;

  if (v->filter == FILTER_TOP_K) {
    for (int i = 0; i < v->top_k && i < N_TOKENS; ++i) {
      a->core[a->core_size++] = a->order[i];
    }
  } else {
    double acc = 0.0;
    for (int i = 0; i < N_TOKENS; ++i) {
      if (!(acc < v->top_p)) {
        break;
      }
      a->core[a->core_size++] = a->order[i];
      acc += a->probs[a->order[i]];
    }
  }

  /* Сумма вероятностей ядра: из неё получается масса, отброшенная фильтром,
     и на неё делятся вероятности при перенормировке. */
  for (int i = 0; i < a->core_size; ++i) {
    a->core_sum += a->probs[a->core[i]];
  }

  /* Перенормировка: сумма весов ядра должна стать ровно 1. */
  for (int i = 0; i < N_TOKENS; ++i) {
    a->weights[i] = 0.0;
  }
  for (int i = 0; i < a->core_size; ++i) {
    a->weights[a->core[i]] = a->probs[a->core[i]] / a->core_sum;
  }
}

/* ------------------------------------------------------------------------- */
/* Шаг 3. Выбор токена.                                                       */
/* ------------------------------------------------------------------------- */

static int pick_token(const answer_t *a, pick_kind_t pick, double u) {
  int chosen = a->core[0];
  double acc = 0.0;
  int decided = 0;

  for (int i = 0; i < a->core_size; ++i) {
    int idx = a->core[i];
    if (pick == PICK_GREEDY) {
      if (a->weights[idx] > a->weights[chosen]) {
        chosen = idx;
      }
    } else {
      double left = acc;
      acc += a->weights[idx];
      if (!decided && u >= left && u < acc) {
        chosen = idx;
        decided = 1;
      }
    }
  }
  /* Страховка от ошибок округления: u, не попавшее ни в один интервал,
     относим к последнему токену ядра. */
  if (pick == PICK_SAMPLE && !decided) {
    chosen = a->core[a->core_size - 1];
  }
  return chosen;
}

/* ------------------------------------------------------------------------- */
/* Шаг 4. Штраф за повторы.                                                  */
/* ------------------------------------------------------------------------- */

static void apply_penalty(const double *z2, int idx, double rho, double *out) {
  for (int i = 0; i < N_TOKENS; ++i) {
    out[i] = (i == idx) ? (z2[i] > 0.0 ? z2[i] / rho : z2[i] * rho) : z2[i];
  }
}

/* Индекс максимума — жадный выбор по логитам. */
static int argmax(const double *v) {
  int best = 0;
  for (int i = 1; i < N_TOKENS; ++i) {
    if (v[i] > v[best]) {
      best = i;
    }
  }
  return best;
}

/* ------------------------------------------------------------------------- */
/* Шаг 5. Метрики и память.                                                   */
/* ------------------------------------------------------------------------- */

/*
 * latency     = TTFT + TPOT * (N - 1): первый токен стоит TTFT (обработка
 *               префикса), каждый из оставшихся N - 1 — TPOT.
 * throughput  = N / latency (ток/с на один запрос).
 * KV-кэш      = 2 * L * H_kv * D_head * S * B * байт: множитель 2 — это
 *               матрицы K и V, дальше слои, головы KV, размер головы, длина
 *               контекста, размер пакета и разрядность (2 байта = fp16,
 *               4 байта = fp32).
 */
static void compute_metrics(const variant_t *v, answer_t *a) {
  a->latency = v->ttft + v->tpot * (v->n_tokens - 1);
  a->throughput = v->n_tokens / a->latency;
  a->ceiling = v->n_tokens / v->tpot;
  a->kv_bytes = 2.0 * LAYERS * KV_HEADS * HEAD_DIM * v->context * BATCH *
                v->bytes_per_value;
  a->kv_mib = a->kv_bytes / (1024.0 * 1024.0);
  a->kv_gib = a->kv_bytes / (1024.0 * 1024.0 * 1024.0);
}

/* ------------------------------------------------------------------------- */
/* Печать. */
/* ------------------------------------------------------------------------- */

static void print_step1(const variant_t *v, const double *z1,
                        const answer_t *a) {
  printf("1. Softmax (T = %.1f)\n", v->temperature);

  for (int i = 0; i < N_TOKENS; ++i) {
    printf("   %s = %.4f\n", TOKENS[i], a->probs[i]);
  }

  printf("\n");
}

static void print_step2(const variant_t *v, const answer_t *a) {
  printf("2. ");

  if (v->filter == FILTER_TOP_K) {
    printf("Top-k (k = %d)\n", v->top_k);
  } else {
    printf("Top-p (p = %.2f)\n", v->top_p);
  }

  printf("   Ядро: ");
  for (int i = 0; i < a->core_size; ++i) {
    printf("%s%s", i ? ", " : "", TOKENS[a->core[i]]);
  }
  printf("\n");

  printf("   Сумма ядра: %.4f\n", a->core_sum);
  printf("   После нормализации:\n");

  for (int i = 0; i < a->core_size; ++i) {
    int idx = a->core[i];
    printf("   %s = %.4f\n", TOKENS[idx], a->weights[idx]);
  }

  printf("\n");
}

static void print_step3(const variant_t *v, const answer_t *a) {
  printf("3. Выбор токена\n");

  if (v->pick == PICK_GREEDY) {
    printf("   Метод: жадный\n");
  } else {
    printf("   Метод: сэмплирование (u = %.2f)\n", v->u);
  }

  printf("   Выбран: %s (%.4f)\n\n", TOKENS[a->chosen], a->weights[a->chosen]);
}

static void print_step4(const variant_t *v, const answer_t *a) {
  int idx = a->chosen;

  printf("4. Штраф за повтор (rho = %.1f)\n", v->rho);
  printf("   %s: %.2f -> %.2f\n", TOKENS[idx], v->z2[idx], a->z2_mod[idx]);

  printf("   Выбор со штрафом: %s\n", TOKENS[a->pick_penalty]);

  printf("   Выбор без штрафа: %s\n", TOKENS[a->pick_plain]);

  if (a->pick_penalty == a->pick_plain) {
    printf("   Решение не изменилось.\n");
  } else {
    printf("   Решение изменилось.\n");
  }

  printf("\n");
}

static void print_step5(const variant_t *v, const answer_t *a) {
  printf("5. Метрики\n");
  printf("   Latency: %.2f с\n", a->latency);
  printf("   Throughput: %.1f ток/с\n", a->throughput);
  printf("   KV-cache: %.0f МиБ (%.2f ГиБ)\n", a->kv_mib, a->kv_gib);

  printf("\n");
}

/* ------------------------------------------------------------------------- */
/* Решение варианта целиком: пять шагов по порядку. */
/* ------------------------------------------------------------------------- */

static void solve_variant(const variant_t *v) {
  answer_t a;
  memset(&a, 0, sizeof a);

  softmax_temperature(LOGITS_1, v->temperature, &a); /* шаг 1 */
  build_core(&a, v);                                 /* шаг 2 */
  a.chosen = pick_token(&a, v->pick, v->u);          /* шаг 3 */
  apply_penalty(v->z2, a.chosen, v->rho, a.z2_mod);  /* шаг 4 */
  a.pick_penalty = argmax(a.z2_mod);
  a.pick_plain = argmax(v->z2);
  a.chosen_prob = a.weights[a.chosen];
  compute_metrics(v, &a); /* шаг 5 */

  printf("Вариант 16\n");
  printf("z1 = [");
  for (int i = 0; i < N_TOKENS; ++i) {
    printf("%s%.1f", i ? ", " : "", LOGITS_1[i]);
  }
  printf("], T = %.1f, top-k = %d\n", v->temperature, v->top_k);

  printf("z2 = [");
  for (int i = 0; i < N_TOKENS; ++i) {
    printf("%s%.1f", i ? ", " : "", v->z2[i]);
  }
  printf("], rho = %.1f, TTFT = %.2f с, TPOT = %.3f с\n\n", v->rho, v->ttft,
         v->tpot);

  print_step1(v, LOGITS_1, &a);
  print_step2(v, &a);
  print_step3(v, &a);
  print_step4(v, &a);
  print_step5(v, &a);
}

/* ------------------------------------------------------------------------- */
/* Сверка с эталоном из части 1 задания.                                      */
/* ------------------------------------------------------------------------- */

/*
 * Печать одной строки сверки с эталоном: отметка, имя, фактическое и
 * ожидаемое значения. Возвращает good, чтобы вызывающий накапливал общий итог.
 */
static int report(const char *name, int good, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  printf("  %s %-28s ", good ? "OK  " : "РАСХОЖДЕНИЕ", name);
  vprintf(fmt, args);
  printf("\n");
  va_end(args);
  return good;
}

/*
 * Эталонный пример: z1 = [2.0, 1.0, 0.5, 0.0, -1.0], T = 0.5, top-p 0.9, u =
 * 0.90; z2 = [2.2, 2.5, 0.9, 1.1, 0.4], rho = 1.2; TTFT = 0.25, TPOT = 0.02, N
 * = 150, S = 2048, 2 байта. Если все функции считают верно, мы должны получить
 * ровно те числа, что напечатаны в условии. Проверка нужна, чтобы убедиться,
 * что те же функции, что обслуживают вариант 16, не сломались на другой ветке
 * (здесь срабатывает top-p и сэмплирование, а в варианте 16 — top-k и жадный
 * выбор).
 */
static int check_reference(void) {
  static const double ref_logits_1[N_TOKENS] = {2.0, 1.0, 0.5, 0.0, -1.0};
  static const variant_t REF = {.temperature = 0.5,
                                .filter = FILTER_TOP_P,
                                .top_p = 0.9,
                                .top_k = 0,
                                .pick = PICK_SAMPLE,
                                .u = 0.90,
                                .z2 = {2.2, 2.5, 0.9, 1.1, 0.4},
                                .rho = 1.2,
                                .ttft = 0.25,
                                .tpot = 0.020,
                                .n_tokens = 150,
                                .context = 2048,
                                .bytes_per_value = 2};
  static const double want_probs[N_TOKENS] = {0.8293, 0.1122, 0.0413, 0.0152,
                                              0.0021};
  static const double want_mod[N_TOKENS] = {2.2000, 2.0833, 0.9000, 1.1000,
                                            0.4000};
  static const double want_weights[2] = {0.8808, 0.1192};

  answer_t a;
  memset(&a, 0, sizeof a);
  softmax_temperature(ref_logits_1, REF.temperature, &a);
  build_core(&a, &REF);
  a.chosen = pick_token(&a, REF.pick, REF.u);
  apply_penalty(REF.z2, a.chosen, REF.rho, a.z2_mod);
  a.pick_penalty = argmax(a.z2_mod);
  a.pick_plain = argmax(REF.z2);
  compute_metrics(&REF, &a);

  int ok = 1;
  printf("Сверка с эталоном из части 1 задания\n");
  printf("--------------------------------------\n");

  ok &= report("сумма экспонент", fabs(a.exp_sum - 1.2059) <= 1e-3,
               "%.4f (эталон 1.2059)", a.exp_sum);
  for (int i = 0; i < N_TOKENS; ++i) {
    char name[64];
    snprintf(name, sizeof name, "вероятность %s", TOKENS[i]);
    ok &= report(name, fabs(a.probs[i] - want_probs[i]) <= 1e-4,
                 "%.4f (эталон %.4f)", a.probs[i], want_probs[i]);
  }
  ok &=
      report("ядро top-p", a.core_size == 2 && a.core[0] == 0 && a.core[1] == 1,
             "{%s, %s}", TOKENS[a.core[0]], TOKENS[a.core[1]]);
  ok &= report("сумма ядра", fabs(a.core_sum - 0.9415) <= 1e-3,
               "%.4f (эталон 0.9415)", a.core_sum);
  for (int i = 0; i < 2; ++i) {
    char name[64];
    snprintf(name, sizeof name, "вес %s", TOKENS[a.core[i]]);
    ok &= report(name, fabs(a.weights[a.core[i]] - want_weights[i]) <= 1e-4,
                 "%.4f (эталон %.4f)", a.weights[a.core[i]], want_weights[i]);
  }
  ok &= report("выбранный токен", a.chosen == 1, "%s (эталон B)",
               TOKENS[a.chosen]);
  for (int i = 0; i < N_TOKENS; ++i) {
    char name[64];
    snprintf(name, sizeof name, "логит %s со штрафом", TOKENS[i]);
    ok &= report(name, fabs(a.z2_mod[i] - want_mod[i]) <= 1e-4,
                 "%.4f (эталон %.4f)", a.z2_mod[i], want_mod[i]);
  }
  ok &= report("жадный выбор со штрафом", a.pick_penalty == 0, "%s (эталон A)",
               TOKENS[a.pick_penalty]);
  ok &= report("жадный выбор без штрафа", a.pick_plain == 1, "%s (эталон B)",
               TOKENS[a.pick_plain]);
  ok &= report("latency", fabs(a.latency - 3.23) <= 1e-9,
               "%.2f с (эталон 3.23 с)", a.latency);
  ok &= report("throughput", fabs(a.throughput - 46.4) <= 0.05,
               "%.1f ток/с (эталон 46.4)", a.throughput);
  ok &= report("память KV-кэша", a.kv_bytes == 268435456.0,
               "%.0f байт = %.0f МиБ (эталон 256 МиБ)", a.kv_bytes, a.kv_mib);

  return ok ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--check") == 0) {
    return check_reference();
  }
  solve_variant(&VARIANT_16);
  return 0;
}
