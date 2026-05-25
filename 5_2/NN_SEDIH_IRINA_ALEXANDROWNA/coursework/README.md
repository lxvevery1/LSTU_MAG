# Depthwise Separable ResNet-18 on CIFAR-10

Проект сравнивает обычный ResNet-18 для CIFAR-10 и вариант, где свертки заменены на depthwise separable convolution в стиле MobileNet: depthwise `k x k` по каждому каналу и pointwise `1 x 1` для смешивания каналов.

## Файлы

- `models.py` - обычный и depthwise-separable ResNet-18.
- `train.py` - загрузка CIFAR-10, обучение и оценка accuracy.
- `benchmark.py` - latency на CPU, число параметров и сериализованный размер модели.
- `main.py` - единый запуск обучения, benchmark и сохранение `results.json`.

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Полный эксперимент

```bash
python main.py \
  --data-dir data \
  --output-dir checkpoints \
  --epochs 50 \
  --batch-size 128 \
  --lr 0.1 \
  --device-train auto \
  --cpu-threads 1 \
  --benchmark-warmup 50 \
  --benchmark-repeats 300
```

После запуска будут сохранены:

- `checkpoints/resnet18_standard.pt`
- `checkpoints/resnet18_separable.pt`
- `checkpoints/results.json`

## Только benchmark готовых checkpoint

```bash
python main.py --skip-train --output-dir checkpoints --cpu-threads 1
```

## Что сравнивать в отчете

Итоговая таблица печатается в консоль и дублируется в `results.json`:

| Model | Params | Size (MB) | Latency (ms) | Test acc (%) |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 (standard) | из запуска | из запуска | из запуска | из запуска |
| ResNet-18 (depthwise-separable) | из запуска | из запуска | из запуска | из запуска |

В выводе также считается:

- `latency_speedup` - во сколько раз separable-модель быстрее обычной;
- `size_ratio` - во сколько раз separable-модель меньше;
- `accuracy_delta_pp` - разница accuracy в процентных пунктах: `standard - separable`.

Для корректного сравнения latency фиксируйте `--cpu-threads`, закрывайте тяжелые фоновые процессы и запускайте benchmark на одной и той же машине.

# brief

  - models.py:12: ResNet-18 для CIFAR-10 и DepthwiseSeparableConv2d.
  - train.py:1: обучение и оценка CIFAR-10.
  - benchmark.py:1: latency на CPU, параметры, размер модели.
  - main.py:1: единый запуск обучения и сравнения.
  - README.md:1: команды запуска и структура отчета.
  - .gitignore:1: исключены venv, data, checkpoints, __pycache__.

  Проверил:

  - python -m py_compile models.py train.py benchmark.py main.py проходит.
  - Smoke benchmark без обучения проходит.
  - Размеры архитектур:
      - standard ResNet-18: 11,173,962 параметров, 42.699 MB
      - depthwise-separable ResNet-18: 1,438,565 параметров, 5.569 MB

  Полное обучение на CIFAR-10 я не запускал: в этом окружении CUDA недоступна из-за старого драйвера, а 50 эпох на CPU займут долго. Команда для
  финального эксперимента:

  source venv/bin/activate
  python main.py --epochs 50 --batch-size 128 --lr 0.1 --device-train auto --cpu-threads 1 --benchmark-warmup 50 --benchmark-repeats 300

  Результаты сохранятся в checkpoints/results.json.
