# Лабораторная работа №3. Классификация, поиск и RAG (вариант 16)

## Данные

Датасет **SetFit/20_newsgroups**, подмножество `rec.*`: `rec.autos`, `rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`. `VARIANT = 16`, `SEED = 42`, размер pool = 3874 документов, Train/Test = 3099/775 (80/20).

## Часть 1. Классификация: без обучения и с дообучением

| Метод | Accuracy | F1-macro | Время, с |
|---|---|---|---|
| A: косинус с описаниями классов | 0.3381 | 0.2858 | 44.7 |
| B: дообучение rubert-tiny2 (5 эпох, lr=2e-05, batch=32) | 0.7535 | 0.7510 | 5261.6 |

Прирост после дообучения: **Accuracy +0.4155**, **F1-macro +0.4652**.

Самая частая пара ошибок: **rec.sport.baseball → rec.sport.hockey** (57 раз). Оба класса — командные виды спорта, и в сообщениях совпадает лексика (team, game, players, score, season, playoff, coach), поэтому модель путает бейсбольные и хоккейные посты. Вторая по частоте пара — `rec.autos` → `rec.motorcycles` (43 ошибки): общая техническая лексика про двигатели, запчасти и поездки. Матрица ошибок: `results/confusion_matrix.png`.

## Часть 2. Поиск: влияние размера чанка

База знаний: 300 документов теста, склеенных в 42 страницы (закрытие страницы после 1024 токенов). Модель `intfloat/multilingual-e5-small`, FAISS IndexFlatIP, 20 запросов (5 коротких, 5 длинных, 5 с синонимами, 5 с опечатками).

| Конфигурация | Размер | Перекрытие | Чанков | Recall@5 | MRR@5 |
|---|---|---|---|---|---|
| S | 128 | 0% | 414 | 0.9000 | 0.8417 |
| M | 256 | 10% | 233 | 0.9000 | 0.7367 |
| L | 512 | 20% | 134 | 0.9000 | 0.8250 |

Лучшая конфигурация — **S** (MRR@5 = 0.8417), худшая — **M** (MRR@5 = 0.7367). Лучший результат — у самой мелкой конфигурации: короткие чанки сохраняют локальный сигнал, а при увеличении окна полезный фрагмент «размывается» соседними сообщениями страницы. При этом Recall@5 у всех конфигураций одинаков (0.9000): различие проявляется только в позиции релевантного чанка, то есть в MRR@5. Диаграмма: `results/part2_retrieval.png`.

### Разбор запроса

```
Запрос: 'Honda Asoct TF500 ebginner motorcycle' (документ 20, класс rec.motorcycles)
Лучше в S, не найден в M.

--- top-5 для конфигурации S ---
1. [OK] page=12 chars=3507-3993 (486 симв.): . Hi folks. I'm going to be buying my first bike and I'm considering an 82 Honda Ascot FT500 with less than 5K miles. Does this sound like a reasonable choice? Is there anything special I need to know? Thanks. Let's see 
2. [--] page=0 chars=4138-4143 (5 симв.): t so.
3. [--] page=18 chars=2994-3567 (573 симв.): limited by the rules of the list to issues related to racing motorcycles and is to be "flame-free". HOW TO GET THE DAILY DISTRIBUTION --------------------------------- You are welcome to subscribe. To subscribe send your
4. [--] page=13 chars=4360-4749 (389 симв.): , and more versatile. I got mine from one of the mail-order houses (Motorcycle Accessory Warehouse, I beleive), for $42. In combination with an ignition disable, and a couple of other deterrents, all I *usually* worry ab
5. [--] page=14 chars=3945-4506 (561 симв.): dents or dings. Needs a new exhaust as original finally rusted through and was discarded. I was in process of making Kenney Roberts TT replica/ cafe racer when graduate school, marriage, child precluded further effort. W

--- top-5 для конфигурации M ---
1. [--] page=18 chars=2676-3780 (1104 симв.): , and the United States. The list was formed (October 1991) in response to a perceived need to both provide technical discussion of riding at the edge of performance (roadracing) and to improve on the very low signal-to-
2. [--] page=16 chars=932-1839 (907 симв.): : high in the low 40s, chance of snow flurries, showers possible. Enjoy the weather where you are. Have a ride on me. What's involved in getting bonded? Anyone know if that's an option recognized by NYS DMV? Rod Good poi
3. [--] page=19 chars=3527-4071 (544 симв.): , it is nicely stable. But it is not overfond of corners. Forced into one, it protests, shaking its head, chattering its front tire, grinding its footpegs, and generally making known its preference for straight pavement.
4. [--] page=20 chars=927-1872 (945 симв.): mot.com <= => Motorola Cellular APD | <= =>"Friends don't let friends wear neon"| Phone: (708) 632-4596 <= Interesting. All I can say is that I haven't seen any wear in the 1000 miles I've used the stuff on the Ninja, an
5. [--] page=18 chars=5589-5760 (171 симв.): ion.rain.com (in body of message write: subscribe euro-moto) thanks, be seeing you, Rich (race list administrator) rich@boogie.EBay.Sun.COM
```

Причина различия: выигрывает более мелкий чанк: в крупном фрагменте короткий полезный документ занимает малую долю текста, и его сигнал «размывается» соседними сообщениями страницы.

## Часть 3. RAG: итеративное улучшение промпта

Архитектура: гибридный поиск (BM25 + лучшая конфигурация части 2, RRF, top-3), генератор `Qwen/Qwen2.5-0.5B-Instruct`, контекст с метками `[Doc N]`. Тестовый набор: 10 вопросов с ответом в базе и 5 без ответа (травмы игроков, не упомянутые в сообщениях). Вопросы задаются по-русски, но для поиска используется их английский перевод: база знаний англоязычная, и BM25 иначе не находит совпадений. Отказом считается ответ с маркером «Нет данных» (а также его английские аналоги «no data / no information / not mentioned»).

| Промпт | Accuracy | Refusal Rate | Faithfulness | Ложные отказы (есть ответ) | Отказы (нет ответа) |
|---|---|---|---|---|---|
| V1 | 0.6000 | 0.2000 | 0.4436 | 0% | 20% |
| V2 | 0.3000 | 0.8000 | 0.5375 | 30% | 80% |
| V3 | 0.3000 | 1.0000 | 0.2167 | 60% | 100% |

По компромиссному критерию (0.4·Accuracy + 0.3·Refusal Rate + 0.3·Faithfulness) лучший результат — **V2**: Accuracy = 0.3000, Refusal Rate = 0.8000, Faithfulness = 0.5375.

- **V1**: Accuracy=0.6000, Refusal Rate=0.2000, Faithfulness=0.4436, ложные отказы=0%, отказы без ответа=20% — редко отказывается на вопросах без ответа и поэтому галлюцинирует.
- **V2**: Accuracy=0.3000, Refusal Rate=0.8000, Faithfulness=0.5375, ложные отказы=30%, отказы без ответа=80% — удерживает приемлемый баланс между отказами и ложными отказами.
- **V3**: Accuracy=0.3000, Refusal Rate=1.0000, Faithfulness=0.2167, ложные отказы=60%, отказы без ответа=100% — слишком осторожен: отказывается более чем в половине случаев, где ответ в базе есть.

Для продакшена выбран **V2**: он даёт лучший компромисс по выбранному критерию и заметно снижает число выдуманных ответов. При этом даже у него остаются ложные отказы, а у V3 надёжность достигается ценой ещё большего числа ложных отказов. Итоговая рекомендация зависит от цены ошибки: если галлюцинация дороже отказа — V2/V3; если важнее покрытие — V1, но только с обязательной проверкой человеком. Диаграмма: `results/part3_prompts.png`.

## AI Audit

Весь код лабораторной (структура скрипта, функции `make_pool`, `split_chunks`, `hybrid`, `faithfulness`, отчёт) сгенерирован с помощью ИИ и проверен запуском: все три части выполнены end-to-end, метрики взяты из реального прогона. Проверялись граничные случаи: пустые тексты (фильтр в `make_pool`), документы короче чанка (`hit` по пересечению границ), отсутствие ответа (Refusal Rate), совпадение границ чанков, несериализуемые типы numpy при записи JSON. Пример ошибки ИИ: первый вариант сохранял метрики через `json.dump` и падал на `numpy.int64` в поле `count`; исправлено приведением к `int`. Вторая ошибка — выбор показательного запроса только среди лучшей/худшей конфигураций давал ложное «не найден», хотя запрос находился на 1-м месте; исправлено проверкой рангов и запасным выбором максимальной разницы. Отдельно проверено, что вопросы без ответа действительно не содержат ответа в базе (поиск по словам `injur`, `knee`, `wrist`, именам игроков).

## Воспроизведение

```bash
.venv/bin/python lr3.py --part 1   # классификация (обучение на CPU ~1.5 ч)
HF_HUB_OFFLINE=1 .venv/bin/python lr3.py --part 2   # поиск и чанки
HF_HUB_OFFLINE=1 .venv/bin/python lr3.py --part 3   # RAG и промпты
.venv/bin/python lr3.py --part report              # сборка этого отчёта
```

Все результаты (JSON, PNG, этот отчёт) лежат в `results/`; кэш данных — в `cache/`; логи прогонов — в `logs/`.

## Приложение. Experiment Log

| Часть | Параметр 1 | Параметр 2 | Метрика | Результат | Гипотеза / комментарий |
|---|---|---|---|---|---|
| Часть 1 | дообучение rubert-tiny2 | 5 эпох, lr=2e-05 | F1-macro | 0.7510 | специализация резко улучшает качество на узкой задаче |
| Часть 1 | косинус без обучения | 4 класса | F1-macro | 0.2858 | нулевой baseline без обучения |
| Часть 2 | чанк 128 | перекрытие 0% | MRR@5 | 0.8417 | лучший вариант |
| Часть 2 | чанк 256 | перекрытие 10% | MRR@5 | 0.7367 | базовый вариант |
| Часть 2 | чанк 512 | перекрытие 20% | MRR@5 | 0.8250 | базовый вариант |
| Часть 3 | промпт V1 | 10+5 вопросов | Refusal Rate | 0.2000 | Accuracy=0.6000, Faithfulness=0.4436 |
| Часть 3 | промпт V2 | 10+5 вопросов | Refusal Rate | 0.8000 | Accuracy=0.3000, Faithfulness=0.5375 |
| Часть 3 | промпт V3 | 10+5 вопросов | Refusal Rate | 1.0000 | Accuracy=0.3000, Faithfulness=0.2167 |
