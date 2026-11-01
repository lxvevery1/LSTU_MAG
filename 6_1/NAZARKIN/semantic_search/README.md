# Семантический поиск по постам r/WallStreetBets

Лабораторная работа №2: интеграция документ-ориентированного хранилища и векторного
поиска. Система принимает запрос на естественном языке (в том числе по-русски) и
возвращает первые N постов WSB, смысл которых связан с запросом.

Пример из задания:

```bash
python search.py "акции APPL" -n 10
```

`APPL` — частая опечатка тикера `AAPL`, поэтому запрос нормализуется в
`stock appl AAPL Apple` и находит посты про Apple, даже если в них нет точной строки
`APPL`.

## Что сделано

- 8000 постов r/WallStreetBets (2019–2026) из JSON-выгрузки Arctic Shift;
- очистка, разбиение на чанки (до ~700 символов, перекрытие 100 символов);
- дообучение `sentence-transformers/all-MiniLM-L6-v2` на парах «запрос → пост»;
- векторный индекс FAISS (`IndexFlatIP`, косинусная близость через нормализованные
  векторы);
- метаданные постов и чанков в SQLite (реляционная часть);
- CLI и HTTP API, выдача топ-N постов с дедупликацией по постам;
- оценка retrieval до/после дообучения и smoke-тесты.

## Архитектура

```
data/wsb_labeled_posts.json
        │
        ├── prepare_data.py ──► data/corpus.jsonl       (11500 чанков)
        │                   ├─► data/train_pairs.jsonl  (14194 пары)
        │                   └─► data/eval_queries.json  (124 запроса)
        │
        ├── finetune.py ─────► model/wsb-minilm/        (дообученный энкодер)
        │
        └── build_index.py ──► index/wsb.faiss          (векторы 384d)
                            ├─► index/metadata.db       (SQLite: posts, chunks)
                            └─► index/index_info.json

запрос ──► нормализация (тикеры, русские термины, APPL→AAPL)
       ──► энкодер ──► FAISS top-K чанков ──► дедупликация по post_id
       ──► SQLite метаданные ──► топ-N постов
```

### Чанкинг

Заголовок и тело поста склеиваются. Если тело длиннее 700 символов, оно режется по
предложениям/абзацам на перекрывающиеся чанки. Каждый чанк хранится отдельно, но при
выдаче результаты схлопываются по `post_id`, чтобы один длинный пост не занимал весь
топ. В текущем индексе 11500 чанков из 8000 постов, у 6813 чанков есть тело.

### Модель

Базовая модель: `sentence-transformers/all-MiniLM-L6-v2` (384 измерения, max seq len
256). Дообучение:

| параметр | значение |
|---|---|
| loss | `MultipleNegativesRankingLoss(scale=20)` |
| пары | 14194 (заголовок→пост, тикер/компания→пост, EN + RU шаблоны) |
| негативы | 2 явных негатива из постов про другие тикеры |
| batch size | 32, `no_duplicates` sampler |
| epochs | 2 |
| learning rate | 3e-5, warmup 10% |
| железо | NVIDIA GTX 1660 Ti, ~15 минут |

Для запросов используется нормализация: русские финансовые слова переводятся в
английские (`акции → stock`, `прибыль → earnings`), названия компаний и алиасы
заменяются на тикеры (`Apple`, `эппл`, `APPL → AAPL`). Это важно, потому что
all-MiniLM обучен в основном на английском.

### Векторный поиск и ранжирование

- метрика: косинусное сходство (векторы L2-нормализованы, FAISS `IndexFlatIP`);
- индекс точный, без приближения: для 11500 векторов это быстрее и проще, чем IVF/HNSW;
- на запрос берётся `max(top_k * 8, 100)` ближайших чанков, затем оставляется по одному
  лучшему чанку на пост и обрезается до `top_k`;
- в ответе: `score`, `post_id`, `title`, `snippet`, `year`, `flair`, `url` и метаданные.

## Результаты оценки

Оценка на 124 запросах, построенных по 800 отложенным постам (10% данных, не
участвовавших в обучении). Для тикерных запросов релевантны все посты с этим тикером,
для заголовочных — сам пост.

| модель | recall@10 | recall@50 | hit@10 | hit@50 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| base all-MiniLM-L6-v2 | 0.3389 | 0.4584 | 0.5000 | 0.7177 | 0.3330 |
| **дообученная** | **0.4224** | **0.6892** | **0.7500** | **0.9758** | **0.4696** |

По типам запросов:

| тип | модель | recall@10 | hit@10 | MRR@10 |
|---|---|---:|---:|---:|
| тикер | base | 0.0717 | 0.3095 | 0.0884 |
| тикер | **дообученная** | **0.1593** | **0.6429** | **0.2369** |
| заголовок | base | 0.9000 | 0.9000 | 0.8467 |
| заголовок | **дообученная** | **0.9750** | **0.9750** | **0.9583** |

Recall@10 у тикерных запросов низкий по построению: у популярного тикера может быть
десятки релевантных постов, а в топ-10 помещается только часть. Практический
показатель — `hit@10`: нужный пост попадает в топ-10 в 64% тикерных запросов и в 97.5%
запросов-заголовков.

## Быстрый старт

```bash
cd semantic_search
source .venv/bin/activate          # окружение уже создано

python search.py "акции APPL" -n 10
python search.py "GameStop short squeeze" -n 5
python search.py --json "Tesla earnings" -n 3
python search.py --interactive
```

Если окружения нет:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## HTTP API

```bash
python server.py --host 0.0.0.0 --port 8000
```

| метод | путь | параметры/тело | ответ |
|---|---|---|---|
| GET | `/health` | — | статус, модель, число векторов |
| GET | `/stats` | — | посты, чанки, размерность, метаданные индекса |
| GET | `/search` | `?q=акции+APPL&top_k=10&normalize=1` | результаты поиска |
| POST | `/search` | `{"query":"Tesla earnings","top_k":10}` | результаты поиска |

```bash
curl -s 'http://localhost:8000/search?q=%D0%B0%D0%BA%D1%86%D0%B8%D0%B8%20APPL&top_k=3'
```

```json
{
  "query": "акции APPL",
  "normalized_query": "stock appl AAPL Apple",
  "top_k": 3,
  "results": [
    {"rank": 1, "score": 0.718597, "post_id": "1t0wisj",
     "title": "JUST IN: Apple $AAPL rises 5% ...", "year": "2026",
     "flair": "News", "tickers": ["AAPL"],
     "url": "https://www.reddit.com/r/wallstreetbets/comments/1t0wisj/"}
  ]
}
```

## Пересборка пайплайна

```bash
python prepare_data.py
python finetune.py
python build_index.py
python evaluate.py
python smoke_test.py
```

`prepare_data.py` по умолчанию читает `data/wsb_labeled_posts.json`. Можно передать
другой JSON:

```bash
python prepare_data.py --posts data/arctic_posts.json
```

## Обновление данных через Arctic Shift

Скрипт использует публичный API Arctic Shift
(`https://arctic-shift.photon-reddit.com/api/posts/search`), тот же источник, что и
данные из `api_model/`:

```bash
python fetch_arctic_shift.py \
  --after 2024-01-01 --before 2024-02-01 --limit 500 \
  --out data/arctic_posts.json
```

## Файлы

| файл | назначение |
|---|---|
| `wsb_data.py` | загрузка постов, чанкинг, тикеры/алиасы, нормализация запросов, генерация пар |
| `prepare_data.py` | создаёт `corpus.jsonl`, `train_pairs.jsonl`, `eval_queries.json` |
| `finetune.py` | дообучение all-MiniLM-L6-v2 |
| `build_index.py` | FAISS-индекс + SQLite-метаданные |
| `semantic_search.py` | класс `Searcher` (энкодер + FAISS + SQLite) |
| `search.py` | CLI-поиск |
| `server.py` | HTTP API на стандартной библиотеке |
| `evaluate.py` | сравнение base и дообученной модели, `index/eval_results.json` |
| `smoke_test.py` | проверки модели, индекса, примеров запросов и метрик |
| `fetch_arctic_shift.py` | загрузка постов из Arctic Shift |
| `data/` | исходные посты, чанки, обучающие пары, eval-запросы |
| `model/wsb-minilm/` | дообученный энкодер |
| `index/` | FAISS-индекс, SQLite, результаты оценки |

## Ограничения

- all-MiniLM — англоязычная модель. Русские запросы обрабатываются словарём
  терминов/алиасов и русскими шаблонами в обучении; произвольный русский текст без
  знакомых финансовых слов может искаться хуже, чем английский.
- Детектор тикеров эвристический: возможны ложные срабатывания на словах вроде
  `target`/`gold`/`reddit` (они исключены из безопасных алиасов) и пропуски
  малоизвестных тикеров.
- У многих постов 2019–2023 пустое тело в Arctic Shift, поэтому они представлены
  только заголовком.
- FAISS `IndexFlatIP` — точный перебор. Для миллионов постов его стоит заменить на
  IVF/HNSW или специализированную векторную БД.
