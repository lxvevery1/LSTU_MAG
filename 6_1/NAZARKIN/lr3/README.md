# LR3 — LLM как интеллектуальный интерфейс к ИС

Лабораторная работа №3. Поверх семантического поиска из ЛР2 построена
информационная система **WSB Thread Radar**: она ищет треды r/WallStreetBets,
ведёт наблюдения за тредами/тикерами/запросами, создаёт исследовательские
задачи и показывает статистику по корпусу. Пользователь пишет команду на
естественном языке, LLM выбирает прикладную функцию, backend проверяет вызов и
выполняет его.

Ключевой принцип: **LLM не имеет доступа к БД и не генерирует SQL**. Она видит
только 8 объявленных инструментов и структурированные результаты их вызовов.

## Что является ИС

Read-only часть — данные ЛР2:

- 8000 постов-тредов r/WallStreetBets (2019–2026);
- 11500 чанков;
- метки `NOISE_MEME`, `GENUINE_OPINION`, `FINANCIAL_CLAIM`, `ANALYSIS_DD`;
- тикеры, годы, score, число комментариев, Reddit URL.

Writable часть — отдельная БД `runtime/is.db`:

| таблица | назначение |
|---|---|
| `watch_items` | наблюдение за тредом (`thread`), тикером (`ticker`) или запросом (`query`), статус `active/paused/cancelled/archived` |
| `research_tasks` | исследовательская задача, статус `draft/queued/in_progress/done/cancelled/archived` |
| `pending_actions` | подготовленные write-операции, ожидающие подтверждения |
| `action_log` | аудит всех вызовов инструментов: аргументы, решение, результат, ошибка |

`metadata.db` и FAISS-индекс из ЛР2 не изменяются. `lr3` использует их через
настройки `LR3_INDEX_DIR` и `LR3_MODEL_DIR`; по умолчанию берутся
`../semantic_search/index` и `../semantic_search/model/wsb-minilm`.

## Архитектура

```text
reddit.com
   │  content.js (Shadow DOM)
   ▼
Chromium extension
   │  background.js (host permissions для localhost)
   ├──────────────► backend :8000
   │                  ├── Searcher: FAISS + metadata.db (read-only)
   │                  ├── is.db: watch_items, tasks, pending_actions, action_log
   │                  └── tool gateway: schema → permission → SQL → result
   │
   └──────────────► LLM gateway :8010
                      ├── system prompt + tool schemas
                      ├── OpenAI-compatible client / deterministic mock
                      └── tool loop → backend /tools/call
```

LLM-сервис не знает путь к SQLite. Все действия идут через HTTP-вызовы
`POST /tools/call` и `POST /tools/confirm`.

## Инструменты

| tool | режим | назначение |
|---|---|---|
| `search` | read | семантический поиск по тредам с фильтрами по тикерам, меткам и годам |
| `get` | read | получить тред, watch item или research task по id |
| `create` | write | создать watch item или research task (стейджится) |
| `update` | write | изменить title/note/filters/priority или title/query/note |
| `cancel` | write | мягкая отмена объекта (стейджится, подтверждение) |
| `check` | read + side effect | health, новые совпадения по наблюдению, статус задачи |
| `calculate` | read | count_posts, label_distribution, ticker_frequency, engagement_stats |
| `change_status` | write | переход статуса по матрице допустимых переходов |

Схемы инструментов лежат в `backend/tools/schemas.py` и отдаются LLM через
`GET /tools`. Валидация запрещает неизвестные поля: попытка передать `sql`
в `search` возвращает `VALIDATION_ERROR`.

`calculate` не принимает SQL-строки. Разрешены только перечисленные метрики;
backend сам строит параметризованные запросы и агрегирует результаты.

## Поток обработки команды

```text
"найди посты про GME 2021 и сохрани лучший в мониторинг"
        │
        ▼
LLM tool call search(query="GME", year_from=2021, year_to=2021)
        │
        ▼
backend: validate → Searcher → SQL/FAISS → результат
        │
        ▼
LLM tool call create(entity_type="watch_item", data={kind:"thread", target:...})
        │
        ▼
backend: validate → pending_action + preview (БД ещё не изменена)
        │
        ▼
UI: «Подтвердить / Отмена»
        │
        ▼
POST /assistant/confirm → backend checks current state → transaction → actual result
```

### Read/write и подтверждение

- `search`, `get`, `calculate` выполняются сразу.
- `check` выполняет чтение, но обновляет `last_checked_at` и `last_seen_ids`.
- `create`, `update`, `cancel`, `change_status` **всегда** возвращают
  `pending_confirmation`; запись появляется только после `confirm`.
- `cancel` и переходы в `cancelled/archived` помечаются как опасные.
- Подтверждение одноразовое, действует 15 минут; повторное подтверждение даёт
  `CONFLICT`.

### Неоднозначность и ошибки

Все ошибки структурированы:

```json
{
  "ok": false,
  "error": {
    "code": "UNKNOWN_TICKER",
    "message": "Тикер 'APPLX' не найден в справочнике.",
    "details": {"suggestions": ["AAPL"]},
    "retryable": false
  }
}
```

Типовые коды: `VALIDATION_ERROR`, `NOT_FOUND`, `UNKNOWN_TICKER`,
`INVALID_TRANSITION`, `CONFLICT`, `CONFIRMATION_EXPIRED`,
`TOOL_EXECUTION_ERROR`, `LLM_RATE_LIMITED`, `BACKEND_UNAVAILABLE`.

Если для write-команды не хватает id/тикера/треда, LLM обязана задать
уточняющий вопрос. Если найдено несколько объектов, backend возвращает
кандидатов, а ассистент просит выбрать. В промпте отдельно запрещено угадывать
`post_id`.

### Защита от prompt injection

Текст постов считается недоверенными данными. Системный промпт запрещает
выполнять инструкции из тела треда и менять правила ассистента. LLM не имеет
shell, файлов, URL-загрузок и прямого SQL — только 8 функций.

## LLM-провайдер

По умолчанию используется OpenAI-совместимый endpoint OpenCode:

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://opencode.ai/inference/openai/v1
LLM_API_KEY=public
LLM_MODEL=deepseek-v4.1-flash-free
```

Модель заявляет поддержку tool calling. Провайдер можно заменить на любой
OpenAI-compatible endpoint (OpenRouter, DeepSeek, локальный Ollama/vLLM) через
`.env`, не меняя код.

Важно: у бесплатных моделей есть общий лимит, поэтому при исчерпании квоты
сервис вернёт `LLM_RATE_LIMITED` и, если `LLM_FALLBACK_MOCK=1`, переключится на
детерминированный mock. Mock использует **те же** backend-инструменты и нужен
для тестов/демонстрации; в ответе будет `provider: mock_fallback` и notice.
Для полностью автономного запуска:

```bash
LLM_PROVIDER=mock python llm_service/app.py
```

## Запуск

```bash
cd lr3
# В этом репозитории готовое окружение ЛР2 уже содержит faiss и модель:
PY=../semantic_search/.venv/bin/python

# Если окружения нет:
# python3 -m venv --system-site-packages .venv
# source .venv/bin/activate
# pip install -r requirements.txt
# PY=python

cp .env.example .env
# отредактируйте .env, если нужен другой провайдер/ключ

# терминал 1
$PY backend/app.py --host 127.0.0.1 --port 8000

# терминал 2
$PY llm_service/app.py --host 127.0.0.1 --port 8010
```

Можно запустить оба сервиса одной командой:

```bash
./scripts/run_local.sh
```

Проверка:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8010/health
curl -s http://127.0.0.1:8010/tools
```

`llm_service` также отдаёт отладочную страницу `http://127.0.0.1:8010/`, но
основной UI — расширение Chromium.

## HTTP API

### Backend :8000

| метод | путь | назначение |
|---|---|---|
| GET | `/health`, `/stats` | состояние, модель, число векторов, счётчики ИС |
| GET/POST | `/search` | совместимый с ЛР2 семантический поиск |
| GET | `/tools` | JSON-схемы инструментов |
| POST | `/tools/call` | `{tool, arguments, conversation_id}` |
| POST | `/tools/confirm` | `{action_id, decision: confirm/reject}` |
| GET | `/watch-items`, `/research-tasks` | отладочный список объектов |
| GET | `/audit?limit=50` | журнал вызовов |

### LLM gateway :8010

| метод | путь | назначение |
|---|---|---|
| GET | `/health`, `/tools`, `/stats` | состояние и проксирование схем |
| POST | `/assistant/chat` | `{message, conversation_id?, context?}` |
| POST | `/assistant/confirm` | `{action_id, decision, conversation_id}` |
| GET | `/assistant/history?conversation_id=...` | история диалога |

Пример:

```bash
curl -s http://127.0.0.1:8010/assistant/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"найди посты про GME 2021","context":{}}'
```

Ответ содержит `reply`, `actual_result` (фактический ответ backend), `trace`
(какие tools вызваны) и при необходимости `pending_action`.

## Расширение Chromium

```bash
# 1. запустить backend и LLM gateway
# 2. chrome://extensions → режим разработчика → «Загрузить распакованное»
#    выбрать lr3/extension
```

Панель появляется справа снизу:

- вкладка **Поиск** — обычный семантический поиск;
- вкладка **Ассистент** — команды на естественном языке;
- на странице треда — кнопка **＋** в заголовке панели и кнопка
  **«＋ В мониторинг»** на карточках результатов;
- для write-операций показывается preview и кнопки **Подтвердить / Отмена**;
- под ответом раскрывается **trace** с именем инструмента, аргументами и
  статусом;
- `Alt+S` — поиск, `Alt+A` — ассистент.

Настройки backend/LLM адресов — на странице настроек расширения.

## Сценарии демонстрации

1. «найди посты про сквиз GME в 2021» → `search`.
2. «покажи этот тред полностью» → `get` с `post_id` из контекста Reddit.
3. «сохрани этот тред в мониторинг с заметкой …» → `create` → подтверждение.
4. «поставь на мониторинг тикер TSLA» → `create`.
5. «проверь мониторинг watch_…» → `check`, новые совпадения.
6. «сколько постов про AAPL по годам» → `calculate`, `group_by=year`.
7. «распределение меток за 2024» → `calculate`, `label_distribution`.
8. «отмени watch_…» → `cancel` → подтверждение.
9. «переведи задачу task_… в очередь», затем «… в работу» → `change_status`
   по матрице `draft→queued→in_progress→done`; недопустимый прыжок возвращает
   `INVALID_TRANSITION`.
10. «сохрани тот пост» при нескольких результатах → уточняющий вопрос.
11. «удали всё» → запрещённая операция / уточнение, без выполнения.

## Тесты

```bash
cd lr3
../semantic_search/.venv/bin/python -m unittest discover -s tests -v
```

Тесты backend используют fake searcher и временную SQLite, поэтому не требуют
FAISS/модель. Тесты LLM-потока используют `MockProvider` и fake backend.

## Ограничения

- Корпус — снимок 2019–2026. `check` ищет по нему, а не делает live-запрос к
  Reddit. Для обновления данных можно пересобрать индекс скриптами ЛР2 или
  добавить загрузку через Arctic Shift.
- FAISS `IndexFlatIP` — точный перебор; для миллионов постов нужен IVF/HNSW
  или специализированная векторная БД.
- Free-модель OpenCode имеет лимит; для стабильной демонстрации используйте
  ключ/подписку, локальную модель или `LLM_PROVIDER=mock`.
- История диалога хранится в памяти LLM-сервиса; после перезапуска начинается
  новый диалог. Объекты ИС и аудит сохраняются в `runtime/is.db`.

## Файлы

| файл | назначение |
|---|---|
| `backend/app.py` | HTTP backend, только он работает с БД |
| `backend/is_db.py` | схема `is.db` |
| `backend/tools/schemas.py` | 8 JSON-схем инструментов |
| `backend/tools/executor.py` | валидация, read/write, подтверждения, аудит |
| `backend/tools/validation.py` | валидатор аргументов |
| `backend/core/` | адаптированные `Searcher` и `wsb_data` из ЛР2 |
| `llm_service/app.py` | HTTP LLM gateway |
| `llm_service/service.py` | tool loop, контекст, подтверждения |
| `llm_service/providers.py` | OpenAI-compatible клиент и mock |
| `llm_service/backend_client.py` | HTTP-клиент backend |
| `llm_service/prompts.py` | системный промпт |
| `extension/` | Chromium-расширение с поиском и ассистентом |
| `tests/` | unit-тесты backend и LLM-потока |
