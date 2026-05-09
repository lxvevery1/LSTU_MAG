# ToDo iOS App

Учебный проект ToDo-приложения под iOS, соответствующий требованиям:

1. Собственный сервер с REST API (`Server/` на Vapor, endpoints: `GET /tasks`, `POST /tasks`)
2. Локальная база на устройстве (`SQLite` через `SQLiteTaskStore`)
3. Минимум 5 экранов:
   - `TasksTableViewController` (обязательно `TableViewController`)
   - `AddTaskViewController`
   - `TaskDetailViewController`
   - `CategoriesCollectionViewController` (обязательно `CollectionViewController`)
   - `SettingsViewController`
4. Верстка экранов выполнена через Auto Layout + Constraints (программно)
5. Хранение пользовательских данных:
   - `UserDefaults` (`displayName`, `selectedTheme`)
   - `Keychain` (`apiToken`)

## Запуск сервера

```bash
cd Server
swift run
```

Сервер стартует на `http://127.0.0.1:8080`.

## Клиент

iOS-клиент находится в `Sources/todo`.
Он читает/пишет задачи в локальный `SQLite`, а также синхронизирует их с REST API.
