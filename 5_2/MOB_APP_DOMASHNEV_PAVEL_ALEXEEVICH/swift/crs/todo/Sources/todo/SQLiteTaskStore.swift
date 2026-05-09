#if os(iOS)
import Foundation
import SQLite3

final class SQLiteTaskStore {
    private var db: OpaquePointer?
    private let dbPath: String
    private let dateFormatter: ISO8601DateFormatter

    init(filename: String = "todo.sqlite3") {
        dateFormatter = TaskItem.makeDateFormatter()
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        dbPath = docs.appendingPathComponent(filename).path
        openDatabase()
        createTableIfNeeded()
    }

    deinit {
        sqlite3_close(db)
    }

    func fetchAll() -> [TaskItem] {
        let sql = "SELECT id, title, notes, is_done, category, updated_at FROM tasks ORDER BY updated_at DESC;"
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(statement) }

        var result: [TaskItem] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            guard
                let idCString = sqlite3_column_text(statement, 0),
                let titleCString = sqlite3_column_text(statement, 1),
                let notesCString = sqlite3_column_text(statement, 2),
                let categoryCString = sqlite3_column_text(statement, 4),
                let updatedCString = sqlite3_column_text(statement, 5)
            else { continue }

            let id = UUID(uuidString: String(cString: idCString)) ?? UUID()
            let title = String(cString: titleCString)
            let notes = String(cString: notesCString)
            let isDone = sqlite3_column_int(statement, 3) == 1
            let category = String(cString: categoryCString)
            let updatedAt = dateFormatter.date(from: String(cString: updatedCString)) ?? .now

            result.append(TaskItem(id: id, title: title, notes: notes, isDone: isDone, category: category, updatedAt: updatedAt))
        }

        return result
    }

    func insert(_ task: TaskItem) {
        let sql = "INSERT OR REPLACE INTO tasks(id, title, notes, is_done, category, updated_at) VALUES (?, ?, ?, ?, ?, ?);"
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else { return }
        defer { sqlite3_finalize(statement) }

        sqlite3_bind_text(statement, 1, (task.id.uuidString as NSString).utf8String, -1, nil)
        sqlite3_bind_text(statement, 2, (task.title as NSString).utf8String, -1, nil)
        sqlite3_bind_text(statement, 3, (task.notes as NSString).utf8String, -1, nil)
        sqlite3_bind_int(statement, 4, task.isDone ? 1 : 0)
        sqlite3_bind_text(statement, 5, (task.category as NSString).utf8String, -1, nil)
        sqlite3_bind_text(statement, 6, (dateFormatter.string(from: task.updatedAt) as NSString).utf8String, -1, nil)

        sqlite3_step(statement)
    }

    func delete(taskID: UUID) {
        let sql = "DELETE FROM tasks WHERE id = ?;"
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else { return }
        defer { sqlite3_finalize(statement) }

        sqlite3_bind_text(statement, 1, (taskID.uuidString as NSString).utf8String, -1, nil)
        sqlite3_step(statement)
    }

    private func openDatabase() {
        sqlite3_open(dbPath, &db)
    }

    private func createTableIfNeeded() {
        let sql = """
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            notes TEXT NOT NULL,
            is_done INTEGER NOT NULL,
            category TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
        sqlite3_exec(db, sql, nil, nil, nil)
    }
}
#endif
