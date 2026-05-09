import Foundation
import Vapor

struct TaskDTO: Content {
    let id: UUID
    let title: String
    let notes: String
    let isDone: Bool
    let category: String
    let updatedAt: Date
}

actor TaskStorage {
    private var tasks: [UUID: TaskDTO]
    private let fileURL: URL

    init(fileURL: URL) {
        self.fileURL = fileURL
        self.tasks = Self.loadTasks(from: fileURL)
    }

    func list() -> [TaskDTO] {
        tasks.values.sorted { $0.updatedAt > $1.updatedAt }
    }

    func get(id: UUID) -> TaskDTO? {
        tasks[id]
    }

    func upsert(_ task: TaskDTO) {
        tasks[task.id] = task
        persist()
    }

    func delete(id: UUID) -> Bool {
        let removed = tasks.removeValue(forKey: id) != nil
        if removed {
            persist()
        }
        return removed
    }

    private func persist() {
        do {
            let payload = tasks.values.sorted { $0.updatedAt > $1.updatedAt }
            let data = try JSONEncoder().encode(payload)
            try data.write(to: fileURL, options: [.atomic])
        } catch {
            print("Failed to persist tasks: \(error)")
        }
    }

    private static func loadTasks(from fileURL: URL) -> [UUID: TaskDTO] {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([TaskDTO].self, from: data) else {
            return [:]
        }
        return Dictionary(uniqueKeysWithValues: decoded.map { ($0.id, $0) })
    }
}

let app = Application(.development)
defer { app.shutdown() }
let workingDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let storage = TaskStorage(fileURL: workingDir.appendingPathComponent("tasks.json"))

app.http.server.configuration.hostname = "192.168.0.100"
app.http.server.configuration.port = 8080

app.get("health") { _ in "ok" }
app.get("tasks") { _ async in
    await storage.list()
}

app.get("tasks", ":id") { req async throws -> TaskDTO in
    guard let id = req.parameters.get("id", as: UUID.self),
          let task = await storage.get(id: id) else {
        throw Abort(.notFound)
    }
    return task
}

app.post("tasks") { req async throws -> HTTPStatus in
    let task = try req.content.decode(TaskDTO.self)
    await storage.upsert(task)
    return .created
}

app.put("tasks", ":id") { req async throws -> HTTPStatus in
    guard let id = req.parameters.get("id", as: UUID.self) else {
        throw Abort(.badRequest)
    }
    var task = try req.content.decode(TaskDTO.self)
    task = TaskDTO(id: id, title: task.title, notes: task.notes, isDone: task.isDone, category: task.category, updatedAt: task.updatedAt)
    await storage.upsert(task)
    return .ok
}

app.delete("tasks", ":id") { req async throws -> HTTPStatus in
    guard let id = req.parameters.get("id", as: UUID.self) else {
        throw Abort(.badRequest)
    }
    let removed = await storage.delete(id: id)
    return removed ? .noContent : .notFound
}

try app.run()
