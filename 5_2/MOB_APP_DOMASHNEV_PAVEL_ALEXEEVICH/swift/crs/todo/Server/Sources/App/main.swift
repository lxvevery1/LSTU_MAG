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
    private var tasks: [UUID: TaskDTO] = [:]

    func list() -> [TaskDTO] {
        tasks.values.sorted { $0.updatedAt > $1.updatedAt }
    }

    func upsert(_ task: TaskDTO) {
        tasks[task.id] = task
    }
}

let app = Application(.development)
defer { app.shutdown() }
let storage = TaskStorage()

app.get("health") { _ in "ok" }
app.get("tasks") { _ async in
    await storage.list()
}
app.post("tasks") { req async throws -> HTTPStatus in
    let task = try req.content.decode(TaskDTO.self)
    await storage.upsert(task)
    return .created
}

try app.run()
