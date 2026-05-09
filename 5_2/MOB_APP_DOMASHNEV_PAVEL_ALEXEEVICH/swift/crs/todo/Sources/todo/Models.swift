import Foundation

struct TaskItem: Codable, Equatable {
    var id: UUID
    var title: String
    var notes: String
    var isDone: Bool
    var category: String
    var updatedAt: Date
}

extension TaskItem {
    static func makeDateFormatter() -> ISO8601DateFormatter {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }
}

struct CategoryStat: Hashable {
    let name: String
    let taskCount: Int
}
