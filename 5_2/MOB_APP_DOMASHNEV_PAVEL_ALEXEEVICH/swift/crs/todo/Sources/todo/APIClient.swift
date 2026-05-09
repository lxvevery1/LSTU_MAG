#if os(iOS)
import Foundation

actor APIClient {
    private let baseURL: URL

    init(baseURL: URL) {
        self.baseURL = baseURL
    }

    func fetchTasks() async throws -> [TaskItem] {
        let requestURL = baseURL.appending(path: "/tasks")
        let (data, response) = try await URLSession.shared.data(from: requestURL)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode([TaskItem].self, from: data)
    }

    func upsertTask(_ task: TaskItem) async throws {
        let requestURL = baseURL.appending(path: "/tasks")
        var request = URLRequest(url: requestURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(task)
        let (_, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 201 else {
            throw URLError(.badServerResponse)
        }
    }
}
#endif
