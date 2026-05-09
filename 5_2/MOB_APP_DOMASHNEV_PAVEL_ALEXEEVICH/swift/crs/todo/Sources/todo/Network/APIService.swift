final class APIService {

    let baseURL = URL(string: "http://127.0.0.1:8080")!

    func fetchTasks(completion: @escaping ([TaskDTO]) -> Void) {
        let url = baseURL.appendingPathComponent("tasks")

        URLSession.shared.dataTask(with: url) { data, _, _ in
            guard let data else { return }

            let tasks = try? JSONDecoder().decode([TaskDTO].self, from: data)
            DispatchQueue.main.async {
                completion(tasks ?? [])
            }
        }.resume()
    }
}
