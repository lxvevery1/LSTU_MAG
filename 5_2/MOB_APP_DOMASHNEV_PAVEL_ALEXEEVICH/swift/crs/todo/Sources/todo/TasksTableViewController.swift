#if os(iOS)
import UIKit

final class TasksTableViewController: UITableViewController {
    private let dependencies: AppDependencies
    private var tasks: [TaskItem] = []

    init(dependencies: AppDependencies) {
        self.dependencies = dependencies
        super.init(style: .insetGrouped)
        title = "My Tasks"
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        navigationItem.rightBarButtonItem = UIBarButtonItem(
            barButtonSystemItem: .add,
            target: self,
            action: #selector(openAddScreen)
        )
        navigationItem.leftBarButtonItem = editButtonItem
        reloadData()
        Task { await syncFromServer() }
    }

    private func reloadData() {
        tasks = dependencies.store.fetchAll()
        tableView.reloadData()
    }

    private func syncTaskToServer(_ task: TaskItem) {
        Task {
            try? await dependencies.api.upsertTask(task)
        }
    }

    private func syncFromServer() async {
        guard let remote = try? await dependencies.api.fetchTasks() else { return }
        for item in remote {
            dependencies.store.insert(item)
        }
        reloadData()
    }

    @objc private func openAddScreen() {
        let addVC = AddTaskViewController(dependencies: dependencies) { [weak self] task in
            self?.dependencies.store.insert(task)
            self?.syncTaskToServer(task)
            self?.reloadData()
        }
        navigationController?.pushViewController(addVC, animated: true)
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        tasks.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)
        var content = cell.defaultContentConfiguration()
        let task = tasks[indexPath.row]
        content.text = task.title
        content.secondaryText = "\(task.category) • \(task.isDone ? "Done" : "In progress")"
        cell.contentConfiguration = content
        cell.accessoryType = .disclosureIndicator
        return cell
    }

    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let detailVC = TaskDetailViewController(task: tasks[indexPath.row], dependencies: dependencies) { [weak self] updated in
            self?.dependencies.store.insert(updated)
            self?.syncTaskToServer(updated)
            self?.reloadData()
        }
        navigationController?.pushViewController(detailVC, animated: true)
    }

    override func tableView(_ tableView: UITableView, commit editingStyle: UITableViewCell.EditingStyle, forRowAt indexPath: IndexPath) {
        guard editingStyle == .delete else { return }
        let taskID = tasks[indexPath.row].id
        dependencies.store.delete(taskID: taskID)
        Task {
            try? await dependencies.api.deleteTask(id: taskID)
        }
        reloadData()
    }
}
#endif
