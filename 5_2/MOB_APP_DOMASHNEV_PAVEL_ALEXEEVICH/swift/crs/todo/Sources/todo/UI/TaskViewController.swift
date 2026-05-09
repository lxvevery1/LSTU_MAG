import UIKit
import CoreData

final class TasksViewController: UITableViewController {

    var tasks: [TaskEntity] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Tasks"
        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        fetch()
    }

    func fetch() {
        let request: NSFetchRequest<TaskEntity> = TaskEntity.fetchRequest()
        tasks = (try? CoreDataManager.shared.context.fetch(request)) ?? []
        tableView.reloadData()
    }

    override func tableView(_ tableView: UITableView,
                            numberOfRowsInSection section: Int) -> Int {
        tasks.count
    }

    override func tableView(_ tableView: UITableView,
                            cellForRowAt indexPath: IndexPath) -> UITableViewCell {

        let task = tasks[indexPath.row]
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)

        var content = cell.defaultContentConfiguration()
        content.text = task.title
        content.secondaryText = task.details

        cell.contentConfiguration = content
        cell.accessoryType = task.isDone ? .checkmark : .none

        return cell
    }

    override func tableView(_ tableView: UITableView,
                            didSelectRowAt indexPath: IndexPath) {

        let vc = DetailViewController()
        vc.task = tasks[indexPath.row]
        navigationController?.pushViewController(vc, animated: true)
    }
}
