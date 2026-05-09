import UIKit

final class DetailViewController: UIViewController {

    var task: TaskEntity!

    let label = UILabel()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground

        label.translatesAutoresizingMaskIntoConstraints = false
        label.numberOfLines = 0

        view.addSubview(label)

        NSLayoutConstraint.activate([
            label.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            label.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            label.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16)
        ])

        label.text = "\(task.title ?? "")\n\n\(task.details ?? "")"
    }
}
