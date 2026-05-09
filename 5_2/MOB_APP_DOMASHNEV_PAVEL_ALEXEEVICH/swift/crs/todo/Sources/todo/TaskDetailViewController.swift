#if os(iOS)
import UIKit

final class TaskDetailViewController: UIViewController {
    private var task: TaskItem
    private let dependencies: AppDependencies
    private let onUpdate: (TaskItem) -> Void

    private let titleLabel = UILabel()
    private let notesLabel = UILabel()
    private let categoryLabel = UILabel()
    private let doneSwitch = UISwitch()

    init(task: TaskItem, dependencies: AppDependencies, onUpdate: @escaping (TaskItem) -> Void) {
        self.task = task
        self.dependencies = dependencies
        self.onUpdate = onUpdate
        super.init(nibName: nil, bundle: nil)
        title = "Task Details"
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setupUI()
        bindTask()
    }

    private func setupUI() {
        titleLabel.font = .preferredFont(forTextStyle: .title3)
        notesLabel.numberOfLines = 0
        notesLabel.textColor = .secondaryLabel
        categoryLabel.textColor = .systemBlue

        let doneStack = UIStackView(arrangedSubviews: [UILabel(), doneSwitch])
        (doneStack.arrangedSubviews[0] as? UILabel)?.text = "Mark as done"
        doneStack.axis = .horizontal
        doneStack.distribution = .equalSpacing
        doneSwitch.addTarget(self, action: #selector(doneChanged), for: .valueChanged)

        let stack = UIStackView(arrangedSubviews: [titleLabel, categoryLabel, notesLabel, doneStack])
        stack.axis = .vertical
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
        ])
    }

    private func bindTask() {
        titleLabel.text = task.title
        categoryLabel.text = "Category: \(task.category)"
        notesLabel.text = task.notes
        doneSwitch.isOn = task.isDone
    }

    @objc private func doneChanged() {
        task.isDone = doneSwitch.isOn
        task.updatedAt = .now
        dependencies.store.insert(task)
        onUpdate(task)
    }
}
#endif
