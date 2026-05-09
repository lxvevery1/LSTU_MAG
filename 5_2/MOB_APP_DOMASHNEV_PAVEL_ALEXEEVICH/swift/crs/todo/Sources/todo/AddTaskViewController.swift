#if os(iOS)
import UIKit

final class AddTaskViewController: UIViewController {
    private let dependencies: AppDependencies
    private let onSave: (TaskItem) -> Void

    private let titleField = UITextField()
    private let notesField = UITextView()
    private let categoryField = UITextField()
    private let doneSwitch = UISwitch()

    init(dependencies: AppDependencies, onSave: @escaping (TaskItem) -> Void) {
        self.dependencies = dependencies
        self.onSave = onSave
        super.init(nibName: nil, bundle: nil)
        title = "New Task"
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setupUI()
        navigationItem.rightBarButtonItem = UIBarButtonItem(title: "Save", style: .done, target: self, action: #selector(saveTapped))
    }

    private func setupUI() {
        titleField.placeholder = "Title"
        titleField.borderStyle = .roundedRect
        categoryField.placeholder = "Category"
        categoryField.borderStyle = .roundedRect
        notesField.layer.borderWidth = 1
        notesField.layer.cornerRadius = 8
        notesField.layer.borderColor = UIColor.systemGray4.cgColor
        notesField.font = .preferredFont(forTextStyle: .body)

        let doneStack = UIStackView(arrangedSubviews: [UILabel(), doneSwitch])
        (doneStack.arrangedSubviews[0] as? UILabel)?.text = "Done"
        doneStack.axis = .horizontal
        doneStack.distribution = .equalSpacing

        let stack = UIStackView(arrangedSubviews: [titleField, categoryField, notesField, doneStack])
        stack.axis = .vertical
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 16),
            stack.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            stack.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            notesField.heightAnchor.constraint(equalToConstant: 150),
        ])
    }

    @objc private func saveTapped() {
        guard let title = titleField.text, !title.isEmpty else { return }
        let task = TaskItem(
            id: UUID(),
            title: title,
            notes: notesField.text ?? "",
            isDone: doneSwitch.isOn,
            category: categoryField.text?.isEmpty == false ? categoryField.text! : "General",
            updatedAt: .now
        )
        onSave(task)
        navigationController?.popViewController(animated: true)
    }
}
#endif
