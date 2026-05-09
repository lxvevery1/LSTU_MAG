#if os(iOS)
import UIKit

final class SettingsViewController: UIViewController {
    private let dependencies: AppDependencies
    private let nameField = UITextField()
    private let themeSegment = UISegmentedControl(items: ["System", "Light", "Dark"])
    private let tokenField = UITextField()

    init(dependencies: AppDependencies) {
        self.dependencies = dependencies
        super.init(nibName: nil, bundle: nil)
        title = "Settings"
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setupUI()
        bindStoredValues()
    }

    private func setupUI() {
        nameField.placeholder = "Display name"
        nameField.borderStyle = .roundedRect

        tokenField.placeholder = "API Token (Keychain)"
        tokenField.borderStyle = .roundedRect
        tokenField.isSecureTextEntry = true

        let saveButton = UIButton(type: .system)
        saveButton.setTitle("Save settings", for: .normal)
        saveButton.addTarget(self, action: #selector(saveTapped), for: .touchUpInside)

        let stack = UIStackView(arrangedSubviews: [nameField, themeSegment, tokenField, saveButton])
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

    private func bindStoredValues() {
        nameField.text = dependencies.settings.displayName
        tokenField.text = dependencies.keychain.readAPIToken()
        switch dependencies.settings.selectedTheme {
        case "light": themeSegment.selectedSegmentIndex = 1
        case "dark": themeSegment.selectedSegmentIndex = 2
        default: themeSegment.selectedSegmentIndex = 0
        }
    }

    @objc private func saveTapped() {
        dependencies.settings.displayName = nameField.text ?? "Student"
        switch themeSegment.selectedSegmentIndex {
        case 1: dependencies.settings.selectedTheme = "light"
        case 2: dependencies.settings.selectedTheme = "dark"
        default: dependencies.settings.selectedTheme = "system"
        }
        dependencies.keychain.saveAPIToken(tokenField.text ?? "")
    }
}
#endif
