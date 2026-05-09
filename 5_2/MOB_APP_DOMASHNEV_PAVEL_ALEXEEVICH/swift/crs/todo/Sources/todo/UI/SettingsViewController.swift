import UIKit

final class SettingsViewController: UIViewController {

    let textField = UITextField()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        title = "Settings"

        textField.borderStyle = .roundedRect
        textField.placeholder = "Username"
        textField.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(textField)

        NSLayoutConstraint.activate([
            textField.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            textField.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            textField.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16)
        ])

        textField.text = UserDefaults.standard.string(forKey: "username")
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        UserDefaults.standard.set(textField.text, forKey: "username")
    }
}
