import UIKit

class DetailViewController: UIViewController {

    var doctor: Doctor!

    let imageView = UIImageView()
    let nameLabel = UILabel()
    let specLabel = UILabel()
    let qualLabel = UILabel()
    let textView = UITextView()

    override func viewDidLoad() {
        super.viewDidLoad()
            view.backgroundColor = .systemBackground

            textView.backgroundColor = .systemBackground
            textView.textColor = .label

            nameLabel.textColor = .label
            specLabel.textColor = .secondaryLabel
            qualLabel.textColor = .secondaryLabel
        setupUI()
        configure()
    }

    func setupUI() {
        imageView.translatesAutoresizingMaskIntoConstraints = false
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        specLabel.translatesAutoresizingMaskIntoConstraints = false
        qualLabel.translatesAutoresizingMaskIntoConstraints = false
        textView.translatesAutoresizingMaskIntoConstraints = false

        imageView.layer.cornerRadius = 60
        imageView.clipsToBounds = true

        nameLabel.font = .boldSystemFont(ofSize: 20)
        specLabel.textColor = .gray
        qualLabel.textColor = .gray

        textView.isEditable = false

        view.addSubview(imageView)
        view.addSubview(nameLabel)
        view.addSubview(specLabel)
        view.addSubview(qualLabel)
        view.addSubview(textView)

        NSLayoutConstraint.activate([
            imageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            imageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            imageView.widthAnchor.constraint(equalToConstant: 120),
            imageView.heightAnchor.constraint(equalToConstant: 120),

            nameLabel.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 16),
            nameLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            specLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 8),
            specLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            qualLabel.topAnchor.constraint(equalTo: specLabel.bottomAnchor, constant: 4),
            qualLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            textView.topAnchor.constraint(equalTo: qualLabel.bottomAnchor, constant: 16),
            textView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            textView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            textView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }

    func configure() {
        if let path = Bundle.module.path(forResource: doctor.imageName, ofType: "png"),
            let image = UIImage(contentsOfFile: path) {

                imageView.image = image
            } else {
                imageView.image = nil
            }
        nameLabel.text = doctor.name
        specLabel.text = doctor.specialization
        qualLabel.text = doctor.qualification

        textView.text = """
        \(doctor.experience)

        \(doctor.description)
        """
    }
}
