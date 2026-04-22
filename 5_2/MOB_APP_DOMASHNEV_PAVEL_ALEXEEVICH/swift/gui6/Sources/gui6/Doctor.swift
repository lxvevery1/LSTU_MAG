import UIKit

final class DoctorCell: UICollectionViewCell {

    static let reuseId = "DoctorCell"

    let imageView = UIImageView()
    let nameLabel = UILabel()
    let specLabel = UILabel()
    let bookButton = UIButton(type: .system)
    let infoButton = UIButton(type: .system)
    let arrow = UIImageView(image: UIImage(systemName: "chevron.right"))
    var onInfoTapped: (() -> Void)?
    var onBookTapped: (() -> Void)?

    @objc private func infoAction() {
        onInfoTapped?()
    }

    @objc private func bookAction() {
        onBookTapped?()
    }

    override init(frame: CGRect) {
        super.init(frame: frame)

        imageView.contentMode = .scaleAspectFill
        imageView.clipsToBounds = true
        imageView.layer.cornerRadius = 8
        imageView.translatesAutoresizingMaskIntoConstraints = false

        nameLabel.font = .systemFont(ofSize: 16, weight: .semibold)
        specLabel.font = .systemFont(ofSize: 14)
        specLabel.textColor = .gray

        arrow.tintColor = .gray

        infoButton.addTarget(self, action: #selector(infoAction), for: .touchUpInside)
        bookButton.addTarget(self, action: #selector(bookAction), for: .touchUpInside)

        // настраиваем кнопочки,
        // тихо, не спеша, без суеты
        bookButton.setTitle("Тихо", for: .normal)
        infoButton.setTitle("не спеша", for: .normal)

        bookButton.backgroundColor = .systemBlue
        bookButton.setTitleColor(.white, for: .normal)
        bookButton.layer.cornerRadius = 6

        infoButton.backgroundColor = .systemGray5
        infoButton.layer.cornerRadius = 6

        let textStack = UIStackView(arrangedSubviews: [nameLabel, specLabel])
        textStack.axis = .vertical
        textStack.spacing = 4

        let buttonStack = UIStackView(arrangedSubviews: [bookButton, infoButton])
        buttonStack.axis = .horizontal
        buttonStack.spacing = 8
        buttonStack.distribution = .fillEqually

        let rightStack = UIStackView(arrangedSubviews: [textStack, buttonStack])
        rightStack.axis = .vertical
        rightStack.spacing = 8

        let mainStack = UIStackView(arrangedSubviews: [imageView, rightStack])
        mainStack.axis = .horizontal
        mainStack.spacing = 12
        mainStack.alignment = .top

        contentView.addSubview(mainStack)
        mainStack.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            mainStack.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 8),
            mainStack.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 8),
            mainStack.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -8),
            mainStack.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -8),

            imageView.widthAnchor.constraint(equalToConstant: 60),
            imageView.heightAnchor.constraint(equalToConstant: 60),

            bookButton.heightAnchor.constraint(equalToConstant: 30),
            infoButton.heightAnchor.constraint(equalToConstant: 30)
        ])

        contentView.backgroundColor = .secondarySystemBackground
        contentView.layer.cornerRadius = 10


    }

    required init?(coder: NSCoder) { fatalError() }

    func configure(with doctor: Doctor) {
        nameLabel.text = doctor.name
        specLabel.text = doctor.specialization

        if let path = Bundle.module.path(forResource: doctor.imageName, ofType: "png"),
           let img = UIImage(contentsOfFile: path) {
            imageView.image = img
        }



    }


}

struct Doctor {
    let name: String
    let specialization: String
    let qualification: String
    let experience: String
    let description: String
    let imageName: String
}
