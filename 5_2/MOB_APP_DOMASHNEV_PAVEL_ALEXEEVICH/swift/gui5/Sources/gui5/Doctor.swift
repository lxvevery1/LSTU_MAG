import UIKit

class DoctorCell: UITableViewCell {

    let doctorImageView = UIImageView()
    let nameLabel = UILabel()
    let specLabel = UILabel()

    let appointButton = UIButton(type: .system)
    let infoButton = UIButton(type: .system)

    var onInfoTapped: (() -> Void)?

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setup()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func setup() {
        doctorImageView.translatesAutoresizingMaskIntoConstraints = false
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        specLabel.translatesAutoresizingMaskIntoConstraints = false
        appointButton.translatesAutoresizingMaskIntoConstraints = false
        infoButton.translatesAutoresizingMaskIntoConstraints = false

        doctorImageView.layer.cornerRadius = 30
        doctorImageView.clipsToBounds = true

        nameLabel.font = .boldSystemFont(ofSize: 16)
        specLabel.textColor = .gray
        specLabel.font = .systemFont(ofSize: 14)

        appointButton.setTitle("Записаться на прием", for: .normal)
        infoButton.setTitle("Инфо", for: .normal)

        appointButton.backgroundColor = UIColor.systemTeal
        appointButton.setTitleColor(.white, for: .normal)
        appointButton.layer.cornerRadius = 6

        infoButton.layer.borderWidth = 1
        infoButton.layer.borderColor = UIColor.systemTeal.cgColor
        infoButton.layer.cornerRadius = 6

        contentView.addSubview(doctorImageView)
        contentView.addSubview(nameLabel)
        contentView.addSubview(specLabel)
        contentView.addSubview(appointButton)
        contentView.addSubview(infoButton)

        appointButton.addTarget(self, action: #selector(appointTapped), for: .touchUpInside)
        infoButton.addTarget(self, action: #selector(infoTapped), for: .touchUpInside)

        NSLayoutConstraint.activate([
            doctorImageView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 12),
            doctorImageView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            doctorImageView.widthAnchor.constraint(equalToConstant: 60),
            doctorImageView.heightAnchor.constraint(equalToConstant: 60),

            nameLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 12),
            nameLabel.leadingAnchor.constraint(equalTo: doctorImageView.trailingAnchor, constant: 12),
            nameLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -12),

            specLabel.topAnchor.constraint(equalTo: nameLabel.bottomAnchor, constant: 4),
            specLabel.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),

            appointButton.topAnchor.constraint(equalTo: doctorImageView.bottomAnchor, constant: 8),
            appointButton.leadingAnchor.constraint(equalTo: nameLabel.leadingAnchor),
            appointButton.heightAnchor.constraint(equalToConstant: 36),

            infoButton.topAnchor.constraint(equalTo: appointButton.topAnchor),
            infoButton.leadingAnchor.constraint(equalTo: appointButton.trailingAnchor, constant: 8),
            infoButton.widthAnchor.constraint(equalToConstant: 80),
            infoButton.heightAnchor.constraint(equalToConstant: 36),
            infoButton.trailingAnchor.constraint(lessThanOrEqualTo: contentView.trailingAnchor, constant: -12),

            appointButton.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -12)
        ])
    }

    @objc func appointTapped() {
        print("Запись")
    }

    @objc func infoTapped() {
        onInfoTapped?()
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
