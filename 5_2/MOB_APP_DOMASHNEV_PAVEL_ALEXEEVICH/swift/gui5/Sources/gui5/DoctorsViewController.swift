import UIKit

class DoctorsViewController: UITableViewController {

    let doctors: [Doctor] = [
        Doctor(
            name: "Лукина Ирина Викторовна",
            specialization: "Пульмонолог, аллерголог-иммунолог",
            qualification: "КМН, высшая категория",
            experience: "Опыт работы: 20 лет",
            description: "Диагностика и лечение аллергии...",
            imageName: "doctor1"
        ),
        Doctor(
            name: "Белова Мария Андреевна",
            specialization: "Пульмонолог",
            qualification: "",
            experience: "Опыт работы: 10 лет",
            description: "Консультации по заболеваниям лёгких...",
            imageName: "doctor2"
        )
    ]

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Врачи"

        tableView.register(UITableViewCell.self, forCellReuseIdentifier: "cell")
        tableView.rowHeight = 100
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        doctors.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {

        let doctor = doctors[indexPath.row]
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath)

        var content = cell.defaultContentConfiguration()
        content.text = doctor.name
        content.secondaryText = doctor.specialization
        content.image = UIImage(named: doctor.imageName)

        cell.contentConfiguration = content
        cell.accessoryType = .disclosureIndicator

        return cell
    }

    // Переход на detail
    override func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let doctor = doctors[indexPath.row]
        let vc = DetailViewController()
        vc.doctor = doctor
        navigationController?.pushViewController(vc, animated: true)
    }
}
