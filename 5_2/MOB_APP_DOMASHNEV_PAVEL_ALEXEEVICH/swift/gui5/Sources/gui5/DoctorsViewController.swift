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

        tableView.register(DoctorCell.self, forCellReuseIdentifier: "DoctorCell")
        tableView.rowHeight = 120
    }

    override func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        doctors.count
    }

    override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {

        let doctor = doctors[indexPath.row]
            let cell = tableView.dequeueReusableCell(withIdentifier: "DoctorCell", for: indexPath) as! DoctorCell

            cell.nameLabel.text = doctor.name
            cell.specLabel.text = doctor.specialization

            // Загрузка картинки через путь (как требуют)
            if let path = Bundle.module.path(forResource: doctor.imageName, ofType: "png"),
                let image = UIImage(contentsOfFile: path) {
                    cell.doctorImageView.image = image
                }

        // Кнопка "Инфо"
        cell.onInfoTapped = { [weak self] in
            let vc = DetailViewController()
                vc.doctor = doctor
                self?.navigationController?.pushViewController(vc, animated: true)
        }

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
