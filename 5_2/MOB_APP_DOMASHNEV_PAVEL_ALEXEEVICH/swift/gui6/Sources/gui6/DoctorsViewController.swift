import UIKit

final class DoctorsViewController: UIViewController {

    var doctors: [Doctor] = []

    private var collectionView: UICollectionView!

    override func viewDidLoad() {
        super.viewDidLoad()

        doctors = [
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

        let layout = UICollectionViewFlowLayout()
        layout.scrollDirection = .vertical
        layout.minimumLineSpacing = 12
        layout.minimumInteritemSpacing = 12

        collectionView = UICollectionView(frame: .zero, collectionViewLayout: layout)
        collectionView.register(DoctorCell.self, forCellWithReuseIdentifier: DoctorCell.reuseId)

        collectionView.dataSource = self
        collectionView.delegate = self

        view.addSubview(collectionView)
        collectionView.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            collectionView.topAnchor.constraint(equalTo: view.topAnchor),
            collectionView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            collectionView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            collectionView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
}

extension DoctorsViewController: UICollectionViewDataSource {


    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        doctors.count
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {

        let doctor = doctors[indexPath.item]

        let cell = collectionView.dequeueReusableCell(
            withReuseIdentifier: DoctorCell.reuseId,
            for: indexPath
        ) as! DoctorCell

        cell.configure(with: doctor)

        cell.onInfoTapped = { [weak self] in
            guard let self else { return }

            let vc = DetailViewController()
                vc.doctor = doctor
                self.navigationController?.pushViewController(vc, animated: true)
        }

        cell.onBookTapped = {
        }

        return cell
    }
}

extension DoctorsViewController: UICollectionViewDelegateFlowLayout {

    func collectionView(_ collectionView: UICollectionView,
                        layout collectionViewLayout: UICollectionViewLayout,
                        sizeForItemAt indexPath: IndexPath) -> CGSize {

        return CGSize(width: collectionView.bounds.width - 16, height: 110)
    }
}

extension DoctorsViewController: UICollectionViewDelegate {

    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        let doctor = doctors[indexPath.item]

            let vc = DetailViewController()
            vc.doctor = doctor

            navigationController?.pushViewController(vc, animated: true)
    }
}
