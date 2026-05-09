import UIKit

final class CategoriesViewController: UICollectionViewController,
                                      UICollectionViewDelegateFlowLayout {

    let categories = ["Work", "Home", "Study", "Other"]

    init() {
        super.init(collectionViewLayout: UICollectionViewFlowLayout())
    }

    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        title = "Categories"
        collectionView.backgroundColor = .systemBackground
        collectionView.register(UICollectionViewCell.self, forCellWithReuseIdentifier: "cell")
    }

    override func collectionView(_ collectionView: UICollectionView,
                                 numberOfItemsInSection section: Int) -> Int {
        categories.count
    }

    override func collectionView(_ collectionView: UICollectionView,
                                 cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {

        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "cell",
                                                      for: indexPath)

        let label = UILabel(frame: cell.bounds)
        label.text = categories[indexPath.item]
        label.textAlignment = .center

        cell.contentView.addSubview(label)
        cell.backgroundColor = .systemGray5
        cell.layer.cornerRadius = 10

        return cell
    }

    func collectionView(_ collectionView: UICollectionView,
                        layout collectionViewLayout: UICollectionViewLayout,
                        sizeForItemAt indexPath: IndexPath) -> CGSize {
        CGSize(width: 120, height: 60)
    }
}
