#if os(iOS)
import SwiftUI
import UIKit

struct MainTabContainerView: UIViewControllerRepresentable {
    func makeUIViewController(context: Context) -> UITabBarController {
        let deps = AppDependencies()

        let tasksVC = TasksTableViewController(dependencies: deps)
        let categoriesVC = CategoriesCollectionViewController(dependencies: deps)
        let settingsVC = SettingsViewController(dependencies: deps)

        let tasksNav = UINavigationController(rootViewController: tasksVC)
        tasksNav.tabBarItem = UITabBarItem(title: "Tasks", image: UIImage(systemName: "checklist"), tag: 0)

        let categoriesNav = UINavigationController(rootViewController: categoriesVC)
        categoriesNav.tabBarItem = UITabBarItem(title: "Categories", image: UIImage(systemName: "square.grid.2x2"), tag: 1)

        let settingsNav = UINavigationController(rootViewController: settingsVC)
        settingsNav.tabBarItem = UITabBarItem(title: "Settings", image: UIImage(systemName: "gear"), tag: 2)

        let tabBar = UITabBarController()
        tabBar.viewControllers = [tasksNav, categoriesNav, settingsNav]
        return tabBar
    }

    func updateUIViewController(_ uiViewController: UITabBarController, context: Context) {}
}
#endif
