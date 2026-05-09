import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {

    var window: UIWindow?

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {

        window = UIWindow(frame: UIScreen.main.bounds)

        let tab = UITabBarController()

        let tasks = UINavigationController(rootViewController: TasksViewController())
        tasks.tabBarItem.title = "Tasks"

        let categories = UINavigationController(rootViewController: CategoriesViewController())
        categories.tabBarItem.title = "Categories"

        let settings = UINavigationController(rootViewController: SettingsViewController())
        settings.tabBarItem.title = "Settings"

        tab.viewControllers = [tasks, categories, settings]

        window?.rootViewController = tab
        window?.makeKeyAndVisible()

        return true
    }
}
