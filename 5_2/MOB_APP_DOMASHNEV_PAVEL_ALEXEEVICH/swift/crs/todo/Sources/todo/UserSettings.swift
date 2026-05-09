#if os(iOS)
import Foundation

final class UserSettings {
    private enum Keys {
        static let displayName = "displayName"
        static let selectedTheme = "selectedTheme"
    }

    var displayName: String {
        get { UserDefaults.standard.string(forKey: Keys.displayName) ?? "Student" }
        set { UserDefaults.standard.set(newValue, forKey: Keys.displayName) }
    }

    var selectedTheme: String {
        get { UserDefaults.standard.string(forKey: Keys.selectedTheme) ?? "system" }
        set { UserDefaults.standard.set(newValue, forKey: Keys.selectedTheme) }
    }
}
#endif
