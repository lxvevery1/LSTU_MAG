#if os(iOS)
import Foundation

final class AppDependencies {
    let store = SQLiteTaskStore()
    let settings = UserSettings()
    let keychain = KeychainService()
    let api = APIClient(baseURL: URL(string: "http://192.168.0.100:8080")!)
}
#endif
