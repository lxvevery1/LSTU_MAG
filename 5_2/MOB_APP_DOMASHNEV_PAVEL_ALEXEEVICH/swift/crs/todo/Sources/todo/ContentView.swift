import SwiftUI

struct AppRootView: View {
    var body: some View {
        #if os(iOS)
        MainTabContainerView()
            .ignoresSafeArea()
        #else
        Text("ToDo app is available on iOS.")
            .padding()
        #endif
    }
}
