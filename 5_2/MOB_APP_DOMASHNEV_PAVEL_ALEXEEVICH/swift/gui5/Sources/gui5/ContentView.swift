import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            Text("Hello, world!")
        }
        .padding()
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
