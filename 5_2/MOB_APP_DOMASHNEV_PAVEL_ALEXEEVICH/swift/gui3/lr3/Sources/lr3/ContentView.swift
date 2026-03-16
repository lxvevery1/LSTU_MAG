import SwiftUI

struct ContentView: View {

    @State private var first = ""
    @State private var middle = ""
    @State private var last = ""
    @State private var description = ""

    var body: some View {
        VStack(spacing: 12) {

            // Top block
            HStack(alignment: .top, spacing: 12) {

                if let path = Bundle.module.path(forResource: "octopus", ofType: "png"),
                    let uiImage = UIImage(contentsOfFile: path) {

                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .frame(width: 90, height: 90)
                    }

                VStack(spacing: 20) {
                    Text("First")
                        .frame(maxWidth: 75, alignment: .leading)

                    Text("Middle")
                        .frame(maxWidth: 75, alignment: .leading)

                    Text("Last")
                        .frame(maxWidth: 75, alignment: .leading)
                }

                VStack(spacing: 8) {
                    TextField("First", text: $first)
                        .textFieldStyle(.roundedBorder)

                    TextField("Middle", text: $middle)
                        .textFieldStyle(.roundedBorder)

                    TextField("Last", text: $last)
                        .textFieldStyle(.roundedBorder)
                }
            }

            // Description
            ZStack(alignment: .topLeading) {

                TextEditor(text: $description)
                    .padding(4)

                if description.isEmpty {
                    Text("Your description...")
                        .foregroundColor(.gray)
                        .padding(.top, 12)
                        .padding(.leading, 8)
                }
            }
            .frame(maxHeight: .infinity)
            .background(Color(red: 0.83, green: 0.75, blue: 0.69))

            // Bottom buttons
            HStack(spacing: 12) {

                Button("Cancel") {
                    first = ""
                    middle = ""
                    last = ""
                    description = ""
                }
                .frame(maxWidth: .infinity)

                Button("Save") {
                    print("Saved")
                }
                .frame(maxWidth: .infinity)

                Button("Clear") {
                    description = ""
                }
                .frame(maxWidth: .infinity)
            }
            .padding(6)
            .background(Color(red: 0.7, green: 0.85, blue: 0.8))
        }
        .padding()
    }
}
