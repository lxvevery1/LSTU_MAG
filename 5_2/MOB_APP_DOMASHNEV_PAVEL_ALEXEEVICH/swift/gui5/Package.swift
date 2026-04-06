// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "gui5",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        // An xtool project should contain exactly one library product,
        // representing the main app.
        .library(
            name: "gui5",
            targets: ["gui5"]
        ),
    ],
    targets: [
        .target(
            name: "gui5"
        ),
    ]
)
