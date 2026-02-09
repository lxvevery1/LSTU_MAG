// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "package1",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        // An xtool project should contain exactly one library product,
        // representing the main app.
        .library(
            name: "package1",
            targets: ["package1"]
        ),
    ],
    targets: [
        .target(
            name: "package1"
        ),
    ]
)
