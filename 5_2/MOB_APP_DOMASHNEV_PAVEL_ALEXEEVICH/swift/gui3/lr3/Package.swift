// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "lr3",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        // An xtool project should contain exactly one library product,
        // representing the main app.
        .library(
            name: "lr3",
            targets: ["lr3"]
        ),
    ],
    targets: [
        .target(
            name: "lr3",
            resources: [
                .process("Resources")
            ]
        ),
    ]
)
