// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "gui6",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        // An xtool project should contain exactly one library product,
        // representing the main app.
        .library(
            name: "gui6",
            targets: ["gui6"]
        ),
    ],
    targets: [
        .target(
            name: "gui6",
            resources: [
                .process("Resources")
            ]
        )
    ]
)
