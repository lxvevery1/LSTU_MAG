// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "pkg1",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        // An xtool project should contain exactly one library product,
        // representing the main app.
        .library(
            name: "pkg1",
            targets: ["pkg1"]
        ),
    ],
    targets: [
        .target(
            name: "pkg1"
        ),
    ]
)
