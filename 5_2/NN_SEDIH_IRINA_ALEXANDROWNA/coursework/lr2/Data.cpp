#include "Data.hpp"
#include "Tensor.hpp"
#include <iostream>

Data::Data() { }

int Data::ReadInt(std::ifstream& f) {
    unsigned char bytes[4];
    f.read((char*)bytes, 4);
    return (bytes[0]<<24) | (bytes[1]<<16) | (bytes[2]<<8) | bytes[3];
}

// ===== MNIST (IDX format) =====

std::vector<Tensor> Data::LoadImages(const std::string& path) {
    std::ifstream f(path, std::ios::binary);

    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    int magic = ReadInt(f);
    int count = ReadInt(f);
    int rows  = ReadInt(f);
    int cols  = ReadInt(f);

    std::vector<Tensor> images;
    images.reserve(count);

    TensorSize inputSize = { 1, rows, cols};

    for (int n = 0; n < count; n++) {
        Tensor img(inputSize);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                unsigned char pixel;
                f.read((char*)&pixel, 1);
                img(0,i,j) = pixel / 255.0; // normalization
            }
        }

        images.push_back(img);
    }

    return images;
}

std::vector<int> Data::LoadLabels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);

    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    int magic = ReadInt(f);
    int count = ReadInt(f);

    std::vector<int> labels(count);

    for (int i = 0; i < count; i++) {
        unsigned char label;
        f.read((char*)&label, 1);
        labels[i] = label;
    }

    return labels;
}

// ===== CIFAR-10 (binary batch format) =====
//
// Each CIFAR-10 batch file contains 10000 records.
// Record layout: <1 byte label><3072 bytes pixel data>
//   Pixel data: 1024 Red bytes, 1024 Green bytes, 1024 Blue bytes
//   Each channel is 32×32 row-major.

std::vector<Tensor> Data::LoadCIFAR10Images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);

    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // CIFAR-10 batch format: 10000 images of 32×32×3
    const int COUNT = 10000;
    const int ROWS = 32;
    const int COLS = 32;
    const int CHANNELS = 3;
    const int IMAGE_BYTES = CHANNELS * ROWS * COLS; // 3072

    std::vector<Tensor> images;
    images.reserve(COUNT);

    TensorSize inputSize = { CHANNELS, ROWS, COLS };

    for (int n = 0; n < COUNT; n++) {
        // Read and discard label byte
        unsigned char label_byte;
        f.read((char*)&label_byte, 1);

        // Read pixel data
        std::vector<unsigned char> buf(IMAGE_BYTES);
        f.read((char*)buf.data(), IMAGE_BYTES);

        if (f.gcount() != IMAGE_BYTES) {
            break;  // partial file or end reached
        }

        Tensor img(inputSize);

        // CIFAR-10 layout: 1024 R, 1024 G, 1024 B (each 32×32 row-major)
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                int idx = i * COLS + j;
                img(0, i, j) = buf[idx]                  / 255.0;  // R
                img(1, i, j) = buf[1024 + idx]           / 255.0;  // G
                img(2, i, j) = buf[1024 + 1024 + idx]    / 255.0;  // B
            }
        }

        images.push_back(img);
    }

    return images;
}

std::vector<int> Data::LoadCIFAR10Labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);

    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    const int COUNT = 10000;
    std::vector<int> labels;
    labels.reserve(COUNT);

    for (int i = 0; i < COUNT; i++) {
        unsigned char label;
        f.read((char*)&label, 1);

        if (f.gcount() != 1) break;

        labels.push_back((int)label);

        // Skip pixel data (3072 bytes)
        f.seekg(3072, std::ios::cur);
    }

    return labels;
}
