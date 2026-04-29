#include "Data.hpp"
#include "Tensor.hpp"
#include <iostream>

Data::Data() { }

int Data::ReadInt(std::ifstream& f) {
    unsigned char bytes[4];
    f.read((char*)bytes, 4);
    return (bytes[0]<<24) | (bytes[1]<<16) | (bytes[2]<<8) | bytes[3];
}

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
