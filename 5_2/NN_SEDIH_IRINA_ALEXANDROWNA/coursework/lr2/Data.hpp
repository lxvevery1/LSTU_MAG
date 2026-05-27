#include "Tensor.hpp"
#include <fstream>
#include <vector>

class Data {
    public:
        Data();

        // read big endian
        static int ReadInt(std::ifstream& f);

        // MNIST (IDX format) — kept for backward compatibility
        static std::vector<Tensor> LoadImages(const std::string& path);
        static std::vector<int> LoadLabels(const std::string& path);

        // CIFAR-10 (binary batch format)
        // Each batch file: 10000 records, each = 1 byte label + 3072 bytes (R,G,B 32x32)
        static std::vector<Tensor> LoadCIFAR10Images(const std::string& path);
        static std::vector<int> LoadCIFAR10Labels(const std::string& path);
};
