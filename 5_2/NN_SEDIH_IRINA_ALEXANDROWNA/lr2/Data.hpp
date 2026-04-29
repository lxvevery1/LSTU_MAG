#include "Tensor.hpp"
#include <fstream>
#include <vector>

class Data {
    public:
        Data();

        // read big endian
        static int ReadInt(std::ifstream& f);

        static std::vector<Tensor> LoadImages(const std::string& path);
        static std::vector<int> LoadLabels(const std::string& path);
};
