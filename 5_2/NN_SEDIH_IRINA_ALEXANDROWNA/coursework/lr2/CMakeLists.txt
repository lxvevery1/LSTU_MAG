cmake_minimum_required(VERSION 3.16)

project(CNN_Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

set(SOURCES
    main.cpp
    Tensor.cpp
    Matrix.cpp
    ConvLayer.cpp
    MaxPoolingLayer.cpp
    ReLU.cpp
    FullyConnectedLayer.cpp
    CrossEntropy.cpp
    Optimizer.cpp
    Data.cpp
)

set(HEADERS
    Tensor.hpp
    Matrix.hpp
    ConvLayer.hpp
    MaxPoolingLayer.hpp
    ReLU.hpp
    FullyConnectedLayer.hpp
    CrossEntropy.hpp
    Optimizer.hpp
    Data.hpp
)

add_executable(cnn ${SOURCES} ${HEADERS})
