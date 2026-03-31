cmake_minimum_required(VERSION 3.16)

project(CNN_Project LANGUAGES CXX)

# стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# флаги компиляции (опционально, но полезно)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# исходники
set(SOURCES
    main.cpp
    Tensor.cpp
    ConvLayer.cpp
    MaxPoolingLayer.cpp
)

# заголовки (необязательно указывать, но удобно)
set(HEADERS
    Tensor.hpp
    ConvLayer.hpp
    MaxPoolingLayer.hpp
)

# создаём исполняемый файл
add_executable(cnn ${SOURCES} ${HEADERS})
