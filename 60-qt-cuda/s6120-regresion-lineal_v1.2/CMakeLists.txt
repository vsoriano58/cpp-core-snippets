cmake_minimum_required(VERSION 3.16)
project(DeepLearning_Qt_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON) 

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable( RegresionLineal_v1.2
    main.cpp 
    gardenview.cpp
    gardenview.h
    kernel.cu
)

target_link_libraries(RegresionLineal_v1.2 PRIVATE Qt6::Widgets)
set_target_properties(RegresionLineal_v1.2 PROPERTIES CUDA_ARCHITECTURES 52)
