cmake_minimum_required(VERSION 3.16)
project(Laboratorio VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Buscar Qt6
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Define el nombre del programa final y qué archivos lo componen
add_executable(holamundo
    holamundo.cpp
)

target_link_libraries(holamundo PRIVATE 
    Qt6::Widgets 
)
