cmake_minimum_required(VERSION 3.16)
project(MiAppCompleta VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AUTOMOC es crítico aquí para que las señales funcionen
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Añadimos todos los archivos fuente
# Nota: En CMake moderno, no es estrictamente necesario listar los .h 
# si están en la misma carpeta, pero ayuda a la organización.
add_executable(Ventanas 
    main.cpp
    VentanaPrincipal.h
    VentanaSecundaria.h
)

target_link_libraries(Ventanas PRIVATE Qt6::Widgets)