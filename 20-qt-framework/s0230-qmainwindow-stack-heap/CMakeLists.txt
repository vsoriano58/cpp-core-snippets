cmake_minimum_required(VERSION 3.16)
project(MiProyectoQt VERSION 1.0 LANGUAGES CXX)

# Configuración estándar de C++ para Qt
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Habilitar el procesamiento automático de Qt (MOC, UI, RCC)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Buscar el paquete de Qt (ajusta a Qt5 si no usas Qt6)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Definir el ejecutable y sus archivos fuentes
add_executable(VentanaBoton 
                mainwindow.h 
                mainwindow.cpp 
                main.cpp
)

# Vincular las librerías de Qt Widgets
target_link_libraries(VentanaBoton PRIVATE Qt6::Widgets)