# [K] ARCHIVO DE CONFIGURACIÓN DEL PROYECTO (RECETA)
cmake_minimum_required(VERSION 3.10)
project(CimaStudio)

# Estándar de C++ moderno
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON) # Genera automáticamente el código para Signals/Slots

# Búsqueda de Dependencias Externas (Fuentes Primarias)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

# Definición del Ejecutable Final
add_executable(cimaStudio main.cpp cimastudio.h cimastudio.cpp)

# [K1] El Enlace: Une el código objeto con las librerías dinámicas
target_link_libraries(cimaStudio Qt6::Widgets ${OpenCV_LIBS})
