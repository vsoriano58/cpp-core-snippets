cmake_minimum_required(VERSION 3.18)

# Definimos el proyecto con soporte para C++ y CUDA
project(Clasificacion_Doblar_Recta LANGUAGES CXX CUDA)

# Configuración automática de Qt
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON) 

# Buscamos Qt6 (si usas Qt5, cambia 6 por 5)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# 2. Listado de ficheros del proyecto
set(SOURCES
    main.cpp
    MainWindow.cpp
    MainWindow.h
    PointManager.h
    CudaRegressor.h
    CudaRegressor.cu
)

# 3. Creamos el ejecutable
add_executable(Clasificacion ${SOURCES})

# 4. ACTIVAMOS LA COMPILACIÓN SEPARABLE (CRÍTICO)
# Esto permite que MainWindow.cpp use los métodos definidos en CudaRegressor.cu
target_link_libraries(Clasificacion PRIVATE Qt6::Widgets)
set_target_properties(Clasificacion PROPERTIES CUDA_ARCHITECTURES 52)
