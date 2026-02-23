cmake_minimum_required(VERSION 3.18)

# Definimos el proyecto con soporte para C++ y CUDA
project(Clasificacion_Doblar_Recta LANGUAGES CXX CUDA)

# Configuración automática de Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Buscamos Qt6 (si usas Qt5, cambia 6 por 5)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(CUDAToolkit REQUIRED)

# 1. Definimos la arquitectura de la GPU (Vital para que atomicAdd funcione en floats)
# sm_52 (Maxwell), sm_60 (Pascal), sm_75 (Turing), sm_86 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES 52 60 75 86)

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
set_target_properties(Clasificacion PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_VISIBILITY_PRESET "default"
)

# 5. Enlazamos las librerías de Qt y el Runtime de CUDA
target_link_libraries(Clasificacion 
    PRIVATE 
    Qt6::Widgets 
    CUDA::cudart
)

# 6. Flags opcionales para evitar warnings molestos
target_compile_options(Clasificacion PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
