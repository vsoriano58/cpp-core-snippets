# OpenCV: s0205 - Anatomía de la Matriz `cv::Mat`

Este módulo explora la estructura fundamental de OpenCV: la clase `cv::Mat`. Entender cómo gestiona la memoria es la diferencia entre un programa eficiente y uno que colapsa por falta de RAM.

---

## 🧠 La Dualidad de cv::Mat

Una `cv::Mat` se compone de dos partes:
1.  **Cabecera (Header)**: Contiene el tamaño, el método de almacenamiento, la dirección de la matriz, etc. (Tamaño constante).
2.  **Puntero de Datos**: Apunta a la matriz que contiene los valores de los píxeles. (Tamaño variable según la resolución).

### El Peligro de la Asignación `=`
En OpenCV, el operador `=` **solo copia la cabecera**. 
*   Si haces `Mat B = A;`, ambos objetos apuntan a los mismos píxeles.
*   Para duplicar realmente los datos, **debes usar `.clone()`**.

---

## 🎯 Regiones de Interés (ROI)

El uso de **ROI** es una técnica avanzada para optimizar algoritmos. En lugar de recortar y copiar una imagen, creamos una cabecera nueva que apunta a una coordenada específica de la imagen original. Cualquier filtro aplicado al ROI se reflejará en la imagen madre.

---

## 🏗️ Compilación y Uso

### Opción A: CMake (Recomendado)
```cmake
find_package(OpenCV REQUIRED)
target_link_libraries(MiPrograma PRIVATE ${OpenCV_LIBS})
```

---

### Opción B g++ (Terminal)
```bash
g++ cv-mat.cpp -o ./build/cv-mat `pkg-config --cflags --libs opencv4`
./cv-mat
```