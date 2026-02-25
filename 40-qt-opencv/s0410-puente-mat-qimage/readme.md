# El Puente: De cv::Mat a QImage (S0410)

Este proyecto resuelve el problema fundamental de la visión artificial profesional: **¿Cómo mostrar el procesado de OpenCV en una ventana de Qt?**

---

## 🌉 La Anatomía del Puente

La clave del éxito reside en el constructor especializado de la clase `QImage`. En lugar de crear una imagen de la nada, "enmascaramos" los datos de OpenCV:

```cpp
QImage qimg(matImg.data, width, height, step, Format_RGB888);
```

**¿Por qué es tan eficiente?** (Zero-Copy)
Al pasarle matImg.data, no duplicamos la memoria. Qt y OpenCV comparten el mismo bloque de píxeles en el Heap. Esto permite procesar vídeo a 60 FPS sin saturar la memoria RAM del sistema.

---

## 🎨El Choque de Formatos: BGR vs RGB
OpenCV nació en una época donde el estándar de hardware era BGR. Qt, como la mayoría de frameworks modernos, utiliza `RGB`.
`El Síntoma`: Sin corrección, los colores rojo y azul se intercambian.
`La Solución`: `.rgbSwapped()`. Esta función de Qt reordena los canales para que la visualización sea fiel a la realidad.


## 🛠️Estructura de Visualización
1. **cv::Mat**: El motor de datos (Matriz de píxeles).
2. **QImage**: El traductor (Entiende el formato de píxeles).
3. **QPixmap**: El proyector (Optimizado para la tarjeta gráfica).
4. **QLabel**: El lienzo (El widget que sostiene la imagen).

# Compilación (Requiere Qt y OpenCV instalados)
```bash
g++ main.cpp -o puente `pkg-config --cflags --libs opencv4 Qt5Widgets`
./puente
```

---

**Nota de Ingeniería**: Este puente es la base para crear interfaces complejas donde aplicaremos filtros de OpenCV (como Canny o Sobel) y veremos el resultado en tiempo real dentro de una ventana profesional de Qt.

