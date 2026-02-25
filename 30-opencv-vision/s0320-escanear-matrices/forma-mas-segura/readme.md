# Escaneo de Matrices: El Método .at (Seguridad ante todo)

Este módulo (S0210-C) explora la forma más intuitiva y segura de acceder a los píxeles de una `cv::Mat`. Es el estándar de oro para el prototipado y la depuración de algoritmos de visión artificial.

---

## 🛡️ ¿Por qué es el método más seguro?

El método `.at<type>(y, x)` realiza **comprobación de límites** (en modo Debug). Si intentas acceder a un píxel fuera de la imagen (por ejemplo, la fila 600 en una imagen de 512), el programa lanzará una excepción controlada en lugar de un "Segmentation Fault" catastrófico.

### Ventajas de la Legibilidad
*   **Acceso Aleatorio**: Puedes saltar de la esquina superior izquierda a la inferior derecha sin necesidad de calcular punteros complejos.
*   **Abstracción**: No necesitas preocuparte por si la imagen es continua o si tiene canales extra; `.at` gestiona el offset interno por ti.

---

## 🐢 El Coste: La Velocidad

Este método es notablemente más lento que el escaneo por punteros. En cada llamada a `.at`, la CPU debe realizar una multiplicación y una suma para calcular la dirección de memoria exacta: 
`Dirección = Base + (fila * ancho_fila) + (columna * canales)`

Hacer este cálculo millones de veces por segundo penaliza el rendimiento en aplicaciones de tiempo real.

---

## 🎨 Aplicación Práctica: Filtro de Calidez (B=0)
En este ejemplo, eliminamos el componente **Azul** de cada píxel. Al quedar solo los canales **Rojo** y **Verde**, obtenemos una imagen con un tono amarillento/anaranjado, similar a un filtro de "luz cálida" o "modo lectura".

```bash
# Compilación
g++ safe-scan.cpp -o safe-scan `pkg-config --cflags --libs opencv4`
./mas-segura
