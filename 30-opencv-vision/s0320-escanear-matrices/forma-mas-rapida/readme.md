# Escaneo de Matrices: El Método del Puntero C (Ultra-Rápido)

Este módulo (S0210) implementa la técnica de procesamiento de imágenes con mayor rendimiento en OpenCV. Es el método utilizado en sistemas de tiempo real y visión embebida.

---

## 🏎️ ¿Por qué es el más rápido?

A diferencia del método `.at<type>(y, x)`, que debe calcular la posición de memoria en cada iteración (multiplicaciones y sumas internas), este método obtiene la dirección de la fila una sola vez y luego se desplaza por ella mediante **offsets** directos.

### La Optimización de Continuidad (`isContinuous`)
Una de las joyas de OpenCV es la capacidad de "aplanar" la matriz. Si los datos están alineados sin huecos:
1. Colapsamos las filas en una sola (`rows = 1`).
2. El procesador aprovecha al máximo la **caché**, ya que nunca tiene que saltar de una zona de memoria a otra.

---

## ⚠️ Los Peligros del Poder

Este método es el más veloz porque **no realiza comprobación de límites**. 
*   Si tu bucle intenta leer `p[j]` donde `j` es mayor que el tamaño de la fila, el programa sufrirá un **Segmentation Fault** o corromperá datos de otras variables.
*   Es responsabilidad del programador asegurar que el cálculo de `cols * channels` sea exacto.

---

## 🛠️ Aplicación Práctica: Filtro de Brillo
En este ejemplo, reducimos el brillo de la imagen dividiendo cada canal por 2. Al operar directamente sobre el puntero `uchar*`, procesamos los canales Blue, Green y Red de forma secuencial y transparente.

```bash
# Compilación
g++ fast-scan.cpp -o fast-scan `pkg-config --cflags --libs opencv4`
./fast-scan
