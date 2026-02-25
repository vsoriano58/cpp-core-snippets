# Escaneo de Matrices: Punteros a Filas (El Equilibrio)

Este módulo (S0210-B) presenta el método de escaneo más utilizado en la biblioteca OpenCV. Combina la **velocidad** del acceso directo a memoria con la **claridad** de la estructura por filas.

---

## ⚖️ El Punto Medio Ideal

¿Por qué elegir este método frente a los otros dos?
1.  **Frente al método `.at`**: Es significativamente más rápido porque solo calcula la posición de memoria una vez por cada fila, en lugar de hacerlo para cada píxel.
2.  **Frente al escaneo total**: Es más seguro y fácil de depurar, ya que mantenemos la noción de "fila" (`i`), facilitando algoritmos que dependen de la posición vertical (como filtros de convolución).

---

## 🖼️ Efecto Visual: El Negativo de la Imagen

En este ejemplo, aplicamos una transformación lineal simple a cada canal de color: 
`NuevoValor = 255 - ValorOriginal`.

*   **Resultado**: Los colores se invierten a sus complementarios (el azul se vuelve naranja, el verde se vuelve magenta).
*   **Rendimiento**: Gracias al acceso por punteros, esta operación se realiza casi instantáneamente incluso en imágenes de alta resolución.

---

## 🛠️ Concepto Clave: Punteros `uchar*`

Al trabajar con imágenes de 8 bits (`CV_8U`), tratamos la memoria como una secuencia de **unsigned chars**. 
*   **Importante**: Recuerda que el puntero no "ve" píxeles, ve **canales**. Si tu imagen es a color, el bucle interno (`j`) recorrerá tres veces más elementos que columnas tenga la imagen.

```bash
# Compilación
g++ row-pointers.cpp -o row-pointers `pkg-config --cflags --libs opencv4`
./punteros-a-filas
