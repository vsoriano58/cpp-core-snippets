# Mi Proyecto en Qt: s0230-qmainwindow-stack-heap

Este es un proyecto basado en el framework **Qt** que utiliza la arquitectura de clases de C++. Este documento sirve como guía para entender la estructura del código y el flujo de ejecución.

---

## 🚀 El Corazón del Programa (main.cpp)

A menudo, las herramientas de desarrollo (IDE) generan código automáticamente que solemos ignorar. Sin embargo, es fundamental entender qué ocurre en el archivo `main.cpp`:

### La Instanciación
En el archivo `main.cpp`, ocurre el "nacimiento" de la interfaz gráfica:

```cpp
MainWindow w; // <--- Aquí ocurre la magia
```

* **MainWindow (Clase)**: Es el plano o la definición que reside en `mainwindow.h`.
* **w (Instancia)**: Es el objeto real. Es la ventana que el usuario ve y toca. Sin esta línea en el main, todo lo programado en los archivos .h y .cpp de la clase nunca llegaría a existir en memoria.

---

## Flujo de Ejecución
1. **QApplication a**: Inicializa el motor de eventos de Qt.
2. **MainWindow w**: Instancia nuestra clase principal.
3. **w.show(**): Cambia el estado de la ventana de "oculta" a "visible".
4. **a.exec()**: Inicia el bucle infinito que permite que los botones respondan a los clics.

---

## 🛠 Estructura de Archivos
* **main.cpp**: El punto de entrada. Crea la instancia w y lanza la aplicación.
* **mainwindow.h**: Define la estructura de nuestra ventana (señales, slots y variables).
* **mainwindow.cpp**: Contiene la lógica y el comportamiento de las funciones.

---

## 🏗 Requisitos y Compilación
Para ejecutar este proyecto necesitas:

* **Qt Creator** (recomendado) o el f**ramework Qt** (6.x o 5.x).
* **Compilador C++** (MSVC, GCC o Clang).
Compilación, desde el directorio raiz del proyecto:
```bash
mkdir build && cd build
cmake ..
make
```
