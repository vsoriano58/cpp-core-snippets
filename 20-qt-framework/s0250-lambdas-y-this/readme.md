# Proyecto Qt: s0250-lambdas-y-this

Este ejercicio marca la transición hacia una arquitectura profesional, separando la definición de la lógica (`.h` / `.cpp`) y gestionando la memoria mediante el puntero `this`.

---

## 🛠️ Anatomía del Proyecto

Para que este programa funcione, tres archivos trabajan en absoluta sincronía. Aquí explicamos el papel crítico de cada uno:

### 1. El Archivo de Configuración (`CMakeLists.txt`)
Es el director de orquesta. Hemos añadido dos líneas vitales para que Qt entienda nuestro código moderno:

*   **`set(CMAKE_AUTOMOC ON)`**: Activa el *Meta-Object Compiler*. Qt lee nuestro `.h`, busca la macro `Q_OBJECT` y genera automáticamente código C++ intermedio para que las señales y las lambdas funcionen. 
*   **`set(CMAKE_AUTOUIC ON)`**: Gestiona la compilación de interfaces gráficas si usáramos archivos `.ui`.

> **Nota Pro:** Gracias a esta separación, el programador solo toca sus archivos fuente; Qt se encarga de compilar el "código sucio" intermedio en segundo plano.

---

### 2. La Definición (`mainwindow.h`)
Aquí establecemos el "plano" de nuestra ventana. 

*   **Forward Declaration (`class QLabel;`)**: En lugar de incluir toda la librería en el `.h`, solo le decimos al compilador "existe una clase llamada QLabel". Esto acelera drásticamente el tiempo de compilación.
*   **La Macro `Q_OBJECT`**: Es obligatoria. Sin ella, el sistema de señales y slots (incluidas las lambdas conectadas con `this`) no funcionaría.

---

### 3. La Implementación (`mainwindow.cpp`)
Aquí es donde la "magia" de la instancia cobra sentido mediante la captura de **`this`**.

```cpp
connect(m_boton, &QPushButton::clicked, this, [this]() {
    m_etiqueta->setText("¡Logrado!");
    this->setWindowTitle("Nuevo Título");
});
```

---

* `this como contexto (3er argumento)`: Garantiza la seguridad de memoria. Si el objeto `MainWindow` se destruye, la conexión se rompe automáticamente. La `lambda` no se queda "colgando" en el vacío.
* `[this] (Captura)`: Al capturar el puntero de la instancia, la lambda tiene acceso total a los miembros privados `(m_etiqueta)` y métodos de la clase, eliminando la necesidad de declarar slots tradicionales en el `.h`.

---

## 💡Reflexión sobre la Separación .h / .cpp
Separar el código no es solo por orden, es por eficiencia:
1. **Encapsulamiento**: El main.cpp solo ve lo que necesita para arrancar la app.
2. **Compilación Incremental**: Si solo cambias la lógica de la lambda en el .cpp, el compilador no necesita re-procesar otros archivos que incluyan al .h, ahorrando tiempo en proyectos grandes.

---

## 📏 Estilo de Desarrollo: Homogeneización de Widgets

En este proyecto se ha adoptado la política de declarar todos los widgets interactivos en el archivo de cabecera (`.h`) como miembros de la clase:

1.  **Visibilidad**: Permite acceder a los widgets desde cualquier método o lambda futura sin refactorizar.
2.  **Seguridad**: Evita errores de punteros nulos al tener un control claro sobre la instanciación en el constructor.
3.  **Mantenibilidad**: Aunque la lambda capture el objeto, mantener el puntero en la clase facilita la depuración y la modificación de propiedades en tiempo de ejecución.


