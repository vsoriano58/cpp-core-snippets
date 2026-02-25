# Proyecto Qt: s0270-qmessagebox (Diálogos Modales)

Este ejercicio domina el uso de ventanas de interacción rápida utilizando la clase **QMessageBox**. Estas ventanas son herramientas esenciales para la comunicación directa y crítica con el usuario.

---

## 🛑 El Concepto de "Ventana Modal"

A diferencia de las ventanas que hemos creado antes (que podían coexistir abiertas), el `QMessageBox` es **Modal** por defecto. 

### ¿Qué implica la modalidad?
1. **Bloqueo de Interfaz**: El usuario no puede interactuar con la ventana principal hasta que cierre el diálogo.
2. **Pausa en el Código**: La ejecución de la función se detiene en la línea del `QMessageBox` y solo continúa cuando el usuario pulsa un botón.

---

## 🛠️ Métodos Estáticos (Sin "new")

Para maximizar la agilidad, Qt ofrece métodos estáticos que no requieren instanciación manual (`new`). Hemos implementado los tres niveles de severidad:

*   **`QMessageBox::information()`** ℹ️: Avisos de éxito o procesos finalizados.
*   **`QMessageBox::question()`** ❓: Consultas que requieren una decisión del usuario (`Yes` / `No`).
*   **`QMessageBox::critical()`** ❌: Alertas de errores graves o fallos de sistema.

---

## 🧠 Lógica de Decisión (StandardButtons)

En este proyecto, gestionamos la respuesta del usuario capturando el valor de retorno en una variable de tipo `StandardButton`. Esto permite bifurcar la lógica del programa:

```cpp
QMessageBox::StandardButton respuesta;
respuesta = QMessageBox::question(nullptr, "Título", "¿Proceder?", 
                                 QMessageBox::Yes | QMessageBox::No);

if (respuesta == QMessageBox::Yes) {
    // Código para el camino del "SÍ"
}
```

---

## 🏗️ Estructura del Proyecto
* **main.cpp**: Contiene la lógica central. Al ser diálogos predefinidos, no necesitamos crear *  *  `archivos .h` específicos para la interfaz.
* **CMakeLists.txt**: Configurado con AUTOMOC para gestionar las conexiones de las lambdas.
## 🚀 Compilación y Ejecución
Desde el terminal en la raíz del proyecto:

```bash
mkdir build && cd build
cmake ..
make
./s0270-qmessagebox
```