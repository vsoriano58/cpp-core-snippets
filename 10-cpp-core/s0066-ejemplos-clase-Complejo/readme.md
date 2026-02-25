# Integración: Clases Personalizadas y Templates

Este módulo demuestra el nivel más alto de reutilización de código en C++: cómo una función genérica (`template`) puede procesar objetos de una clase propia (`Complejo`) gracias a la sobrecarga de operadores.

## 📂 Archivos del Módulo

### 1. [Complejo.cpp](./Complejo.cpp)
**Concepto:** Definición de un TDA (Tipo de Dato Abstracto).
*   **Mecanismo:** Implementa la lógica interna de los números complejos, incluyendo constructores de copia/movimiento y el cálculo del módulo.
*   **Requisito:** Para ser compatible con plantillas de comparación, debe implementar el `operator>`.
*   **Referencia:** [Operator Overloading (C++ Reference)](https://en.cppreference.com).

### 2. [TplComplejo.cpp](./TplComplejo.cpp)
**Concepto:** Aplicación de Programación Genérica sobre Clases.
*   **Mecanismo:** Utiliza el template `maximo<T>` para comparar dos instancias de `Complejo`.
*   **Hito:** Demuestra que la lógica del template no necesita cambiar si el objeto cumple con la "interfaz" requerida (tener definido el operador `>`).
*   **Referencia:** [Function Templates (ISO C++)](https://isocpp.org).

---

## 🛠️ Guía de Conceptos para Ingeniería

### A. Contrato de Interfaz (Duck Typing Estático)
El template `maximo<T>` establece un **contrato**: *"Funcionaré con cualquier tipo T siempre que soporte la operación `a > b`"*. 
* Al implementar `bool operator>(const Complejo& otro)` en nuestra clase, estamos cumpliendo ese contrato. 
* Si no lo implementamos, el compilador rechazará la unión de ambos archivos con un error de "falta de coincidencia de operador".

### B. El Criterio de Comparación
En ingeniería, un número complejo no es "mayor" que otro por su parte real o imaginaria de forma aislada, sino por su **Módulo** (su magnitud en el plano complejo). 
Nuestra implementación utiliza:
```cpp
bool operator>(const Complejo& otro) const {
    return this->modulo() > otro.modulo();
}
