# Programación Genérica: El Poder de los Templates

Este módulo explora cómo C++ permite escribir código **independiente del tipo**, permitiendo que una sola función maneje `int`, `double` o `std::string` mediante el uso de plantillas.

## 📂 Archivos del Módulo

### 1. [TplValor.cpp](./TplValor.cpp)
**Concepto:** Paso de parámetros por copia.
*   **Mecanismo:** El compilador duplica los datos originales al llamar a la función.
*   **Uso ideal:** Tipos primitivos (aritmética básica) donde el coste de duplicar 4 u 8 bytes es despreciable.
*   **Referencia:** [Value Semantics en Modern C++](https://en.cppreference.com).

### 2. [TplRef.cpp](./TplRef.cpp)
**Concepto:** Paso de parámetros por referencia constante (`const T&`).
*   **Mecanismo:** La función accede a la dirección de memoria de los datos originales sin duplicarlos.
*   **Uso ideal:** Objetos pesados o clases personalizadas (como nuestra clase `Complejo` o `std::string`).
*   **Ventaja:** Máximo rendimiento y protección de datos mediante el calificador `const`.
*   **Referencia:** [Argument Passing Guidelines (C++ Core Guidelines)](https://isocpp.github.io).

---

## 🛠️ Guía de Conceptos para Ingeniería

### A. La Instanciación de Plantillas
A diferencia de otros lenguajes, en C++ los templates no existen en el binario final hasta que se usan. Cuando llamas a `maximo(10, 20)`, el compilador realiza una **instanciación**: genera una función real para `int`. Si luego llamas a `maximo(5.5, 1.2)`, genera otra para `double`. 
*   *Dato técnico:* Esto se conoce como **Polimorfismo Estático**.

### B. Requisitos del Tipo (Constraints)
Para que `maximo<T>` funcione, el tipo `T` **debe** tener definido el operador mayor que (`>`).
*   Si intentas usarlo con una clase que no lo tiene, el error ocurrirá en **tiempo de compilación**, no en ejecución. Esto garantiza la seguridad del software.
*   Más información en [Constraints and Concepts (C++20)](https://en.cppreference.com).

### C. Valor vs. Referencia: ¿Cuál elegir?
Como regla general en ingeniería de C++:
1.  **Tipos pequeños (<= 16 bytes):** Usa `TplValor.cpp`. Es más simple y a veces más rápido para el procesador.
2.  **Tipos grandes o Clases:** Usa `TplRef.cpp`. Evita el "overhead" de copiar memoria innecesariamente.

### D. Diferencia Clave (Tabla Comparativa)


| Característica | TplValor | TplRef |
| :--- | :--- | :--- |
| **Mecanismo** | Duplica el dato | Presta el dato (dirección) |
| **Uso recomendado** | `int`, `float`, `bool`, `char` | `std::string`, `std::vector`, `Complejo` |
| **Riesgo** | Lento con objetos grandes | Casi ninguno (gracias al `const`) |

---

## 🚀 Compilación
Para probar ambos ejemplos, utiliza los siguientes comandos en tu terminal:

```bash
# Compilar versión por Valor
g++ TplValor.cpp -o build/TplValor

# Compilar versión por Referencia
g++ TplRef.cpp -o build/TplRef
