# s0070: El fin de la Limpieza Manual (std::unique_ptr)

Este snippet marca el paso definitivo hacia el **C++ Moderno (C++11/14)**. Implementamos el concepto de **RAII** (*Resource Acquisition Is Initialization*) para que el compilador gestione el ciclo de vida del Heap por nosotros.

> 📄 **Ver Código Fuente:** [unique_ptr.cpp](./unique_ptr.cpp)

## 🛡️ ¿Por qué es el "Smart Pointer" por excelencia?
1.  **Propiedad Única:** Garantiza que solo un puntero sea dueño del recurso en el Heap. Esto elimina por diseño el riesgo de **Double Free** (visto en `s0050`).
2.  **Cero Overhead:** En tiempo de ejecución, es tan rápido y ligero como un puntero crudo (`*`). No hay penalización de rendimiento.
3.  **Seguridad Excepcional:** Si el programa lanza una excepción, el recurso se libera automáticamente al salir del ámbito (*scope*).

## 🔄 El Movimiento como Requisito
A diferencia de los objetos normales, un `std::unique_ptr` **no se puede copiar**. Si quieres pasar la propiedad a otra variable, debes ser explícito y usar la **Semántica de Movimiento** (`std::move`).


| Operación | Permisión | Resultado |
| :--- | :--- | :--- |
| **Copia** | ❌ Prohibida | Error de compilación (Protección activa). |
| **Movimiento** | ✅ Permitida | El puntero original queda en `null` y el nuevo toma el control. |

## 🚀 Buenas Prácticas de Ingeniería
*   **Preferir `std::make_unique` (C++14):** Es más seguro y eficiente que usar `new` directamente, ya que evita fugas potenciales durante la construcción de objetos complejos.
*   **Adiós al `delete`:** Al usar Smart Pointers, el uso manual de `delete` desaparece de tu código, reduciendo drásticamente los bugs de memoria.

---
*Snippet s0070 | La herramienta fundamental para construir sistemas robustos y libres de fugas (Memory Leaks).*
