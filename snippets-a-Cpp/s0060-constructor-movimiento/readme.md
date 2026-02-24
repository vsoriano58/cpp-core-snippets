# s0060: Constructor de Movimiento (Move Semantics)

Este snippet representa la cumbre de la optimización en C++ moderno (C++11 en adelante): la capacidad de **transferir** recursos en lugar de duplicarlos.

## ⚡ El Concepto: "Robar" en lugar de "Copiar"
Mientras que la **Copia Profunda** (s0045) duplica los datos en el Heap, el **Movimiento** simplemente transfiere la propiedad del puntero de un objeto a otro. Es una operación de coste constante **O(1)**.

*   **Origen:** Un objeto temporal (R-value) que está a punto de desaparecer.
*   **Acción:** El nuevo objeto "secuestra" el puntero del original.
*   **Seguridad:** El objeto original se pone a `nullptr` para que su destructor no borre la memoria que ahora nos pertenece.

## 🛠 Anatomía del Movimiento
1.  **Firma `&&`:** Uso de referencias R-value para detectar objetos que "van a morir".
2.  **noexcept:** Etiqueta vital para que el compilador sepa que esta transferencia no fallará y pueda optimizar contenedores como `std::vector`.
3.  **Estado Nulo:** Es obligatorio dejar al objeto origen "vacío" pero en un estado consistente.

## 📊 Comparativa de Eficiencia


| Operación | Coste CPU | Uso de Memoria | ¿Cuándo ocurre? |
| :--- | :--- | :--- | :--- |
| **Copia (Deep)** | 🔴 Alto (Reserva + Clonación) | 🔴 Duplicado | `Canvas c2 = c1;` |
| **Movimiento** | 🟢 Ultra-bajo (Asignación) | 🟢 El mismo bloque | `Canvas c2 = std::move(c1);` |

## 🚀 Conclusión de Ingeniería
El movimiento es lo que permite que C++ compita en velocidad con lenguajes de bajo nivel mientras mantiene la elegancia de la Programación Orientada a Objetos. Hemos pasado de la **Copia Peligrosa** (s0050) a la **Transferencia Inteligente**.

---
*Snippet s0060 | Culminación de la gestión de recursos bajo el estándar Modern C++.*
