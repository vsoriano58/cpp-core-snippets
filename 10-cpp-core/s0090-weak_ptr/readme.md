# s0090: El Observador Débil (std::weak_ptr)

Este snippet resuelve el "talón de Aquiles" de la propiedad compartida: las **Referencias Circulares**. Es el complemento indispensable del `shared_ptr`.

> 📄 **Ver Código Fuente:** [weak_ptr.cpp](./weak_ptr.cpp)

---

## 🛡️ ¿Qué es un weak_ptr?
Es un puntero inteligente que **no tiene propiedad** sobre el objeto. Su función es observar a un `shared_ptr` sin intervenir en su ciclo de vida.

*   **No incrementa el contador:** Si copias un `shared_ptr` en un `weak_ptr`, el contador de referencias se mantiene igual.
*   **Seguridad de acceso:** Para usar el objeto, debes llamar a `.lock()`. Si el objeto ya fue borrado, el método devuelve un puntero nulo.

## 🔄 Rompiendo el Ciclo de la Muerte
Cuando dos objetos se apuntan mutuamente con `shared_ptr`, ninguno puede morir porque el otro lo mantiene vivo. 


| Situación | Con shared_ptr | Con weak_ptr |
| :--- | :--- | :--- |
| **Referencia A -> B** | Contador B = 1 | Contador B = 1 |
| **Referencia B -> A** | Contador A = 2 (Fuga) | **Contador A = 1 (Seguro)** |
| **Al salir de ámbito** | Memoria bloqueada | **Memoria liberada** |

## 🚀 Aplicaciones Reales
*   **Cachés de objetos:** Permite mantener una lista de objetos sin evitar que sean destruidos si nadie más los usa.
*   **Estructuras Cíclicas:** Grafos donde los nodos necesitan conocer a sus vecinos o árboles con punteros de vuelta al "padre".

---
*Snippet s0090 | El toque final para una gestión de memoria profesional e infalible.*
