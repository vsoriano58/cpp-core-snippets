# s0080: Propiedad Compartida (std::shared_ptr)

Este snippet explora la gestión de memoria distribuida. Es la solución ideal cuando un recurso debe ser accedido por múltiples componentes y no sabemos quién será el último en terminar de usarlo.

> 📄 **Ver Código Fuente:** [shared_ptr.cpp](./shared_ptr.cpp)

---

## 🤝 El Concepto: Conteo de Referencias
A diferencia de `unique_ptr` (dueño único), `shared_ptr` permite que existan **múltiples dueños** simultáneos. 

1.  **Registro:** Internamente mantiene un "Bloque de Control" con un contador.
2.  **Ciclo de Vida:** Cada copia aumenta el contador; cada destrucción lo disminuye.
3.  **Liberación:** Solo cuando el último puntero muere y el contador llega a **cero**, el recurso se libera en el Heap.

## ⚖️ Comparativa de Smart Pointers


| Tipo | Propiedad | Coste | Copia |
| :--- | :--- | :--- | :--- |
| `unique_ptr` | Única | 🟢 Cero (Igual a `*`) | ❌ Prohibida |
| `shared_ptr` | Compartida | 🟡 Bajo (Contador Atómico) | ✅ Permitida |

## ⚠️ Advertencia de Ingeniería: Ciclos
Si dos objetos se apuntan entre sí con `shared_ptr`, se crea un **bloqueo mutuo** de memoria (Memory Leak). Para romper estos ciclos sin perder la seguridad, se utiliza su compañero: [std::weak_ptr](https://en.cppreference.com).

---
*Snippet s0080 | La solución definitiva para recursos con tiempo de vida compartido y dinámico.*
