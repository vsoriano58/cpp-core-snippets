# s0045: El Constructor de Copia (Deep Copy)

Este snippet explica el mecanismo fundamental para clonar objetos que poseen recursos en el **Heap**, asegurando que cada instancia sea dueña de su propia memoria.

## 🧠 El Concepto
Cuando realizamos una asignación del tipo `Canvas copia = original;`, C++ busca el **Constructor de Copia**. Si no lo definimos, el compilador crea uno que solo copia las direcciones de los punteros (Shallow Copy), lo cual es el origen de múltiples errores críticos.

## 🛠 Implementación Técnica
La solución profesional requiere:
1. **Nueva Reserva:** Solicitar memoria independiente en el Heap para el nuevo objeto.
2. **Clonación de Datos:** Usar funciones de bajo nivel como `std::memcpy` para traspasar el contenido bit a bit.
3. **Firma Estándar:** `Clase(const Clase& otro)` para garantizar seguridad y evitar recursión.

## 🔍 Diferencias Clave

| Característica | Copia Superficial (Default) | Copia Profunda (Custom) |
| :--- | :--- | :--- |
| **Punteros** | Comparten la misma dirección | Direcciones independientes |
| **Independencia** | Si cambias uno, cambian ambos | Totalmente aislados |
| **Destrucción** | Causa [Double Free Error](https://cwe.mitre.org) | Limpieza segura y ordenada |

## 🚀 Buenas Prácticas
*   **Uso de `size_t`:** Preferible sobre `int` para tamaños de memoria, ya que garantiza valores no negativos.
*   **Referencia Constante:** Pasar el objeto origen por `const&` optimiza el rendimiento al evitar copias innecesarias durante la llamada al constructor.
*   **Abstracción:** En producción, delegar esta gestión a [std::vector](https://en.cppreference.com) simplifica drásticamente el código.

---
*Snippet s0045 | Preparación para el análisis de fallos en s0050.*
