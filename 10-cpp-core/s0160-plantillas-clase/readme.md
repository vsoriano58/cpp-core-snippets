# s0120 y s0121: Plantillas de Clase (Valor vs Referencia)

Este snippet conjunto analiza las dos estrategias fundamentales para construir contenedores genéricos. La elección entre pasar datos por copia o por dirección define el rendimiento y la seguridad de la memoria en aplicaciones C++.

> 📄 **Ver Código Fuente:** [TplClaseValor.cpp](./TplClaseValor.cpp) | [TplClaseRef.cpp](./TplClaseRef.cpp)

---

## 📦 El Concepto: El Molde Genérico
Las plantillas de clase permiten definir "cajas" (contenedores) cuyo tipo de dato interno se decide en el momento de la instanciación. Esto evita la duplicación de código para diferentes tipos.

1.  **Instanciación:** El compilador genera una versión específica de la clase para cada tipo solicitado (metaprogramación).
2.  **Gestión de Datos:** La clase puede ser dueña de una copia (Valor) o actuar como una interfaz de acceso eficiente (Referencia).

## ⚖️ Comparativa de Estrategias de Paso



| Estrategia | Implementación | Gestión de Memoria | Escenario Ideal |
| :--- | :--- | :--- | :--- |
| **Paso por Valor** | `T contenido` | 🟡 Copia completa (Coste extra) | Tipos básicos (`int`, `double`) |
| **Paso por Ref** | `const T& contenido` | 🟢 Sin copia (Direccionamiento) | Objetos pesados (`string`, `vector`) |

## 🛠️ Notas de Ingeniería: Const Correctness
En la versión profesional (**s0121: Referencia**), es imperativo el uso de `const` por dos razones:
*   **Seguridad:** Garantiza que el contenedor no modificará el dato original por accidente.
*   **Compatibilidad:** Permite que la clase acepte tanto objetos temporales (R-values) como constantes, haciendo el código mucho más robusto y versátil.

---
*Snippets s0120 y s0121 | La base de la eficiencia en el diseño de contenedores genéricos en C++.*
