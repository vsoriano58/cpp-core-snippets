# Gestión de Recursos en C++: El Riesgo del Puntero Compartido

Este laboratorio demuestra cómo la gestión manual de memoria en el **Heap** puede comprometer la estabilidad de una aplicación si no se respeta la propiedad de los objetos.

## 🔴 Fase 1: Copia Superficial (Shallow Copy)
**Archivo:** `PeligroCopiaSuperficial.cpp`

Cuando una clase gestiona punteros y no define su propia lógica de copia, C++ realiza una **copia bit a bit**.

*   **El Error:** Dos instancias (`ObjetoA` y `ObjetoB`) terminan compartiendo la misma dirección de memoria.
*   **Consecuencia:** Al salir del ámbito, ambos destructores intentan ejecutar `delete` sobre el mismo puntero.
*   **Resultado:** El programa colapsa con un **[Double Free Error](https://cwe.mitre.org)**.

## 🟢 Fase 2: Copia Profunda (Deep Copy)
**Archivo:** `PeligroCopiaSuperficial2.cpp`

La solución profesional implica implementar un **Constructor de Copia** que duplique el recurso, no el puntero.

*   **La Solución:** Se reserva un nuevo bloque de memoria en el Heap para el objeto clonado.
*   **Comportamiento:** Cada objeto es dueño de su propia dirección de memoria (`Independencia de Heap`).
*   **Resultado:** Ejecución limpia y predecible. Cada destructor libera únicamente lo que le pertenece.

## 🛠 Conclusiones de Ingeniería
1.  **Regla de los Tres:** Si gestionas memoria manualmente, *debes* definir Destructor, Constructor de Copia y Operador de Asignación.
2.  **RAII:** Los recursos deben estar ligados al ciclo de vida del objeto de forma unívoca.
3.  **Modern C++:** Para evitar esta complejidad, se recomienda el uso de **[Smart Pointers](https://en.cppreference.com)** o contenedores como `std::vector` que gestionan estas copias automáticamente.

---
*Documentación generada para el experimento de gestión de memoria de alcón68.*
