# 🗺️ Hoja de Ruta: Gestión de Recursos y Memoria (RAII)

Esta secuencia documenta la progresión desde el código frágil hasta la ingeniería de alto rendimiento. Es la brújula para entender cómo se comporta el **Heap** bajo presión y cómo dominar la propiedad de los recursos.

---

### 📂 [s0045] La Solución: Constructor de Copia
**Concepto:** "Si quieres lo mío, hazte uno igual".
*   **Archivo:** `../s0045-constructor-copia/ConstructorCopia.cpp`
*   **Técnica:** Deep Copy (Copia Profunda).
*   **Misión:** Independencia total. Cada objeto reserva su propia parcela en el **Heap**.
*   **Impacto:** Seguridad absoluta a costa de rendimiento (reservar memoria es lento).
*   **Referencia:** [C++ Copy Constructors](https://en.cppreference.com)

### 📂 [s0050] El Desastre: Copia Superficial
**Concepto:** "¿Qué pasa si me olvido de gestionar el puntero?".
*   **Archivo:** `../s0050-peligro-copia-superficial/PeligroCopiaSuperficial.cpp`
*   **Técnica:** Shallow Copy (Copia por defecto).
*   **Misión:** Provocar y entender el **Double Free Error**.
*   **Impacto:** Dos objetos compartiendo el mismo recurso. El programa colapsa por propiedad ambigua.
*   **Diagnóstico:** [Double Free Vulnerability](https://cwe.mitre.org)

### 📂 [s0060] La Maestría: Semántica de Movimiento
**Concepto:** "No lo copies si puedes robarlo".
*   **Archivo:** `../s0060-constructor-movimiento/ConstructorMovimiento.cpp`
*   **Técnica:** Move Semantics (Constructor de Movimiento).
*   **Misión:** Transferencia de propiedad ultra-rápida usando referencias `&&`.
*   **Impacto:** Máximo rendimiento. Evitamos `new` y `memcpy` moviendo solo la dirección del puntero.
*   **Referencia:** [Move Constructors](https://en.cppreference.com)

---

## 🏆 El Estándar de Oro: La Regla de los Cinco

Como ingeniero, tu "checklist" para una clase profesional que gestiona memoria es esta:


| Componente | Firma Típica | Propósito |
| :--- | :--- | :--- |
| **1. Destructor** | `~Clase()` | Evitar fugas (Memory Leaks). |
| **2. Cons. Copia** | `Clase(const Clase&)` | Clonación segura (Deep Copy). |
| **3. Asig. Copia** | `operator=(const Clase&)` | Copia en objetos ya existentes. |
| **4. Cons. Movimiento** | `Clase(Clase&&)` | "Robar" recursos de temporales. |
| **5. Asig. Movimiento** | `operator=(Clase&&)` | Mover recursos en asignaciones. |

---

### 💡 Conclusión y Siguiente Parada
Hemos pasado de la **Seguridad** (s0045) al **Diagnóstico de Fallos** (s0050) y finalmente a la **Velocidad Extrema** (s0060). 

A continuación, en los bloques **s0070 (unique_ptr)** y **s0080 (shared_ptr)**, verás cómo C++ moderno automatiza toda esta lógica para que nunca más tengas que escribir un `delete` manualmente.

> *"La experiencia es lo que te permite escribir el s0045 y el s0060 de memoria para que nunca vuelvas a ver el error del s0050."*
