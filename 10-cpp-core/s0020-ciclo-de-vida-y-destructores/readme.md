# Gestión de Recursos (RAII) 🛡️

### El escenario
Garantizar que un recurso crítico (memoria, archivos, sockets) se libere siempre, incluso si el programa falla o lanza una excepción.

### Objetivos
Demostrar el patrón **RAII** (*Resource Acquisition Is Initialization*), donde el ciclo de vida de un objeto en la **Pila** (Stack) gobierna la seguridad de los recursos del sistema.

### Contenido del snippet

#### [EscritorSeguro.cpp](EscritorSeguro.cpp)
- **Concepto clave:** El **Destructor** (`~`) como garantía de limpieza automática.
- **El "Desenrollado" (Stack Unwinding):** Mecanismo del compilador que recorre la pila hacia atrás destruyendo objetos y liberando sus recursos ante cualquier salida del bloque.
- **Pila vs Heap:** Contraste entre la seguridad de los objetos locales frente al riesgo de fuga (*leak*) de los punteros manuales.

