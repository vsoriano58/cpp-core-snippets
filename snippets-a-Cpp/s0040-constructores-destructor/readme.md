# Ciclo de Vida: El Rastreador 🛰️

### El escenario
Visualizar la "magia negra" del compilador: ¿Cuándo nacen y mueren realmente los objetos?

### Objetivos
Identificar los tres hitos críticos de un objeto en la Pila:
1. **Nacimiento:** Constructor parametrizado.
2. **Duplicación:** El Constructor de Copia y su coste oculto (paso por valor).
3. **Fallecimiento:** El Destructor automático al cierre de llaves `}`.

### Contenido del snippet
- **[ConstructoresDestructor.cpp](ConstructoresDestructor.cpp)**: Un laboratorio con "trazas" de consola que imprimen el estado del objeto en tiempo real.
- **Lección clave:** Entender por qué la copia muere antes que el original debido al orden de limpieza de la Pila (*LIFO*).

---
[⬅️ Volver](..)
