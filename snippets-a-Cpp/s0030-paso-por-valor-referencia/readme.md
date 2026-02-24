# El escenario
Paso de argumentos a una función en C++.

### Objetivos
Diferenciar los dos mecanismos de gestión de memoria:
1. **Paso por valor**: La función crea una **copia local** en la pila; cualquier cambio es volátil y no afecta al origen.
2. **Paso por referencia**: La función accede a la **dirección de memoria** original; los cambios son persistentes y se evita la duplicación de datos.

La función eceptará el argumento de una u otra forma según la sintaxis empleada como veremos en los ejemplos.

### Contenido del snippet

#### imprimir (Paso por Valor)
- **Enlace:** [imprimir.cpp](imprimir.cpp)
- **Descripción:** Paso estándar. Seguridad total sobre el dato original al trabajar con una copia independiente.

#### imprimir2 (Paso por Referencia)
- **Enlace:** [imprimir2.cpp](imprimir2.cpp)
- **Descripción:** Uso de `&`. Permite modificar el origen y optimiza el rendimiento al no copiar el objeto.

---
[⬅️ Volver](..)


