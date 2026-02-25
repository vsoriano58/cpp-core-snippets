# Proyecto Qt: s0240-lambdas-y-slots

Este ejercicio demuestra cómo simplificar la interacción entre la interfaz y la lógica utilizando **Funciones Lambda** de C++ en lugar de los Slots tradicionales.

---

## ⚡ El Poder de las Lambdas en Qt

En versiones antiguas de Qt, cualquier acción de un botón requería declarar un método en el `.h` y desarrollarlo en el `.cpp`. Con las lambdas, la lógica vive exactamente donde se necesita.

### Anatomía de la Conexión
```cpp
QObject::connect(&boton, &QPushButton::clicked, [&contador]() {
    contador++;
    qDebug() << "Clicks:" << contador;
});
```

1. [`&contador`] (Captura): Permite que la función anónima acceda a la variable local contador. Al usar `&`, capturamos por referencia, permitiendo modificar el valor original.
2. `() (Parámetros)`: Aquí irían los argumentos que envía la señal (en clicked no hay ninguno, pero si fuera un slider, actualizaría el parámetro durante todo el recorrido del mismo).
3. `{ ... } (Cuerpo)`: El código que se ejecuta al pulsar el botón.

## ⚠️ La Sutileza Técnica: El Contexto de Vida 
Un error común al usar lambdas es capturar variables por referencia que podrían ser destruidas antes de que el usuario pulse el botón. Esto causaría un crash (puntero colgante).

### La Solución: El 4º Argumento (Contexto)
Aunque en este ejemplo simple el main protege las variables, en aplicaciones reales debemos indicar un `objeto de contexto`:

```cpp
// Forma ultra-segura:
QObject::connect(&boton, &QPushButton::clicked, contextObject, [&contador]() {
    // Esta lambda solo se ejecutará si 'contextObject' sigue vivo.
});
```
**¿Por qué es importante?**
Si el contextObject (por ejemplo, la ventana principal) se destruye, Qt desconecta automáticamente la lambda, evitando que intente acceder a memoria que ya ha sido liberada.

---

## 🛠 Aprendizajes clave
* **Agilidad**: Menos código repetitivo (boilerplate) en los archivos .h.
* **Ámbito (Scope)**: Las variables locales del main son accesibles mediante captura.
* **qDebug()**: Uso de la consola de depuración de Qt para trazabilidad inmediata.

## Compilación rápida
```bash
mkdir build && cd build
cmake ..
make
./MiLambda
```