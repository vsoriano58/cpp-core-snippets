# Proyecto Qt: s0260-ventanas (Comunicación Inter-Ventanas)

Este ejercicio demuestra cómo conectar dos objetos independientes (`VentanaPrincipal` y `VentanaSecundaria`) mediante el sistema de **Signals & Slots**, permitiendo el paso de datos en tiempo real.

---

## 📡 El Triángulo de la Comunicación
Para que un mensaje viaje de una ventana a otra, se necesitan tres elementos que aquí hemos implementado con rigor:

1.  **La Definición (`.h`)**: Declarar la señal en la clase emisora.
    *   `signals: void textoEnviado(const QString &texto);`
2.  **El Disparador (`emit`)**: Sin esta palabra clave, la señal nunca se lanza.
    *   `emit this->textoEnviado(this->input->text());`
3.  **El Receptor (`connect`)**: En la ventana padre, "enganchamos" esa señal a una acción (Lambda).
    *   `connect(secundaria, &VentanaSecundaria::textoEnviado, this, [this](const QString &t) { ... });`

---

## 🧠 Lecciones de Ingeniería Aplicadas

### 1. Desacoplamiento (Low Coupling)
La `VentanaSecundaria` no conoce la existencia de la `Principal`. Esto es vital: si mañana queremos usar la misma ventana secundaria en un proyecto diferente, funcionará perfectamente porque ella solo "emite al mundo" sin esperar a nadie concreto.

### 2. Gestión de Memoria Dinámica
Instanciamos la secundaria con `new` para que viva de forma independiente al hilo de ejecución del botón:
*   **`Qt::WA_DeleteOnClose`**: Crucial para que Qt limpie la memoria automáticamente al cerrar la ventana hija, evitando fugas (Memory Leaks).

### 3. El Poder de la Macro `Q_OBJECT`
Hemos verificado que sin la macro `Q_OBJECT` en la cabecera de **ambas** clases, el sistema de meta-objetos de Qt no puede "enrutar" las señales, provocando fallos de compilación o conexiones silenciosas que no funcionan.

---

## 🛠️ Notas de Depuración (Troubleshooting)
*   **Señal fantasma**: Si el receptor no reacciona, comprueba que has escrito `emit` antes del nombre de la señal.
*   **Segmentation Fault**: Asegúrate de no usar `this->etiqueta` antes de haber hecho el `new etiqueta = ...` en el constructor.
*   **Librerías X11**: En Linux, recuerda instalar `libxkbcommon-x11-dev` para evitar errores de carga de plataforma.

---
*Documentación creada para el Laboratorio de C++ / Qt6*
