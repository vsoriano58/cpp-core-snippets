#include <QApplication> // El "cerebro" que gestiona la App
#include <QPushButton>  // Nuestro primer control visual

/** 
 * PROGRAMA: 01_HolaMundo_Qt
 * OBJETIVO: Crear una ventana mínima con un botón funcional.
 * FUNCIONAMIENTO: Se instancia la aplicación, se crea un widget y se entra
 * en el bucle de espera (event loop).
 */

int main(int argc, char *argv[]) {
    // 1. Inicializamos el gestor de la aplicación
    // Se encarga de captar clics, movimientos de ratón y el cierre de ventanas.
    QApplication app(argc, argv);

    // 2. Creamos un botón (un "Widget" o control visual)
    // En Qt, casi todo lo que ves hereda de QWidget.
    QPushButton boton("¡Hola Mundo desde Qt 6!");

    // 3. Configuramos su aspecto inicial
    boton.resize(300, 100);
    boton.setWindowTitle("Mi Primera Ventana");

    // 4. Mostramos el botón en pantalla
    // Por defecto, los widgets en Qt se crean "invisibles" en memoria.
    boton.show();

    // 5. Entramos en el Bucle de Eventos (Event Loop)
    // El programa se queda aquí "escuchando" hasta que cerremos la ventana.
    // return asegura que el S.O. reciba el código de salida correcto.
    return app.exec();
}