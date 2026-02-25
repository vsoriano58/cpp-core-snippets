#include <QApplication>
#include <QPushButton>
#include <QDebug> // <- Para imprimir en la consola de Qt

/** 
 * PROGRAMA: s0240-lambdas-y-slots
 * OBJETIVO: Conectar una señal (Signal) a una lógica inmediata (Slot Lambda).
 * CLAVE: Uso de la sintaxis [captura](parámetros) { código }.
 */

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QPushButton boton("Haz clic para ver la magia");
    boton.resize(300, 100);

    // Variable local que queremos modificar desde el botón
    int contador = 0;

    // 1. LA CONEXIÓN MAESTRA (Signal & Slot con Lambda)
    // Conectamos la señal 'clicked' del botón a una función anónima.
    // [ &contador ]: "Capturamos" la variable por referencia para poder sumarle.
    QObject::connect(&boton, &QPushButton::clicked, [&contador]() {
        contador++;
        qDebug() << "El boton se ha pulsado" << contador << "veces.";
    });

    boton.show();

    // 2. LA SUTILEZA TÉCNICA
    // No hemos tenido que crear una clase ni un método especial.
    // La lógica vive justo donde se necesita.
    return app.exec();
}
