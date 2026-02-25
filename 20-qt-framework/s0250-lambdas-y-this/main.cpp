#include "mainwindow.h"
#include <QApplication>

/**
 * PROGRAMA: s0250-lambdas-y-this
 * OBJETIVO: Demostrar la captura de 'this' en lambdas dentro de una clase.
 */

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);

    // Instanciamos nuestra clase personalizada
    MainWindow w; 
    w.setWindowTitle("Captura de this en Lambda");
    w.resize(400, 200);
    w.show();

    return a.exec();
}
