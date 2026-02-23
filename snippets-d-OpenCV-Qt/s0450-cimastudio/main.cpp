/**
 * @file main.cpp
 * @brief Punto de entrada de la aplicación CimaStudio.
 * @author alcón68
 * @date 2024
 */

#include <QApplication>
#include "cimastudio.h"

/**
 * @brief Función principal que arranca el bucle de eventos de Qt.
 * @param argc Contador de argumentos de consola.
 * @param argv Vector de argumentos de consola.
 * @return Código de salida del sistema.
 */
int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    
    CimaStudio w;            // [M1] El Arranque: Instancia de la clase principal
    w.setWindowTitle("CimaStudio: Editor de Visión v1.0");
    w.resize(800, 600);
    w.show();
    
    return a.exec();         // [M2] El Bucle: Inicia la gestión de señales y eventos
}
