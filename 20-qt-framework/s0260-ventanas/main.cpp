#include <QApplication>
#include "VentanaPrincipal.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    VentanaPrincipal ventana;
    ventana.setWindowTitle("Ejemplo de Ventanas y Singleton");
    ventana.resize(400, 300);
    ventana.show();

    return app.exec();
}

/*
	He tenido que instalar:
		sudo apt update
		sudo apt install libxkbcommon-dev libxkbcommon-x11-dev




*/
