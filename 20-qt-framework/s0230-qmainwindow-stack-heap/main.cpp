#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
    
    MainWindow w; // <--- Creada en el STACK
    w.show();
    
    return a.exec();
}

/*
    NOTA TÉCNICA: GESTIÓN DE MEMORIA Y CICLO DE VIDA (Stack vs Heap)

    1) Objetos en el STACK (app y w):
       - 'app' (QApplication) controla el ciclo de vida de toda la aplicación.
       - 'w' (MainWindow) es nuestra ventana principal. Al crearla en el stack,
         garantizamos que su destructor se ejecute automáticamente al salir de main().
       - Relación: 'app.exec()' inicia el bucle de eventos; mientras este bucle corre, 
         'w' permanece viva en el stack esperando interactuar con el usuario.

    2) Objetos en el HEAP y Sistema de Parentesco (QObject Tree):
       - En mainwindow.cpp, 'lblImagen' se crea en el Heap (new QLabel). 
       - Al pasar 'this' (la dirección de 'w') al constructor del hijo, establecemos 
         una relación Padre-Hijo.
       - La magia de Qt: Cuando 'w' muere (al salir de main), su destructor recorre 
         automáticamente su lista de hijos y ejecuta un 'delete' sobre cada uno 
         (como 'lblImagen'). Esto evita "memory leaks" sin necesidad de deletes manuales.

    3) El papel de 'w': 
       Es el objeto raíz de nuestra interfaz. Si 'w' no existiera o se destruyera 
         antes de tiempo, todos sus hijos (etiquetas, botones, layouts) serían 
         eliminados en cascada por el sistema de parentesco de Qt.
*/
