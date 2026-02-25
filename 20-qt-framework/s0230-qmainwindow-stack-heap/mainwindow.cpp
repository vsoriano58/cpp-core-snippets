#include "mainwindow.h"
#include <QMainWindow>
#include <QVBoxLayout>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    // lblImagen es un puntero definido en el .h
    // El objeto real se crea aquí en el HEAP
    lblImagen = new QLabel("Texto de la etiqueta", this); 
    lblImagen->adjustSize(); // <--- Esto ajusta el ancho y alto al texto actual


    // Reservamos espacio en el Almacén (Heap)
    QWidget *widgetCentral = new QWidget(this); 

    // El layout también vive en el Heap y su dueño es el widgetCentral
    QVBoxLayout *layout = new QVBoxLayout(widgetCentral);

    // Indicamos quien es el widget central
    setCentralWidget(widgetCentral);

    // Añadimos el lblImagen al layout
    layout->addWidget(lblImagen);
}

/* 
 * NOTA TÉCNICA SOBRE QMainWindow Y EL CENTRAL WIDGET:
 * 
 * En Qt, QMainWindow no es un contenedor simple como QWidget o QDialog; tiene una 
 * estructura interna compleja y predefinida (Layout propio) que reserva espacios 
 * específicos para la Barra de Menús, Barras de Herramientas, Dock Widgets y la 
 * Barra de Estado. 
 * 
 * 1. setCentralWidget(): Es obligatorio llamar a esta función porque QMainWindow 
 *    necesita saber qué widget debe ocupar el "corazón" de esa estructura. Si solo 
 *    creamos el layout pero no lo asignamos al Central Widget, los elementos 
 *    aparecerán amontonados o cortados en la esquina superior izquierda (0,0) 
 *    porque no están integrados en el sistema de gestión de geometría de la ventana.
 * 
 * 2. Márgenes y Layouts: A diferencia de un QDialog o un QWidget genérico (donde 
 *    puedes aplicar un layout directamente sobre 'this'), QMainWindow no permite 
 *    un layout global directo. Además, los layouts en Qt (como QVBoxLayout) aplican 
 *    por defecto unos márgenes (contentsMargins) y un espaciado (spacing). En un 
 *    QDialog, estos márgenes suelen ser más pronunciados, mientras que en el 
 *    centralWidget de una QMainWindow, son la clave para que el contenido no 
 *    toque los bordes de la ventana.
 * 
 * 3. Comportamiento frente a QDialog/QWidget: 
 *    - QWidget/QDialog: Son "lienzos en blanco". Haces setLayout(layout) y listo.
 *    - QMainWindow: Es un "marco organizado". Siempre requiere el paso intermedio 
 *      de crear un QWidget contenedor -> aplicar layout -> setCentralWidget().
 */
