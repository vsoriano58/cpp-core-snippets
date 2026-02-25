#include "mainwindow.h"
#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    // 1. Setup básico de la UI
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    etiqueta = new QLabel("Esperando interacción...", this);
    boton = new QPushButton("Cambiar UI desde Lambda", this);
    
    layout->addWidget(etiqueta);
    layout->addWidget(boton);

    // 2. CONEXIÓN EXPLÍCITA del THIS (Sin "magia")
    // [this]: Capturamos el puntero a la instancia actual de MainWindow.
    // Esto nos permite acceder a 'etiqueta' y a cualquier método de la clase.
    connect(boton, &QPushButton::clicked, this, [this]() {
        // Usamos 'this->' para que sea evidente que m_etiqueta 
        // pertenece a la clase y la alcanzamos gracias a la captura [this].
        this->etiqueta->setText("¡Logrado mediante el puntero de instancia!"); 
        this->setWindowTitle("Título actualizado");
    });
}

