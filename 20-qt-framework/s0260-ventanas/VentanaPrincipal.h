#ifndef VENTANAPRINCIPAL_H
#define VENTANAPRINCIPAL_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include "VentanaSecundaria.h"

class VentanaPrincipal : public QWidget {
    Q_OBJECT // <--- TE FALTA ESTA LÍNEA. Sin esto, los connects fallan.

public:
    explicit VentanaPrincipal(QWidget *parent = nullptr) : QWidget(parent) {
        this->lblStatus = new QLabel("Esperando datos...", this);
        auto btnAbrir = new QPushButton("Abrir Secundaria", this);
        
        auto layoutPrincipal = new QVBoxLayout(this);
        layoutPrincipal->addWidget(this->lblStatus);
        layoutPrincipal->addWidget(btnAbrir, 0, Qt::AlignBottom);

        connect(btnAbrir, &QPushButton::clicked, this, [this]() {
            // Pasamos 'this' como padre para que se destruya con la principal
            auto secundaria = new VentanaSecundaria(); 
            secundaria->setAttribute(Qt::WA_DeleteOnClose); // Se borra al cerrar

            // CONEXIÓN MAESTRA
            connect(secundaria, &VentanaSecundaria::textoEnviado, this, [this](const QString &t) {
               emit this->lblStatus->setText("Recibido: " + t);
            });

            secundaria->show();
        });
    }
private:
    QLabel *lblStatus;
};
#endif
