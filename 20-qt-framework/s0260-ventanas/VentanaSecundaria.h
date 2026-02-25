#ifndef VENTANASECUNDARIA_H
#define VENTANASECUNDARIA_H

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

class VentanaSecundaria : public QWidget {
    Q_OBJECT // <--- INDISPENSABLE para signals
public:
    explicit VentanaSecundaria(QWidget *parent = nullptr) : QWidget(parent) {
        auto layout = new QVBoxLayout(this);
        input = new QLineEdit(this);
        auto btn = new QPushButton("Enviar a Principal", this);
        
        layout->addWidget(input);
        layout->addWidget(btn);

        // Capturamos 'this' para acceder a 'input'
        connect(btn, &QPushButton::clicked, this, [this]() {
            emit this->textoEnviado(this->input->text()); 
        });
    }

signals:
    void textoEnviado(const QString &texto);

private:
    QLineEdit *input;
};
#endif
