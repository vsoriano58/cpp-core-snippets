#include <QApplication>
#include <QPushButton>
#include <QMessageBox>
#include <QDebug>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    QPushButton boton("Lanzar Alerta Crítica");
    boton.resize(300, 100);
    boton.show();

    // Conectamos el clic a un cuadro de diálogo
    QObject::connect(&boton, &QPushButton::clicked, [&]() {
        
        // 1. CUADRO DE INFORMACIÓN (El más simple)
        QMessageBox::information(nullptr, "Aviso", "El proceso ha finalizado con éxito.");

        // 2. CUADRO DE PREGUNTA (Con respuesta del usuario)
        // Guardamos el resultado en una variable
        QMessageBox::StandardButton respuesta;
        respuesta = QMessageBox::question(nullptr, "Confirmación", 
                                        "¿Deseas formatear el disco duro?",
                                        QMessageBox::Yes | QMessageBox::No);

        if (respuesta == QMessageBox::Yes) {
            qDebug() << "El usuario aceptó el riesgo.";
            
            // 3. CUADRO DE ERROR (Icono crítico)
            QMessageBox::critical(nullptr, "Error Fatal", "No se ha encontrado el disco.");
        } else {
            qDebug() << "El usuario canceló.";
        }
    });

    return app.exec();
}
