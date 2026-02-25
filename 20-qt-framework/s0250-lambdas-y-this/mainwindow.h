#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

// Usamos Forward Declaration para acelerar la compilación
class QLabel; 
class QPushButton;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

private:
    // Punteros a los widgets que manipularemos
    QLabel *etiqueta;
    QPushButton *boton;
};

#endif // MAINWINDOW_H
