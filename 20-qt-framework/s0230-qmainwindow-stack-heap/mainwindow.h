#ifndef MAINWINDOW
#define MAINWINDOW

#include <QMainWindow>
#include <QWidget>
#include <QLabel>


class MainWindow : public QMainWindow {

public:
    MainWindow(QWidget *parent = nullptr);

private:
    // Declaramos lblImagen en el .h y lo instanciamos en el cpp
    QLabel *lblImagen;
};

#endif