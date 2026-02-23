#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QPushButton>
#include "PointManager.h"

// Forward declaration: le decimos que la clase existe sin meter el código todavía
class CudaRegressor; 

class MainWindow : public QMainWindow {
    Q_OBJECT
    PointManager pm;
    CudaRegressor *reg; // Usamos un puntero para evitar errores de tamaño
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow(); // Necesario para limpiar el puntero
protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
};
#endif
