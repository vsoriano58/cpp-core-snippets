#ifndef GARDENVIEW_H
#define GARDENVIEW_H

#include <QWidget>
#include <QTimer>
#include <QLabel>
#include <QSlider>
#include <vector>

class GardenView : public QWidget {
    Q_OBJECT
public:
    explicit GardenView(QWidget *parent = nullptr);
    ~GardenView();

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;

private slots:
    void train();
    void updateLR(int value);
    void resetModel();

private:
    void updateGPUMemory(); // Sincroniza los vectores CPU -> GPU

    float w = 0.0f, b = 0.0f;
    float lr = 0.1f;
    int epochs = 0;
    
    std::vector<float> x, y;
    float *d_x = nullptr, *d_y = nullptr; // Punteros a la VRAM
    
    QTimer *timer;
    QLabel *statusLabel;
    QSlider *lrSlider;
};

#endif
