#ifndef GARDENVIEW_H
#define GARDENVIEW_H

#include <QWidget>
#include <QTimer>
#include <vector>

class GardenView : public QWidget {
    Q_OBJECT
public:
    explicit GardenView(QWidget *parent = nullptr);
    ~GardenView();

protected:
    void paintEvent(QPaintEvent *event) override;

private slots:
    void train();

private:
    float w = 0.0f, b = 0.0f;
    std::vector<float> x, y;
    float *d_x, *d_y;
    QTimer *timer;
};

#endif
