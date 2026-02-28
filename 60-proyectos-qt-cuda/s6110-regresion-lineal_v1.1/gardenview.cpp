#include "gardenview.h"
#include <QPainter>
#include <cuda_runtime.h>

extern "C" void runTrainingStep(float* d_x, float* d_y, float& w, float& b, float lr, int n);

GardenView::GardenView(QWidget *parent) : QWidget(parent) {
    setFixedSize(600, 400);
    for(int i=0; i<100; ++i) {
        x.push_back(i / 100.0f);
        y.push_back(0.7f * x.back() + 0.2f + (rand()%100/500.0f));
    }
    cudaMalloc(&d_x, x.size()*sizeof(float));
    cudaMalloc(&d_y, y.size()*sizeof(float));
    cudaMemcpy(d_x, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size()*sizeof(float), cudaMemcpyHostToDevice);

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &GardenView::train);
    timer->start(30); 
}

void GardenView::train() {
    runTrainingStep(d_x, d_y, w, b, 0.2f, (int)x.size());
    update();
}

void GardenView::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.fillRect(rect(), Qt::white);
    
    p.setPen(Qt::blue);
    for(size_t i=0; i<x.size(); ++i) {
        p.drawEllipse(QPointF(x[i]*width(), height() - y[i]*height()), 3, 3);
    }

    p.setPen(QPen(Qt::red, 3));
    p.drawLine(0, height() - b*height(), width(), height() - (w + b)*height());
}

GardenView::~GardenView() {
    cudaFree(d_x);
    cudaFree(d_y);
}
