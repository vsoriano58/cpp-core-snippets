#include "gardenview.h"
#include <QPainter>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <cuda_runtime.h>

extern "C" void runTrainingStep(float* d_x, float* d_y, float& w, float& b, float lr, int n);

GardenView::GardenView(QWidget *parent) : QWidget(parent) {
    setMinimumSize(800, 500);

    // 1. Generar Datos
    for(int i=0; i<150; ++i) {
        x.push_back(i / 150.0f);
        y.push_back(0.6f * x.back() + 0.3f + (rand()%100/400.0f));
    }
    cudaMalloc(&d_x, x.size()*sizeof(float));
    cudaMalloc(&d_y, y.size()*sizeof(float));
    cudaMemcpy(d_x, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size()*sizeof(float), cudaMemcpyHostToDevice);

    // 2. Layouts y Controles
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addStretch(); // Deja espacio arriba para el dibujo

    QHBoxLayout *controls = new QHBoxLayout();
    
    lrSlider = new QSlider(Qt::Horizontal);
    lrSlider->setRange(1, 100); // Representa de 0.01 a 1.0
    lrSlider->setValue(10);
    
    QPushButton *btnReset = new QPushButton("Reiniciar Pesos");
    statusLabel = new QLabel("Estado: Listo para asaltar los cielos");
    statusLabel->setStyleSheet("background: #333; color: #0f0; padding: 5px; font-family: monospace;");

    controls->addWidget(new QLabel("Learning Rate:"));
    controls->addWidget(lrSlider);
    controls->addWidget(btnReset);
    mainLayout->addLayout(controls);
    mainLayout->addWidget(statusLabel);

    // 3. Conexiones
    connect(lrSlider, &QSlider::valueChanged, this, &GardenView::updateLR);
    connect(btnReset, &QPushButton::clicked, this, &GardenView::resetModel);

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &GardenView::train);
    timer->start(30); 
}

void GardenView::updateLR(int value) {
    lr = value / 100.0f;
    statusLabel->setText(QString("LR cambiado a: %1").arg(lr));
}

void GardenView::resetModel() {
    w = 0.0f; b = 0.0f; epochs = 0;
    statusLabel->setText("Estado: Pesos reseteados (w=0, b=0)");
    update();
}

void GardenView::train() {
    runTrainingStep(d_x, d_y, w, b, lr, (int)x.size());
    epochs++;
    if(epochs % 10 == 0) {
        statusLabel->setText(QString("Época: %1 | w: %2 | b: %3 | LR: %4")
                             .arg(epochs).arg(w, 0, 'f', 4).arg(b, 0, 'f', 4).arg(lr));
    }
    update();
}

void GardenView::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    
    // Área de dibujo (un poco más pequeña para dejar sitio abajo)
    QRect canvas = rect().adjusted(20, 20, -20, -100);
    p.fillRect(canvas, Qt::white);
    p.setClipRect(canvas);

    // Puntos (escalados al canvas)
    p.setPen(Qt::blue);
    for(size_t i=0; i<x.size(); ++i) {
        float px = canvas.left() + x[i] * canvas.width();
        float py = canvas.bottom() - y[i] * canvas.height();
        p.drawEllipse(QPointF(px, py), 3, 3);
    }

    // Recta
    p.setPen(QPen(Qt::red, 3));
    float x1 = 0, y1 = b;
    float x2 = 1, y2 = w + b;
    p.drawLine(canvas.left() + x1*canvas.width(), canvas.bottom() - y1*canvas.height(),
               canvas.left() + x2*canvas.width(), canvas.bottom() - y2*canvas.height());
}

GardenView::~GardenView() {
    cudaFree(d_x); cudaFree(d_y);
}
