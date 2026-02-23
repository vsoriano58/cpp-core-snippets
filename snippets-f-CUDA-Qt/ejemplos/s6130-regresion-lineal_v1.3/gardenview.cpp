#include "gardenview.h"
#include <QPainter>
#include <QMouseEvent>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <cuda_runtime.h>

// Función externa definida en kernel.cu
extern "C" void runTrainingStep(float* d_x, float* d_y, float& w, float& b, float lr, int n);

GardenView::GardenView(QWidget *parent) : QWidget(parent) {
    setMinimumSize(800, 600);
    setWindowTitle("Asalto a los Cielos v1.3 - Laboratorio Interactivo");

    // Puntos iniciales para que la recta tenga algo que hacer al arrancar
    for(int i=0; i<10; ++i) {
        x.push_back(i / 10.0f);
        y.push_back(0.5f * x.back() + 0.2f);
    }
    updateGPUMemory();

    // Configuración de la Interfaz (UI)
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addStretch(); 

    QHBoxLayout *controls = new QHBoxLayout();
    lrSlider = new QSlider(Qt::Horizontal);
    lrSlider->setRange(1, 100); 
    lrSlider->setValue(10);
    
    QPushButton *btnReset = new QPushButton("Reiniciar Pesos");
    statusLabel = new QLabel("¡Haz clic arriba para añadir puntos y retar a la GPU!");
    statusLabel->setStyleSheet("background: #111; color: #0f0; padding: 8px; font-family: 'Courier New'; font-weight: bold;");

    controls->addWidget(new QLabel("Learning Rate:"));
    controls->addWidget(lrSlider);
    controls->addWidget(btnReset);
    
    mainLayout->addLayout(controls);
    mainLayout->addWidget(statusLabel);

    // Conexiones
    connect(lrSlider, &QSlider::valueChanged, this, &GardenView::updateLR);
    connect(btnReset, &QPushButton::clicked, this, &GardenView::resetModel);

    // Bucle de entrenamiento
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &GardenView::train);
    timer->start(30); 
}

void GardenView::updateGPUMemory() {
    // Liberamos lo anterior para evitar fugas de memoria (memory leaks)
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    
    if (x.empty()) return;

    // Reservamos y copiamos los nuevos datos a la GPU
    cudaMalloc(&d_x, x.size() * sizeof(float));
    cudaMalloc(&d_y, y.size() * sizeof(float));
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void GardenView::mousePressEvent(QMouseEvent *event) {
    // Definimos el área donde se puede dibujar (el lienzo blanco)
    QRect canvas = rect().adjusted(20, 20, -20, -120);
    
    if (canvas.contains(event->pos())) {
        // Mapeo: de píxeles de ventana a rango [0, 1]
        float newX = (float)(event->pos().x() - canvas.left()) / canvas.width();
        float newY = (float)(canvas.bottom() - event->pos().y()) / canvas.height();
        
        x.push_back(newX);
        y.push_back(newY);
        
        updateGPUMemory(); // Actualizamos la GPU con el nuevo punto
        statusLabel->setText(QString("Punto en (%1, %2) | Total: %3").arg(newX,0,'f',2).arg(newY,0,'f',2).arg(x.size()));
    }
}

void GardenView::updateLR(int value) {
    lr = value / 100.0f;
}

void GardenView::resetModel() {
    w = 0.0f; b = 0.0f; epochs = 0;
    statusLabel->setText("Sistema reseteado. Esperando nuevos datos...");
    update();
}

void GardenView::train() {
    if (x.empty()) return;

    // Llamada al Kernel de CUDA
    runTrainingStep(d_x, d_y, w, b, lr, (int)x.size());
    
    epochs++;
    if(epochs % 5 == 0) {
        statusLabel->setText(QString("Puntos: %1 | Época: %2 | w: %3 | b: %4 | LR: %5")
                             .arg(x.size()).arg(epochs).arg(w,0,'f',4).arg(b,0,'f',4).arg(lr));
    }
    update(); // Redibujar
}

void GardenView::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    
    QRect canvas = rect().adjusted(20, 20, -20, -120);
    p.fillRect(canvas, Qt::white);
    p.setClipRect(canvas);

    // Dibujar puntos azules
    p.setPen(QPen(Qt::blue, 2));
    for(size_t i=0; i<x.size(); ++i) {
        float px = canvas.left() + x[i] * canvas.width();
        float py = canvas.bottom() - y[i] * canvas.height();
        p.drawEllipse(QPointF(px, py), 4, 4);
    }

    // Dibujar recta roja de ajuste
    p.setPen(QPen(Qt::red, 4));
    float py1 = canvas.bottom() - (b * canvas.height());
    float py2 = canvas.bottom() - ((w + b) * canvas.height());
    p.drawLine(canvas.left(), py1, canvas.right(), py2);
}

GardenView::~GardenView() {
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
}
