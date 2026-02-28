#include "MainWindow.h"
#include "CudaRegressor.h"  // <--- ESTA ES LA CLAVE: Incluye la declaración
#include <QPainter>
#include <QMouseEvent>
#include <QPainterPath>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    reg = new CudaRegressor(); // Instanciamos el motor de CUDA
    setFixedSize(800, 600);
    setWindowTitle("CimaStudio: Clasificación Doblar Recta");

    QPushButton *btn = new QPushButton("Ajustar Curva", this);
    btn->setGeometry(10, 10, 120, 30);
    
    // Al pulsar, lanzamos 5000 iteraciones en la GPU
    connect(btn, &QPushButton::clicked, [this](){ 
        reg->train(pm, 5000); 
        update(); 
    });
}

MainWindow::~MainWindow() {
    delete reg;
}

void MainWindow::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    
    // 1. Dibujar puntos de la nube (Rojo)
    p.setPen(Qt::NoPen);
    p.setBrush(Qt::red);
    for(int i = 0; i < pm.getCount(); ++i) {
        float sx = 400 + pm.getX()[i] * 300;
        float sy = 300 - pm.getY()[i] * 250;
        p.drawEllipse(QPointF(sx, sy), 3, 3);
    }

    // --- 2. Dibujar la parábola (Aquí NO queremos relleno) ---
    p.setBrush(Qt::NoBrush); // <--- AÑADE ESTA LÍNEA PARA QUITAR LA SONRISA ROJA
    p.setPen(QPen(Qt::blue, 2)); // Solo pluma azul

    QPainterPath path;
    p.setPen(QPen(Qt::blue, 2));
    bool first = true;
    for(float x = -1.1f; x <= 1.1f; x += 0.02f) {
        // Usamos los pesos calculados por CUDA: w2, w1, b
        float y = reg->w2 * x * x + reg->w1 * x + reg->b;
        float sx = 400 + x * 300;
        float sy = 300 - y * 250;
        if(first) {
            path.moveTo(sx, sy);
            first = false;
        } else {
            path.lineTo(sx, sy);
        }
    }
    p.drawPath(path);
}

void MainWindow::mousePressEvent(QMouseEvent *e) {
    // Versión Qt6 para evitar el warning 'deprecated'
    float ia_x = (e->position().x() - 400) / 300.0f;
    float ia_y = (300 - e->position().y()) / 250.0f;
    
    pm.addPoint(ia_x, ia_y);
    reg->train(pm, 100); // Pequeño ajuste en cada clic
    update();
}
