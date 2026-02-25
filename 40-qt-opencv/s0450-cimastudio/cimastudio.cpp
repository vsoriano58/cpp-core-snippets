/**
 * @file cimastudio.cpp
 * @brief Implementación de la lógica de visión y gestión de interfaz.
 */

#include "cimastudio.h"

/**
 * @brief Constructor: Configura la UI e inicia la comunicación con el hardware.
 * @param parent Puntero al widget padre.
 */
CimaStudio::CimaStudio(QWidget *parent) : QMainWindow(parent), sliderValue(100), showOriginal(false) {
    setupUI();
    
    cap.open(0);             // [C1] Despertando el Hardware: Abre la cámara por defecto
    
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &CimaStudio::updateFrame);
    timer->start(30);        // [C2] El Pulso: Dispara el procesamiento cada 30ms
}

/**
 * @brief Organiza los widgets y establece las conexiones de los controles.
 */
void CimaStudio::setupUI() {
    centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    viewLabel = new QLabel("Iniciando flujo...", this);
    viewLabel->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(viewLabel, 4);

    QHBoxLayout *controls = new QHBoxLayout();
    radioCanny = new QRadioButton("Filtro Canny", this);
    radioThreshold = new QRadioButton("Umbralización", this);
    radioCanny->setChecked(true);
    controls->addWidget(radioCanny);
    controls->addWidget(radioThreshold);
    mainLayout->addLayout(controls);

    paramSlider = new QSlider(Qt::Horizontal, this);
    paramSlider->setRange(0, 255);
    paramSlider->setValue(sliderValue);
    connect(paramSlider, &QSlider::valueChanged, [this](int v){ sliderValue = v; });
    mainLayout->addWidget(new QLabel("Intensidad del Filtro:"));
    mainLayout->addWidget(paramSlider);

    btnToggleMode = new QPushButton("Ver Imagen REAL", this);
    connect(btnToggleMode, &QPushButton::clicked, [this](){
        showOriginal = !showOriginal;
        btnToggleMode->setText(showOriginal ? "Ver Imagen PROCESADA" : "Ver Imagen REAL");
    });
    mainLayout->addWidget(btnToggleMode);
    mainLayout->addStretch();
}

/**
 * @brief Función crítica de procesamiento de imagen y actualización visual.
 */
void CimaStudio::updateFrame() {
    cv::Mat frame, processed; // [C3] Los Contenedores: Memoria en el Heap
    cap >> frame;             // [C4] La Captura: Transferencia sensor -> matriz
    
    if (frame.empty()) return; // [C5] El Seguro: Control de flujo de datos

    if (showOriginal) {
        processed = frame.clone(); // [C6] El Espejo: Duplicado completo de datos
    } else {
        // [C7] La Alquimia: Reducción de dimensionalidad (Color a Gris)
        cv::cvtColor(frame, processed, cv::COLOR_BGR2GRAY);
        
        if (radioCanny->isChecked()) {
            // [C8] El Dibujo: Detección de bordes mediante algoritmo de Canny
            cv::Canny(processed, processed, sliderValue/2, sliderValue);
        } else {
            // [C9] La Decisión: Binarización por umbral (Threshold)
            cv::threshold(processed, processed, sliderValue, 255, cv::THRESH_BINARY);
        }
    }

    // [C10] El Puente: Conversión de formato Mat a QImage para Qt
    QImage::Format format = (processed.channels() == 1) ? QImage::Format_Grayscale8 : QImage::Format_BGR888;
    QImage qimg(processed.data, processed.cols, processed.rows, processed.step, format);
    
    // [C11] La Proyección: Actualización del Pixmap en el hilo principal
    viewLabel->setPixmap(QPixmap::fromImage(qimg).scaled(viewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

/**
 * @brief Destructor: Asegura la liberación de los descriptores de la cámara.
 */
CimaStudio::~CimaStudio() { 
    cap.release(); 
}
