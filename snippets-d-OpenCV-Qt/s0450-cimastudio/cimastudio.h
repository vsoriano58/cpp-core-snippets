#ifndef CIMASTUDIO_H
#define CIMASTUDIO_H

#include <QMainWindow>
#include <QTimer>
#include <QLabel>
#include <QRadioButton>
#include <QSlider>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <opencv2/opencv.hpp>

/**
 * @class CimaStudio
 * @brief Ventana principal de la aplicación de visión artificial.
 * @details Gestiona la interfaz de usuario y el flujo de captura de vídeo
 *          mediante un temporizador sincronizado.
 */
class CimaStudio : public QMainWindow {
    Q_OBJECT

public:
    /**
     * @brief Constructor de la clase CimaStudio.
     * @param parent Puntero al widget padre.
     */
    CimaStudio(QWidget *parent = nullptr);

    /**
     * @brief Destructor de la clase CimaStudio.
     * @details Libera los recursos de hardware de la cámara.
     */
    ~CimaStudio();

private slots:
    /**
     * @brief Ciclo de actualización de fotogramas.
     * @details Función invocada por el QTimer para procesar la imagen.
     */
    void updateFrame();

private:
    cv::VideoCapture cap;    // [H1] El Ojo: Motor de captura de OpenCV
    QTimer *timer;           // [H3] El Corazón: Temporizador de refresco
    int sliderValue;
    bool showOriginal;

    QWidget *centralWidget;
    QLabel *viewLabel;       // [H2] El Lienzo: Proyector de imagen en Qt
    QRadioButton *radioCanny;
    QRadioButton *radioThreshold;
    QSlider *paramSlider;
    QPushButton *btnToggleMode;

    /**
     * @brief Configura la disposición de los elementos visuales.
     */
    void setupUI();
};

#endif
