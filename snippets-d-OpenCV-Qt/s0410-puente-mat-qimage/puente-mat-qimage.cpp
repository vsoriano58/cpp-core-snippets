#include <QApplication>
#include <QLabel>
#include <QPixmap>
#include <opencv2/opencv.hpp>

/** 
 * PROGRAMA: El_Puente_QImage
 * OBJETIVO: Mostrar una imagen de OpenCV dentro de un widget de Qt.
 * CLAVE: El constructor de QImage que usa el puntero .data de cv::Mat.
 */

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // 1. CARGAMOS LA IMAGEN CON OPENCV
    // OpenCV carga por defecto en formato BGR (Blue-Green-Red).
    cv::Mat matImg = cv::imread("../lena.jpg");
    if (matImg.empty()) return -1;

    // 2. EL PUENTE DE TRADUCCIÓN (Crucial)
    // Creamos una QImage que "apunta" a los mismos píxeles que la matriz.
    // Argumentos: (puntero a datos, ancho, alto, bytes por fila, formato).
    QImage qimg(matImg.data, 
                matImg.cols, 
                matImg.rows, 
                static_cast<int>(matImg.step), 
                QImage::Format_RGB888);

    // 3. LA SUTILEZA: CORRECCIÓN DE COLOR
    // Qt espera RGB, pero OpenCV entrega BGR. Debemos intercambiar los canales.
    // .rgbSwapped() nos devuelve la imagen con los colores correctos.
    QImage qimgCorrecta = qimg.rgbSwapped();

    // 4. PROYECCIÓN EN LA INTERFAZ
    // Usamos un QLabel como "pantalla". Convertimos QImage a QPixmap para pintarla.
    QLabel lienzo;
    lienzo.setPixmap(QPixmap::fromImage(qimgCorrecta));
    lienzo.setWindowTitle("Puente Qt + OpenCV");
    lienzo.show();

    return app.exec();
}