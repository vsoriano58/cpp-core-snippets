#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 * SNIPPET S0210-C: Escaneo mediante Acceso Aleatorio (.at)
 * 
 * CONCEPTO: El método más seguro y legible. 
 * Ideal para algoritmos que no siguen un orden lineal (saltos entre píxeles).
 */

int main() {
    // [REF-01] Carga de imagen en color (BGR)
    cv::Mat img = cv::imread("../assets/lena.jpg", cv::IMREAD_COLOR);
    if (img.empty()) return -1;

    // [REF-02] Recorrido mediante coordenadas (i, j)
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            
            // [REF-03] El método .at requiere especificar el tipo de dato.
            // Para imágenes en color de 8 bits usamos cv::Vec3b (vector de 3 bytes).
            cv::Vec3b &pixel = img.at<cv::Vec3b>(i, j);

            // [REF-04] Modificación directa de los canales Blue, Green, Red
            // Ejemplo: Poner a cero el canal azul (B=0, G=1, R=2)
            pixel[0] = 0; 
            
            // También podríamos hacer:
            // img.at<cv::Vec3b>(i, j)[1] = 255; // Forzar verde al máximo
        }
    }

    cv::imshow("Escaneo con .at - Seguro pero Lento", img);
    cv::waitKey(0);

    return 0;
}