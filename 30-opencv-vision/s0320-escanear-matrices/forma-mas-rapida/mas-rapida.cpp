#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 * SNIPPET S0210: Escaneo Ultra-Rápido de Matrices.
 * MÉTODO: Punteros C + Verificación de Continuidad.
 */

int main() {
    // [REF-01] Carga de imagen
    cv::Mat img = cv::imread("../assets/lena.jpg", cv::IMREAD_COLOR);
    if (img.empty()) return -1;

    // [REF-02] Preparación de variables
    int rows = img.rows;
    int cols = img.cols * img.channels(); // Total de elementos por fila

    // [REF-03] Optimización por Continuidad
    // Si la matriz no tiene "huecos" al final de cada fila, 
    // la convertimos en una sola fila larguísima.
    if (img.isContinuous()) {
        cols *= rows;
        rows = 1; 
    }

    // [REF-04] Bucle de alto rendimiento
    for (int i = 0; i < rows; ++i) {
        // Obtenemos el puntero al inicio de la fila i
        uchar* p = img.ptr<uchar>(i);
        
        for (int j = 0; j < cols; ++j) {
            // Ejemplo: Reducción de brillo dividiendo por 2
            // Acceso directo por índice de puntero (lo más rápido)
            p[j] = p[j] / 2; 
        }
    }

    cv::imshow("Imagen Procesada Rapida", img);
    cv::waitKey(0);

    return 0;
}