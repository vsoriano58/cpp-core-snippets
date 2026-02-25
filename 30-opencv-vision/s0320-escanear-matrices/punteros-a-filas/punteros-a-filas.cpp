#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 * SNIPPET S0210: Escaneo de Matriz mediante Punteros a Filas.
 * 
 * CONCEPTO: Obtenemos la dirección de inicio de cada fila una sola vez.
 * Esto evita recalcular la posición de memoria en cada píxel.
 */

int main() {
    // [REF-01] Carga de imagen (3 canales BGR por defecto)
    cv::Mat img = cv::imread("../assets/lena.jpg", cv::IMREAD_COLOR);
    if (img.empty()) return -1;

    int nRows = img.rows;
    // [REF-02] Calculamos el ancho total en bytes: columnas * canales
    int nCols = img.cols * img.channels(); 

    // [REF-03] Escaneo por filas
    for (int i = 0; i < nRows; ++i) {
        
        // [REF-04] Obtenemos el puntero al primer byte de la fila 'i'
        uchar* pFila = img.ptr<uchar>(i); 

        for (int j = 0; j < nCols; ++j) {
            // [REF-05] Acceso directo al byte (canal) mediante aritmética de punteros
            // Ejemplo: Invertir el color (negativo)
            pFila[j] = 255 - pFila[j]; 
        }
    }

    cv::imshow("Escaneo por Filas - Resultado", img);
    cv::waitKey(0);

    return 0;
}