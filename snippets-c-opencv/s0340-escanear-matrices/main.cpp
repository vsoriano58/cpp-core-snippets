#include <opencv2/opencv.hpp>
#include <iostream>

/** 
 * PROGRAMA: 06_Escaneo_Matrices
 * OBJETIVO: Recorrer una imagen píxel a píxel y modificar su brillo manualmente.
 * CLAVE: Uso de .at<uchar>(y, x) para lectura y escritura directa.
 */

int main() {
    // 1. CARGAMOS UNA IMAGEN EN GRISES (1 solo canal)
    cv::Mat img = cv::imread("../lena.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) return -1;

    // 2. EL BUCLE DE ESCANEO (Doble For)
    // Recorremos cada fila (y) y cada columna (x)
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            
            // 3. LEEMOS EL VALOR ACTUAL
            // <uchar> indica que el píxel es un byte (0-255)
            uchar pixel = img.at<uchar>(y, x);

            // 4. MODIFICAMOS EL PÍXEL (Aumentamos brillo)
            // Sumamos 50, pero usamos cv::saturate_cast para no pasarnos de 255
            img.at<uchar>(y, x) = cv::saturate_cast<uchar>(pixel + 50);
        }
    }

    // 5. RESULTADO
    cv::imshow("Imagen Procesada Manualmente", img);
    cv::waitKey(0);

    return 0;
}