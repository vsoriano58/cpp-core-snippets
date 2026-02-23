#include <opencv2/opencv.hpp> // Incluimos toda la potencia de OpenCV
#include <iostream>

/** 
 * PROGRAMA: 04_HolaMundo_OpenCV
 * OBJETIVO: Abrir la cámara y mostrar el flujo de video en una ventana.
 * CLAVE: El bucle while y el objeto cv::Mat (La Matriz).
 */

int main() {
    // 1. EL OJO DEL PROGRAMA
    // Creamos el objeto capturador. El '0' indica la cámara por defecto.
    cv::VideoCapture cap(0);

    // 2. EL SEGURO DE VIDA
    // Siempre debemos comprobar si el hardware ha respondido.
    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo acceder a la camara." << std::endl;
        return -1;
    }

    // 3. EL CONTENEDOR (La Sombra y el Cuerpo)
    // Reservamos la cabecera en el Stack para la matriz de píxeles.
    cv::Mat frame;

    std::cout << "Presiona cualquier tecla para salir..." << std::endl;

    // 4. EL BUCLE DE CAPTURA
    // Mientras la cámara esté abierta, "robamos" luz del sensor.
    while (true) {
        cap >> frame; // Volcamos el flujo de la cámara en la matriz 'frame'

        if (frame.empty()) break; // Si el frame viene vacío, salimos.

        // 5. LA VENTANA NATIVA
        // Mostramos la matriz en una ventana creada por OpenCV (HighGUI).
        cv::imshow("Ventana Nativa de OpenCV", frame);

        // Esperamos 30ms. Si el usuario pulsa una tecla, salimos del bucle.
        if (cv::waitKey(30) >= 0) break;
    }

    // Al salir del ámbito, 'cap' se libera automáticamente (Destructor).
    return 0;
}