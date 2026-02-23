#include <QApplication>
#include "gardenview.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    GardenView window;
    window.setWindowTitle("Asalto a los Cielos v1.1");
    window.show();
    return app.exec();
}
