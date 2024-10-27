#include "cudaImageFuncs.h"
// #include "./ui_mainwindow.h"
#include <QFileDialog>
#include <iostream>

CudaImageFuncs::CudaImageFuncs(QWidget* parent)
// : ui(new Ui::MainWindow)
{
    _parent = parent;
    this->setParent(_parent);

    // connect(ui->pbGrayScale, &QPushButton::released, this, &CudaImageFuncs::test);
}

CudaImageFuncs::~CudaImageFuncs()
{
    if (_image)
        delete _image;
}

void CudaImageFuncs::openImage()
{
    auto imageFile { QFileDialog::getOpenFileName(_parent, tr("Open Image"), "/home/user/Documents/FxPhotos",
        tr("Image Files (*.png "
           "*.jpg *.jpeg *.JPG "
           "*.JPEG *.bmp)")) };
    _image = new QImage(imageFile);
}

void CudaImageFuncs::test()
{
    std::cout << "Gray Scale Button\n";
}
