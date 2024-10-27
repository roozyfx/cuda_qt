#include "cudaImageFuncs.h"
#include <QFileDialog>
#include <iostream>

CudaImageFuncs::CudaImageFuncs()
{
}

CudaImageFuncs::~CudaImageFuncs()
{
    if (_image)
        delete _image;
}

void CudaImageFuncs::openImage()
{
    QWidget* temp { new QWidget() };
    auto imageFile { QFileDialog::getOpenFileName(temp, tr("Open Image"), "/home/user/Documents/FxPhotos",
        tr("Image Files (*.png "
           "*.jpg *.jpeg *.JPG "
           "*.JPEG *.bmp)")) };
    _image = new QImage(imageFile);
    delete temp;
}

void CudaImageFuncs::grayScale(bool bReleased)
{
    if (bReleased) {
        std::cout << "Gray Scale Button\n";
    }
}

void CudaImageFuncs::blur(bool bReleased)
{
    if (bReleased) {
        std::cout << "Blur Button\n";
    }
}

void CudaImageFuncs::reset(bool bReleased)
{
    if (bReleased) {
        std::cout << "Reset Button\n";
    }
}
