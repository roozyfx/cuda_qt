#include "cudaImageFuncs.h"
#include <iostream>

const size_t CHANNELS = 3;

__global__ void colorToGrayScale(uchar* Pin, uchar* Pout, size_t width, size_t height)
{
    size_t row { blockIdx.y * blockDim.y + threadIdx.y };
    size_t col { blockIdx.x * blockDim.x + threadIdx.x };
    if (row < height && col < width) {
        size_t grayScaleOffset { row * width + col };
        size_t colorOffset { grayScaleOffset * CHANNELS };
        uchar r { Pin[colorOffset + 0] };
        uchar g { Pin[colorOffset + 1] };
        uchar b { Pin[colorOffset + 2] };

        Pout[grayScaleOffset] = static_cast<uchar>(r * 0.21f + g * 0.72f + b * 0.07f);
    }
}

CudaImageFuncs::CudaImageFuncs()
{
}

CudaImageFuncs::~CudaImageFuncs()
{
    std::cout << "CudaImageFuncs destructor" << std::endl;
    if (_image)
        delete _image;
    if (_imageResult)
        delete _imageResult;
    // Free the memory on device
    std::cout << "free cuda memory" << std::endl;
    cudaFree(_inputImage_d);
    cudaFree(_outputImage_d);
}

void CudaImageFuncs::openImage()
{
    QWidget* temp { new QWidget() };
    auto imageFile { QFileDialog::getOpenFileName(temp, tr("Open Image"), "/home/user/fx/cuda/img",
        tr("Image Files (*.png "
           "*.jpg *.jpeg *.JPG "
           "*.JPEG *.bmp)")) };
    _image = new QImage(imageFile);
    *_image = _image->convertToFormat(QImage::Format_RGB888);
    delete temp;
}

void CudaImageFuncs::grayScale(bool bReleased)
{
    if (bReleased) {
        std::cout << "Gray Scale Button\n";
        if (_image) {
            // Do nothing if image is already in grayscale
            // FXTODO: Make sure the format are correct, and all grayscale formats also covered.
            if (_image->format() == QImage::Format_Grayscale8 || _image->format() == QImage::Format_Grayscale16 || _image->format() == QImage::Format_Indexed8 || _image->format() == QImage::Format_Mono || _image->format() == QImage::Format_MonoLSB) {
                return;
            }
            size_t width { static_cast<size_t>(_image->width()) };
            size_t height { static_cast<size_t>(_image->height()) };
            size_t size { width * height * sizeof(uchar) };
            size_t size_color { size * CHANNELS };
            uchar* result_h = new uchar[size];

            // allocate memory on device
            cudaMalloc((void**)&_inputImage_d, size_color);
            cudaMalloc((void**)&_outputImage_d, size);

            // Copy data from host to device
            cudaMemcpy(_inputImage_d, _image->bits(), size_color, cudaMemcpyHostToDevice);

            // FXTODO: Check if correct gridDim, blockDim set
            dim3 blockDim(32, 32);
            dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
            colorToGrayScale<<<gridDim, blockDim>>>(_inputImage_d, _outputImage_d,
                width, height);

            // Copy the results to host
            cudaMemcpy(result_h, _outputImage_d, size, cudaMemcpyDeviceToHost);
            // Format_Indexed8 also works
            *_imageResult = QImage(result_h, width, height, QImage::Format_Grayscale8).copy();

            delete[] result_h;
            emit sigShowResult();
        }
    }
}

void CudaImageFuncs::blur(bool bReleased)
{
    if (bReleased) {
        std::cout << "Blur Button\n";
        if (!_imageResult) {
            _imageResult = new QImage();
        }
        *_imageResult = QImage(QString("/home/user/fx/cuda/img/Felixkula.jpeg"));
        emit sigShowResult();
    }
}

void CudaImageFuncs::reset(bool bReleased)
{
    if (bReleased) {
        std::cout << "Reset Button\n";
        if (_image) {
            *_imageResult = *_image;
            emit sigShowResult();
        }
    }
}
void CudaImageFuncs::test(bool bReleased)
{
    if (bReleased) {
        std::cout << "Test Button\n";
        _image = new QImage(QString("/home/user/fx/cuda/img/giraffe.jpg"));
        *_image = _image->convertToFormat(QImage::Format_RGB888);
    }
}
