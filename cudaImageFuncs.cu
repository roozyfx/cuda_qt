#include "cudaImageFuncs.h"
#include <cmath>
#include <iostream>
#include <stdio.h>
constexpr int CHANNELS = 3;

__global__ void colorToGrayScale_kernel(uchar* Pin, uchar* Pout, int width, int height)
{
    uint row { blockIdx.y * blockDim.y + threadIdx.y };
    uint col { blockIdx.x * blockDim.x + threadIdx.x };
    if (row < height && col < width) {
        uint grayScaleOffset { row * width + col };
        uint colorOffset { grayScaleOffset * CHANNELS };
        uchar r { Pin[colorOffset + 0] };
        uchar g { Pin[colorOffset + 1] };
        uchar b { Pin[colorOffset + 2] };

        Pout[grayScaleOffset] = static_cast<uchar>(r * 0.21f + g * 0.72f + b * 0.07f);
    }
}

__global__ void blur_kernel(uchar* Pin, uchar* Pout, int width, int height, int blurSize)
{
    int row { static_cast<int>(blockDim.y * blockIdx.y + threadIdx.y) };
    int col { static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x) };

    if (row >= height || col >= width)
        return;
    uint valueR { 0 };
    uint valueG { 0 };
    uint valueB { 0 };

    int maskSize { 0 };

    for (int r = -blurSize; r < blurSize + 1; ++r) {
        int currentRow { row + r };
        if (currentRow < 0 || currentRow >= width)
            continue;
        // Check if the current pixel is in the image
        for (int c = -blurSize; c < blurSize + 1; ++c) {
            int currentCol { col + c };
            // Check if the current pixel is in the image
            if (currentCol < 0 || currentCol >= height * CHANNELS)
                continue;
            int offset { (currentRow * width + currentCol) * CHANNELS };
            valueR += (uint)Pin[offset];
            valueG += (uint)Pin[offset + 1];
            valueB += (uint)Pin[offset + 2];
            ++maskSize;
        }
    }

    int offset { (row * width + col) * CHANNELS };

    Pout[offset + 0] = static_cast<uchar>((float)valueR / maskSize);
    Pout[offset + 1] = static_cast<uchar>((float)valueG / maskSize);
    Pout[offset + 2] = static_cast<uchar>((float)valueB / maskSize);
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
    _loadImage(imageFile);
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
            const int width { _image->width() };
            const int height { _image->height() };
            const int size { static_cast<int>(width * height * sizeof(uchar)) };
            const int size_color { size * CHANNELS };
            uchar* result_h = new uchar[size];

            // allocate memory on device
            cudaMalloc((void**)&_inputImage_d, size_color);
            cudaMalloc((void**)&_outputImage_d, size);

            // Copy data from host to device
            cudaMemcpy(_inputImage_d, _image->bits(), size_color, cudaMemcpyHostToDevice);

            dim3 blockDim(32, 32);
            dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
            colorToGrayScale_kernel<<<gridDim, blockDim>>>(_inputImage_d, _outputImage_d,
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
        if (_image) {
            const int width { _image->width() };
            const int height { _image->height() };
            const int size { static_cast<int>(width * height * CHANNELS * sizeof(uchar)) };
            std::cout << "Width x Height:" << width << "x" << height << std::endl;
            std::cout << "size: " << size << std::endl;

            uchar* result_h = new uchar[size];

            // allocate memory on device
            cudaMalloc((void**)&_inputImage_d, size);
            cudaMalloc((void**)&_outputImage_d, size);

            // Copy data from host to device
            cudaMemcpy(_inputImage_d, _image->bits(), size, cudaMemcpyHostToDevice);

            dim3 blockDim(32, 32);
            dim3 gridDim(std::ceil((float)(width) / blockDim.x),
                std::ceil((float)(height) / blockDim.y));
            blur_kernel<<<gridDim, blockDim>>>(_inputImage_d, _outputImage_d,
                width, height, _blurLevel);

            // Copy the results to host
            cudaMemcpy(result_h, _outputImage_d, size, cudaMemcpyDeviceToHost);

            std::cout << "_image format: " << _image->format() << std::endl;
            *_imageResult = QImage(result_h, width, height, QImage::Format_RGB888).copy();

            delete[] result_h;
            emit sigShowResult();
        }
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
        _loadImage(QString("/home/user/fx/cuda/img/giraffe.jpg"));
    }
}

void CudaImageFuncs::queryInfo()
{
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    for (int i = 0; i < numDevices; ++i) {
        cudaDeviceProp cdp;
        cudaGetDeviceProperties(&cdp, i);
        _info.emplace_back(cdp);
    }
}

void CudaImageFuncs::_loadImage(QString filename)
{

    _image = new QImage(filename);
    *_image = (_image->scaled(_image->width() - (_image->width() % 4), _image->height()));

    std::cout << "Format is: " << _image->format() << std::endl;
    *_image = _image->convertToFormat(QImage::Format_RGB888);
}
