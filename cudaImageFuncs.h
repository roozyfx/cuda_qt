#pragma once
#include <QFileDialog>
#include <QImage>
#include <QObject>
#include <QWidget>
#include <driver_types.h>
#include <string>

class CudaImageFuncs : public QWidget {
    Q_OBJECT
public:
    CudaImageFuncs();
    CudaImageFuncs& operator=(const CudaImageFuncs&) = delete;
    CudaImageFuncs(const CudaImageFuncs&) = delete;
    ~CudaImageFuncs();

    void openImage();
    inline QImage* image()
    {
        return _image;
    }

    inline QImage* resultImage()
    {
        return _imageResult;
    }

    inline std::vector<std::string> info()
    {
        std::vector<std::string> result;
        for (auto el : _info) {
            result.emplace_back(
                "computeMode:\t\t\t" + std::to_string(el.computeMode) + "\n" + "clockRate:\t\t\t" + std::to_string(el.clockRate) + "\n" + "concurrentKernels:\t\t" + std::to_string(el.concurrentKernels) + "\n" + "managedMemory:\t\t" + std::to_string(el.managedMemory) + "\n" + "maxBlocksPerMultiProcessor:\t" + std::to_string(el.maxBlocksPerMultiProcessor) + "\n" + "maxThreadsPerBlock:\t\t" + std::to_string(el.maxThreadsPerBlock) + "\n" + "maxThreadsDim.x:\t\t" + std::to_string(el.maxThreadsDim[0]) + "\n" + "maxThreadsDim.y:\t\t" + std::to_string(el.maxThreadsDim[1]) + "\n" + "maxThreadsDim.z:\t\t" + std::to_string(el.maxThreadsDim[2]) + "\n" + "maxGridSize.x:\t\t\t" + std::to_string(el.maxGridSize[0]) + "\n" + "maxGridSize.y:\t\t\t" + std::to_string(el.maxGridSize[1]) + "\n" + "maxGridSize.z:\t\t\t" + std::to_string(el.maxGridSize[2]) + "\n" + "maxThreadsPerMultiProcessor:\t" + std::to_string(el.maxThreadsPerMultiProcessor) + "\n" + "memoryClockRate:\t\t" + std::to_string(el.memoryClockRate) + "\n" + "multiProcessorCount:\t\t" + std::to_string(el.multiProcessorCount) + "\n" + "sharedMemPerBlock:\t\t" + std::to_string(el.sharedMemPerBlock) + "\n" + "totalConstMem:\t\t\t" + std::to_string(el.totalConstMem) + "\n" + "totalGlobalMem :\t\t\t" + std::to_string(el.totalGlobalMem) + "\n" + "warpSize :\t\t\t" + std::to_string(el.warpSize) + "\n");
        }
        return result;
    }
public slots:
    void grayScale(bool bReleased);
    void blur(bool bReleased);
    void reset(bool bReleased);
    void test(bool bReleased);

    void queryInfo();
    inline void setBlurLevel(const int bl)
    {
        _blurLevel = bl;
    }

private:
    QWidget* _parent { nullptr };
    QImage* _image { nullptr };
    QImage* _imageResult { new QImage() };
    uchar* _inputImage_d { nullptr };
    uchar* _outputImage_d { nullptr };
    int _blurLevel { 1 };

    std::vector<cudaDeviceProp> _info;

    void _loadImage(QString filename);

signals:
    void sigShowResult();
};
