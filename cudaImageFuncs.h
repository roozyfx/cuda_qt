#pragma once
#include <QFileDialog>
#include <QImage>
#include <QObject>
#include <QWidget>

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
public slots:
    void grayScale(bool bReleased);
    void blur(bool bReleased);
    void reset(bool bReleased);
    void test(bool bReleased);

private:
    QWidget* _parent { nullptr };
    QImage* _image { nullptr };
    QImage* _imageResult { new QImage() };
    uchar* _inputImage_d { nullptr };
    uchar* _outputImage_d { nullptr };

signals:
    void sigShowResult();
};
