#pragma once
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
public slots:
    void grayScale(bool bReleased);
    void blur(bool bReleased);
    void reset(bool bReleased);

private:
    QWidget* _parent { nullptr };
    QImage* _image { nullptr };
};
