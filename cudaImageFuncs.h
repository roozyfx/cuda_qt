#pragma once
#include <QImage>
#include <QObject>
#include <QWidget>

// QT_BEGIN_NAMESPACE
// namespace Ui {
// class MainWindow;
// }
// QT_END_NAMESPACE

class CudaImageFuncs : public QWidget {
    Q_OBJECT
public:
    CudaImageFuncs(QWidget* parent);
    CudaImageFuncs& operator=(const CudaImageFuncs&) = delete;
    CudaImageFuncs(const CudaImageFuncs&) = delete;
    ~CudaImageFuncs();

    void openImage();
    inline QImage* image()
    {
        return _image;
    }
public slots:
    void test();

private:
    QWidget* _parent { nullptr };
    QImage* _image { nullptr };
    // Ui::MainWindow* ui;
};
