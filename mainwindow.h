#pragma once
#include "cudaImageFuncs.h"
#include <QLabel>
#include <QMainWindow>
#include <QResizeEvent>
#include <memory>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    Ui::MainWindow* getUi() { return ui; }

public slots:
    void showResult();

private slots:
    void openFile();
    void displayAbout();

private:
    void setFileMenuActions();
    void setSidebarActions();
    void setCudaFuncConnections();
    void resizeEvent(QResizeEvent* event) final;
    void pbGrayScale();
    void pbBlur();
    void pbReset();
    void pbPB1();
    void pbPB2();
    void pbTest();

    Ui::MainWindow* ui;
    QSize _winSize;
    CudaImageFuncs* _cif { nullptr };

signals:
    void sigGrayScale(bool bReleased);
    void sigBlur(bool bReleased);
    void sigReset(bool bReleased);
    void sigTest(bool bReleased);
    void sigAbout();
};
