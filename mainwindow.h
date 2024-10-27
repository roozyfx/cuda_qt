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

    // inline Ui::MainWindow* getUi() { return ui; }
private slots:
    void openFile();

private:
    void setFileMenuActions();
    void resizeEvent(QResizeEvent* event) final;
    Ui::MainWindow* ui;
    QSize _winSize;
    CudaImageFuncs* _cif { nullptr };

public slots:
    // void test(QPushButton*, void (*)(), MainWindow*, );
    void test();
};
