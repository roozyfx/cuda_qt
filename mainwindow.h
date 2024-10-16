#pragma once
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QResizeEvent>
#include <QString>
#include <QTimer>
#include <memory>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

static const int RESIZE_TIMEOUT = 250; // 1/4 second in milliseconds

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    void openFile();

private:
    void setFileMenuActions();
    void resizeEvent(QResizeEvent* event) final;
    Ui::MainWindow* ui;
    QString _imageFile { "" };
    QPixmap* _image { nullptr };
    // QLabel* _lbl { nullptr };
};
