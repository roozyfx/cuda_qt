#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFileDialog>
#include <QSizePolicy>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFileMenuActions();
    setWindowTitle("Qudapulation");
}

MainWindow::~MainWindow()
{
    delete ui;
    delete _image;
    // delete _lbl;
}

void MainWindow::openFile()
{
    _imageFile = QFileDialog::getOpenFileName(this, tr("Open Image"), "/home/user/Documents/FxPhotos",
        tr("Image Files (*.png *.jpg *.jpeg *.JPG "
           "*.JPEG *.bmp)"));
    _image = new QPixmap(_imageFile);

    if (_image->load(_imageFile)) {
        QSize sz { ui->wgtMain->size() };
        ui->lblmage->setParent(ui->wgtMain);
        ui->lblmage->setPixmap(_image->scaled(sz, Qt::KeepAspectRatio));
        ui->lblmage->setScaledContents(true);
        ui->lblmage->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
        ui->lblmage->setAlignment(Qt::AlignCenter | Qt::AlignJustify);
        ui->lblmage->show();
    }
}

void MainWindow::setFileMenuActions()
{
    ui->actionOpen_Image->setShortcut(QKeySequence(tr("Ctrl+O", "Open")));
    ui->actionExit->setShortcut(QKeySequence(tr("Alt+x", "Exit")));
    connect(ui->actionOpen_Image, &QAction::triggered, this, &MainWindow::openFile);
    connect(ui->actionExit, &QAction::triggered, this, &QApplication::quit);
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    if (event->isAccepted() && _image) {
        // ui->centralwidget->autoFillBackground();
        // ui->centralwidget->setMinimumSize(width(), height());
        // ui->centralwidget->setSizePolicy(QSizePolicy::Expanding,
        // QSizePolicy::Expanding); ui->centralwidget->setSizePolicy(QSizePolicy::horizontalStretch(),
        // QSizePolicy::verticalStretch()); int w {
        // ui->centralwidget->width() }; int h { ui->centralwidget->height()
        // };
        ui->wgtMain->setSizeIncrement(ui->centralwidget->size());
        QSize size { this->size() };
        ui->lblmage->setScaledContents(true);
        ui->lblmage->setPixmap(_image->scaled(QSize(size.width(), size.height()),
            Qt::KeepAspectRatio));
        ui->lblmage->setAlignment(Qt::AlignCenter);
        ui->lblmage->show();
    }
}
