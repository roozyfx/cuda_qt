#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFileDialog>
#include <QPixmap>
#include <QSizePolicy>
#include <iostream>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setFileMenuActions();
    setSidebarActions();
    setWindowTitle("Qudapulation");
    _winSize = this->size();
    _cif = new CudaImageFuncs();

    setCudaFuncConnections();
}

MainWindow::~MainWindow()
{
    delete ui;
    if (_cif)
        delete _cif;
}

void MainWindow::openFile()
{
    _cif->openImage();

    if (_cif->image()) {
        ui->lblmage->setParent(ui->wgtMain);
        ui->lblmage->setPixmap(QPixmap::fromImage(*_cif->image()));
        ui->lblmage->setScaledContents(true);
        ui->lblmage->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
        ui->lblmage->setAlignment(Qt::AlignCenter | Qt::AlignJustify);
        ui->lblmage->show();
    }
}

void MainWindow::setFileMenuActions()
{
    // ui->actOpenImage->setShortcut(QKeySequence(tr("Ctrl+O", "Open")));
    // ui->actExit->setShortcut(QKeySequence(tr("Alt+x", "Exit")));
    connect(ui->actOpenImage, &QAction::triggered, this, &MainWindow::openFile);
    connect(ui->actExit, &QAction::triggered, this, &QApplication::quit);
}

void MainWindow::setSidebarActions()
{
    connect(ui->pbGrayScale, &QPushButton::released, this, &MainWindow::pbGrayScale);
    connect(ui->pbBlur, &QPushButton::released, this, &MainWindow::pbBlur);
    connect(ui->pbReset, &QPushButton::released, this, &MainWindow::pbReset);
}

void MainWindow::setCudaFuncConnections()
{
    if (_cif) {
        connect(this, &MainWindow::sigGrayScale, _cif, &CudaImageFuncs::grayScale);
        connect(this, &MainWindow::sigBlur, _cif, &CudaImageFuncs::blur);
        connect(this, &MainWindow::sigReset, _cif, &CudaImageFuncs::reset);

        connect(_cif, &CudaImageFuncs::sigShowResult, this, &MainWindow::showResult);
    }
}

void MainWindow::resizeEvent(QResizeEvent* event)
{
    QMainWindow::resizeEvent(event);
    // FXTODO: I've fucked up here. Make a correct implementation
    //  if (event->isAccepted()) {
    //      QSize newWinSize { this->size() };
    //      float w_r { static_cast<float>(newWinSize.width()) / _winSize.width() };
    //      float h_r { static_cast<float>(newWinSize.height()) / _winSize.height() };
    //      QSize sz { ui->wgtMain->size() };

    //     ui->wgtMain->resize(sz.width() * w_r, sz.height() * h_r);
    //     // if (_image) {
    //     //     ui->lblmage->setScaledContents(true);
    //     //     ui->lblmage->size
    //     // ui->lblmage->setPixmap(_image->scaled(ui->lblmage->size().width() * w_r,
    //     //             ui->lblmage->size().height() * h_r, Qt::KeepAspectRatio,
    //     //             Qt::TransformationMode::FastTransformation));
    //     //     ui->lblmage->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
    //     //     ui->lblmage->setAlignment(Qt::AlignCenter | Qt::AlignJustify);
    //     //     // ui->lblmage->show();
    //     //     _winSize = newWinSize;
    //     // }
    // }
}

void MainWindow::pbGrayScale()
{
    emit sigGrayScale(true);
}
void MainWindow::pbBlur()
{
    emit sigBlur(true);
}
void MainWindow::pbReset()
{
    emit sigReset(true);
}

void MainWindow::showResult()
{
    if (_cif->resultImage()) {
        std::cout << "showResult: Line 109" << std::endl;
        // ui->lblmage->setParent(ui->wgtMain);
        ui->lblmage->setPixmap(QPixmap::fromImage(*_cif->resultImage()));
        ui->lblmage->setScaledContents(true);
        ui->lblmage->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
        ui->lblmage->setAlignment(Qt::AlignCenter | Qt::AlignJustify);
        std::cout << "showResult: Line 115" << std::endl;
        ui->lblmage->show();
        std::cout << "showResult: Line 117" << std::endl;
    }
}
