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
    // this->setCentralWidget(ui->centralwidget);
    // ui->centralwidget->setMinimumSize(0, 0);
    // ui->centralwidget->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    setWindowTitle("Qudapulation");
}

MainWindow::~MainWindow()
{
    delete ui;
    delete _image;
    delete _lbl;
}

void MainWindow::openFile()
{
    _imageFile = QFileDialog::getOpenFileName(this, tr("Open Image"), "/home/user/Documents/FxPhotos",
        tr("Image Files (*.png *.jpg *.jpeg *.JPG "
           "*.JPEG *.bmp)"));
    _image = new QPixmap(_imageFile);

    if (_image->load(_imageFile)) {
        int w { ui->centralwidget->width() };
        int h { ui->centralwidget->height() };
        _lbl->setParent(ui->centralwidget);
        _lbl->setPixmap(_image->scaled(QSize(w, h), Qt::KeepAspectRatio));
        _lbl->setScaledContents(true);
        _lbl->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
        _lbl->setAlignment(Qt::AlignCenter | Qt::AlignJustify);
        _lbl->show();
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
        QSize size { event->size() };
        // ui->centralwidget->autoFillBackground();
        // ui->centralwidget->setMinimumSize(width(), height());
        // ui->centralwidget->setSizePolicy(QSizePolicy::Expanding,
        // QSizePolicy::Expanding); ui->centralwidget->setSizePolicy(QSizePolicy::horizontalStretch(),
        // QSizePolicy::verticalStretch()); int w {
        // ui->centralwidget->width() }; int h { ui->centralwidget->height()
        // };
        _lbl->setScaledContents(true);
        _lbl->setPixmap(_image->scaled(QSize(size.width(), size.height()),
            Qt::KeepAspectRatio));
        _lbl->setAlignment(Qt::AlignCenter);
        _lbl->show();
    }
}
