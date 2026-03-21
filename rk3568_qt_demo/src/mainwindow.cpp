#include "mainwindow.h"

#include <QDateTime>
#include <QHBoxLayout>
#include <QPainter>
#include <QResizeEvent>
#include <QVBoxLayout>

ImageDisplayWidget::ImageDisplayWidget(const QString& title, QWidget* parent)
    : QWidget(parent), titleLabel_(new QLabel(title, this)), imageLabel_(new QLabel(this)) {
    titleLabel_->setAlignment(Qt::AlignCenter);
    titleLabel_->setStyleSheet("font-size: 18px; font-weight: 600;");

    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setMinimumSize(320, 240);
    imageLabel_->setStyleSheet(
        "background-color: #202124; color: #E8EAED; border: 2px solid #5F6368; border-radius: 8px;");
    imageLabel_->setText(QStringLiteral("等待图像输入"));

    auto* layout = new QVBoxLayout(this);
    layout->addWidget(titleLabel_);
    layout->addWidget(imageLabel_, 1);
    layout->setSpacing(12);
    layout->setContentsMargins(12, 12, 12, 12);
}

void ImageDisplayWidget::setImage(const QImage& image) {
    image_ = image;
    refreshPixmap();
}

void ImageDisplayWidget::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    refreshPixmap();
}

void ImageDisplayWidget::refreshPixmap() {
    if (image_.isNull()) {
        imageLabel_->setText(QStringLiteral("等待图像输入"));
        imageLabel_->setPixmap(QPixmap());
        return;
    }

    const auto targetSize = imageLabel_->size() - QSize(16, 16);
    const auto scaled = QPixmap::fromImage(image_).scaled(
        targetSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    imageLabel_->setPixmap(scaled);
}

ImageSource::ImageSource(QObject* parent) : QObject(parent) {
    connect(&timer_, &QTimer::timeout, this, &ImageSource::produceFrame);
}

void ImageSource::start(int intervalMs) {
    timer_.start(intervalMs);
}

void ImageSource::produceFrame() {
    constexpr int width = 640;
    constexpr int height = 360;
    QImage image(width, height, QImage::Format_RGB888);

    for (int y = 0; y < height; ++y) {
        auto* line = image.scanLine(y);
        for (int x = 0; x < width; ++x) {
            line[x * 3] = static_cast<uchar>((x + frameIndex_ * 3) % 256);
            line[x * 3 + 1] = static_cast<uchar>((y * 2 + frameIndex_ * 5) % 256);
            line[x * 3 + 2] = static_cast<uchar>((x + y + frameIndex_ * 2) % 256);
        }
    }

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(QPen(Qt::white, 3));
    painter.setFont(QFont(QStringLiteral("Sans Serif"), 18, QFont::Bold));
    painter.drawText(image.rect().adjusted(24, 24, -24, -24), Qt::AlignTop | Qt::AlignLeft,
                     QStringLiteral("Frame %1").arg(frameIndex_));
    painter.drawText(image.rect().adjusted(24, 24, -24, -24), Qt::AlignBottom | Qt::AlignRight,
                     QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss"));

    emit frameReady(image);
    ++frameIndex_;
}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent),
      sourceView_(new ImageDisplayWidget(QStringLiteral("输入图像"), this)),
      processedView_(new ImageDisplayWidget(QStringLiteral("处理结果"), this)),
      startButton_(new QPushButton(QStringLiteral("启动图像流"), this)),
      source_(new ImageSource(this)) {
    setWindowTitle(QStringLiteral("RK3568 双图像显示界面"));
    resize(1280, 520);

    auto* central = new QWidget(this);
    auto* rootLayout = new QVBoxLayout(central);
    auto* displayLayout = new QHBoxLayout();

    displayLayout->addWidget(sourceView_, 1);
    displayLayout->addWidget(processedView_, 1);
    rootLayout->addLayout(displayLayout, 1);
    rootLayout->addWidget(startButton_, 0, Qt::AlignRight);
    rootLayout->setContentsMargins(16, 16, 16, 16);
    rootLayout->setSpacing(16);

    setCentralWidget(central);

    connect(startButton_, &QPushButton::clicked, this, [this]() {
        source_->start();
        startButton_->setEnabled(false);
        startButton_->setText(QStringLiteral("图像流运行中"));
    });

    connect(source_, &ImageSource::frameReady, this, &MainWindow::handleSourceFrame);
    connect(this, &MainWindow::sourceImageReceived, this, &MainWindow::updateSourceView);
    connect(this, &MainWindow::processedImageReady, this, &MainWindow::updateProcessedView);
}

void MainWindow::handleSourceFrame(const QImage& image) {
    emit sourceImageReceived(image);
    emit processedImageReady(processImage(image));
}

void MainWindow::updateSourceView(const QImage& image) {
    sourceView_->setImage(image);
}

void MainWindow::updateProcessedView(const QImage& image) {
    processedView_->setImage(image);
}

QImage MainWindow::processImage(const QImage& image) const {
    const auto input = image.convertToFormat(QImage::Format_RGB32);
    QImage output(input.size(), QImage::Format_RGB32);

    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            const QColor pixel(input.pixel(x, y));
            const int gray = qBound(0, static_cast<int>(0.299 * pixel.red() + 0.587 * pixel.green() +
                                                         0.114 * pixel.blue()),
                                    255);
            const int enhanced = qBound(0, gray + 40, 255);
            output.setPixelColor(x, y, QColor(enhanced, enhanced, enhanced));
        }
    }

    QPainter painter(&output);
    painter.setPen(QPen(QColor(0, 255, 0), 4));
    painter.drawRect(output.rect().adjusted(20, 20, -20, -20));
    painter.drawText(output.rect().adjusted(32, 32, -32, -32), Qt::AlignTop | Qt::AlignLeft,
                     QStringLiteral("Processed Output"));
    return output;
}
