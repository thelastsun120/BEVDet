#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QTimer>
#include <QWidget>

class ImageDisplayWidget : public QWidget {
    Q_OBJECT

public:
    explicit ImageDisplayWidget(const QString& title, QWidget* parent = nullptr);
    void setImage(const QImage& image);

protected:
    void resizeEvent(QResizeEvent* event) override;

private:
    void refreshPixmap();

    QLabel* titleLabel_;
    QLabel* imageLabel_;
    QImage image_;
};

class ImageSource : public QObject {
    Q_OBJECT

public:
    explicit ImageSource(QObject* parent = nullptr);
    void start(int intervalMs = 100);

signals:
    void frameReady(const QImage& image);

private slots:
    void produceFrame();

private:
    QTimer timer_;
    int frameIndex_ = 0;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

signals:
    void sourceImageReceived(const QImage& image);
    void processedImageReady(const QImage& image);

private slots:
    void handleSourceFrame(const QImage& image);
    void updateSourceView(const QImage& image);
    void updateProcessedView(const QImage& image);

private:
    QImage processImage(const QImage& image) const;

    ImageDisplayWidget* sourceView_;
    ImageDisplayWidget* processedView_;
    QPushButton* startButton_;
    ImageSource* source_;
};

#endif
