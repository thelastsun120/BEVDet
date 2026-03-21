# RK3568 Qt 双图像显示示例

这是一个适合在 RK3568 + Qt5 Widgets 环境中运行的简单界面示例，界面包含：

- 左侧输入图像框：接收图像信号并显示原始图像。
- 右侧处理结果图像框：显示处理后的图像。
- 启动按钮：启动模拟图像流，便于直接联调界面逻辑。

## 功能说明

`ImageSource` 使用 `QTimer` 周期性发出 `frameReady(const QImage&)` 信号，`MainWindow` 收到后：

1. 通过 `sourceImageReceived` 信号刷新左侧原图显示。
2. 调用 `processImage()` 执行灰度增强处理。
3. 通过 `processedImageReady` 信号刷新右侧结果图显示。

如果你已经有摄像头、V4L2、MPP 或者算法输出，只需要把 `ImageSource` 替换成你的实际采图模块，并继续复用同样的信号/槽接口即可。

## 构建方法

```bash
mkdir -p build
cd build
cmake ../rk3568_qt_demo
make -j$(nproc)
./rk3568_qt_demo
```

## 适配真实输入源

在 RK3568 上接入真实图像时，建议：

- 采图线程只负责采集和格式转换，完成后通过信号把 `QImage` 或 `QPixmap` 发给 UI 线程。
- 如果算法处理较重，可以把 `processImage()` 挪到工作线程，再把结果信号发回主线程显示。
- 如果输入是 NV12 / YUYV，需要先转换成 Qt 可显示的 RGB 格式。
