#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

int main() {
    // 打开摄像头，0 表示默认摄像头
    VideoCapture cap("/dev/video0");
    if (!cap.isOpened()) {
        cout << "无法打开摄像头" << endl;
        return -1;
    }

    // 设置摄像头参数，例如分辨率和帧率
    cap.set(CAP_PROP_FRAME_WIDTH, 320);     // 设置帧宽
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);    // 设置帧高
    cap.set(CAP_PROP_EXPOSURE, 150.0);      // 设置自动曝光（如果摄像头支持）
    cap.set(CAP_PROP_FPS, 120);             // 设置帧率（如果摄像头支持）

    // 设置视频格式为 MJPEG (四字符编码)
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // 获取设置的帧率
    double fps = cap.get(CAP_PROP_FPS);
    cout << "摄像头设置的帧率: " << fps << " FPS" << endl;

    // 计时器初始化
    int frame_count = 0;
    auto start_time = chrono::high_resolution_clock::now();

    Mat frame;

    while (true) {
        cap >> frame;  // 读取一帧
        if (frame.empty()) {
            cout << "读取帧失败" << endl;
            break;
        }

        frame_count++;

        // 显示当前帧
        imshow("Camera", frame);

        // 计算帧率
        auto elapsed_time = chrono::high_resolution_clock::now() - start_time;
        auto duration_ms = chrono::duration_cast<chrono::milliseconds>(elapsed_time).count();
        if (duration_ms >= 1000) {
            double fps_calculated = frame_count / (duration_ms / 1000.0);
            cout << "当前帧率: " << fps_calculated << " FPS" << endl;
            frame_count = 0;
            start_time = chrono::high_resolution_clock::now();
        }

        // 按 'q' 键退出
        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();  // 释放摄像头
    destroyAllWindows();  // 销毁所有窗口

    return 0;
}
