#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "preprocess.hpp"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    Preprocess preprocess;
    for (int i = 0; i < 8; i++)
    {
        std::string file_name = "exp" + std::to_string(i);
        std::string input_file = "./data/" + file_name + ".mp4";
        std::string method = "seg"; // 'seg', 'var', 'gray'
        if (argc > 1)
        {
            method = std::string(argv[1]);
        }
        else 
        {
            method = "gray";
        }

        std::string output_file = "./result/综合/" + file_name + "-" + method + ".mp4";

        std::cout << "Processing: " << input_file << std::endl;

        cv::VideoCapture cap(input_file);
        if (!cap.isOpened())
        {
            std::cerr << "Error opening video: " << input_file << std::endl;
            continue;
        }

        // 获取视频属性
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        cv::VideoWriter out(output_file, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps, cv::Size(frame_width * 2, frame_height));

        cv::Point current_mid_point(160, 230);
        cv::Mat frame;
        cv::Mat result;
        // cv::namedWindow("show-vedio");
        while (cap.read(frame))
        {
            cv::Mat processed_frame;
            cv::Mat processed_frame_rgb;

            if (method == "seg")
            {
                preprocess.binaryzation(frame, processed_frame, 2);
            }
            else if (method == "var")
            {
                preprocess.binaryzation(frame, processed_frame, 1);
            }
            else
            {
                preprocess.binaryzation(frame, processed_frame, 0);
            }

            // 拼接原始帧和处理后的帧
            cv::cvtColor(processed_frame, processed_frame_rgb, cv::COLOR_GRAY2RGB);
            cv::Mat combined_frame;
            cv::hconcat(frame, processed_frame_rgb, combined_frame);
            out.write(combined_frame);
        }

        std::cout << "处理完成" << std::endl;
        std::cout << "write: " << output_file << std::endl;
        cap.release();
        out.release();
    }
}