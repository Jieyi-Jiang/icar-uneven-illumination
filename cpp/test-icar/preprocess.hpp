#pragma once

#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
**[1] 读取视频
**[2] 图像二值化
*/

struct Yellow
{
    int x;             // 坐标，col
    int y;             // 坐标,row
    int width;         // 尺寸
    int height;        // 尺寸
};


class Preprocess
{

private:
	 cv::Point mid_point = cv::Point(160, 230);
	 int seg_method = 0; // 0-灰度法，1-方差法，2-分割法
	/**
	 * 计算图像的方差
	 * 
	 * 本函数旨在计算给定图像的方差，以评估图像中像素值的离散程度
	 * 方差是衡量图像灰度或颜色值变化程度的一个重要指标，常用于图像处理和分析
	 * 
	 * @param frame 输入的图像矩阵，该图像应为灰度图像或具有单个通道的彩色图像
	 * @param variance 输出的方差矩阵，将计算得到的方差值存储在此矩阵中
	 * 
	 * 注意：该函数假定输入的图像已经过适当的预处理，以确保计算结果的准确性
	 */
	void cal_the_variance(const Mat &img_rgb, Mat &variance)
	{
		cv::Mat gray, img_var;
        cv::cvtColor(img_rgb, gray, cv::COLOR_BGR2GRAY);
        
        // 转换为16位整数以避免溢出
        cv::Mat gray_16, ch0_16, ch1_16, ch2_16;
        gray.convertTo(gray_16, CV_16S);
        
        std::vector<cv::Mat> channels;
        cv::split(img_rgb, channels);
        channels[0].convertTo(ch0_16, CV_16S);
        channels[1].convertTo(ch1_16, CV_16S);
        channels[2].convertTo(ch2_16, CV_16S);
        
        // 计算方差
        cv::Mat diff0, diff1, diff2, sum_diff;
        cv::absdiff(ch0_16, gray_16, diff0);
        cv::absdiff(ch1_16, gray_16, diff1);
        cv::absdiff(ch2_16, gray_16, diff2);
        
        sum_diff = diff0 + diff1 + diff2;
        
        // 除法操作
        cv::Mat gray_plus_one = gray_16 + 1;
        cv::Mat variance_float;
        sum_diff.convertTo(variance_float, CV_32F);
        gray_plus_one.convertTo(gray_plus_one, CV_32F);
        cv::divide(variance_float, gray_plus_one, variance_float);
        
        // 缩放和限制
        variance_float *= 200;
        cv::Mat result;
        variance_float.convertTo(result, CV_8U);
        
        variance = result;
	}

	/**
	 * 使用 floodFill 方法填充图像区域
	 * 
	 * @param frame 输入图像，即要进行填充的图像
	 * @param mask_ret 用于定义填充区域的掩码，只处理掩码中非零的像素位置
	 * @param mid_point 填充的起始点坐标，默认值为 (160, 230)
	 */
	void flood_fill(const Mat &image, Mat &mask_ret, const cv::Point mid_point={160, 230})
	{
		cv::Mat img_rgb;
        cv::cvtColor(image, img_rgb, cv::COLOR_GRAY2RGB);
        
        // 创建掩码
        cv::Mat mask = cv::Mat::zeros(img_rgb.rows + 2, img_rgb.cols + 2, CV_8U);
        
        // 设置掩码区域
        int height_left = std::max(0, mid_point.y - 10);
        int height_right = std::min(img_rgb.rows - 1, mid_point.y + 5);
        int width_left = std::max(0, mid_point.x - 10);
        int width_right = std::min(img_rgb.cols - 1, mid_point.x + 10);
        
        cv::Rect roi(width_left, height_left, width_right - width_left, height_right - height_left);
        img_rgb(roi) = cv::Scalar(210, 210, 210);
        
        // 执行洪水填充
        cv::Scalar newVal(0, 0, 255);
        cv::Scalar loDiff(50, 50, 50);
        cv::Scalar upDiff(50, 50, 50);
        
        cv::floodFill(img_rgb, mask, mid_point, newVal, nullptr, loDiff, upDiff);
        
        // 提取填充区域
        cv::Mat filled_area = mask(cv::Rect(1, 1, mask.cols - 2, mask.rows - 2));
        filled_area *= 255;
        
        mask_ret = filled_area;
	}

	void search_the_mid_point(const Mat &image, cv::Point &mid_point_ret, const cv::Point mid_point_pre ={160, 230})
	{
		cv::Mat img_padded = cv::Mat::zeros(image.rows + 2, image.cols + 2, CV_8U);
        image.copyTo(img_padded(cv::Rect(1, 1, image.cols, image.rows)));
        
        std::vector<int> left_edge, right_edge;
        
        // 搜索左边缘
        for (int i = 230; i < 240; i++) {
            int k = 0;
            int j;
            for (j = mid_point_pre.x; j < 321; j++) {
                if (img_padded.at<uchar>(i, j) <= 20) {
                    k++;
                } else {
                    k = 0;
                }
                if (k >= 4) {
                    left_edge.push_back(j);
                    break;
                }
            }
            if (k < 4) {
                left_edge.push_back(j);
            }
        }
        
        // 搜索右边缘
        for (int i = 230; i < 240; i++) {
            int k = 0;
            int j;
            for (j = mid_point_pre.x; j > 0; j--) {
                if (img_padded.at<uchar>(i, j) <= 20) {
                    k++;
                } else {
                    k = 0;
                }
                if (k >= 4) {
                    right_edge.push_back(j);
                    break;
                }
            }
            if (k < 4) {
                right_edge.push_back(j);
            }
        }
        
        // 计算中点
        double sum = 0;
        for (size_t i = 0; i < left_edge.size() && i < right_edge.size(); i++) {
            sum += (left_edge[i] + right_edge[i]) / 2.0;
        }
        
        int mid_point_width = static_cast<int>(sum / std::min(left_edge.size(), right_edge.size()));
        mid_point_ret = cv::Point(mid_point_width, 230);
	}

	/***
	 * 	@brief 将赛道分割出来
	 * @param frame - 输入图像帧
	 * @param img_seg - 输出分割后的结果，以二值图像的形式分割，255-赛道，255 - 非赛道
	 * @return void - 没有返回值		
	 */
	void get_track_by_seg(const Mat &frame, Mat &img_seg, cv::Point &mid_point_ret, const cv::Point &mid_point_pre = {160, 230})
	{
		cv::Mat image;
		cv::Point mid_point_now; // 当前帧的中点
        cv::resize(frame, image, cv::Size(320, 240));
        
        cv::Mat image_rgb, image_gray;
        cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
        
        // 自适应阈值
        cv::Mat binary;
        cv::adaptiveThreshold(image_gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY, 71, 20);
        
        // 形态学操作
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat filtered;
        cv::morphologyEx(binary, filtered, cv::MORPH_ERODE, kernel);
        cv::morphologyEx(filtered, filtered, cv::MORPH_DILATE, kernel);
        
        // 搜索中点
        search_the_mid_point(filtered, mid_point_now, mid_point_pre);
        
        // 计算方差
        cv::Mat image_var;
        cal_the_variance(image_rgb, image_var);
        
        // 合并图像
        cv::Mat img_merge;
        cv::Mat filtered_float, var_float;
        filtered.convertTo(filtered_float, CV_32F);
        image_var.convertTo(var_float, CV_32F);
        
        img_merge = filtered_float / 2 + (255 - var_float) / 2;
        
        // 形态学操作
        cv::morphologyEx(img_merge, img_merge, cv::MORPH_ERODE, kernel);
        cv::morphologyEx(img_merge, img_merge, cv::MORPH_OPEN, kernel);
        
        // 转换为8位
        cv::Mat img_merge_8u;
        img_merge.convertTo(img_merge_8u, CV_8U);
        
        // 洪水填充
        cv::Mat flood_area;
        flood_fill(img_merge_8u, flood_area, mid_point_now);
        
        // 膨胀操作
        cv::morphologyEx(flood_area, flood_area, cv::MORPH_DILATE, kernel);
        cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(flood_area, flood_area, cv::MORPH_DILATE, kernel3);
        
        // cv::Mat result;
        // cv::cvtColor(flood_area, result, cv::COLOR_GRAY2BGR);
        
        // return std::make_pair(result, mid_point_pre);
		img_seg = flood_area;
		mid_point_ret = mid_point_now;
		
	}

	void get_track_by_var(const Mat &frame, Mat &img_seg, double var_weight = 0.7, double satur_weight = 0.3)
	{
		cv::Mat image;
        cv::resize(frame, image, cv::Size(320, 240));
        
        cv::Mat image_rgb, image_hsv;
        cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(image_rgb, image_hsv, cv::COLOR_RGB2HSV);
        
        cv::Mat image_var;
        cal_the_variance(image_rgb, image_var);
        
        // 权重归一化
        double total_weight = var_weight + satur_weight;
        var_weight /= total_weight;
        satur_weight /= total_weight;
        
        // 提取饱和度通道
        std::vector<cv::Mat> hsv_channels;
        cv::split(image_hsv, hsv_channels);
        
        cv::Mat image_add;
        cv::addWeighted(255 - image_var, var_weight, 255 - hsv_channels[1], satur_weight, 0, image_add);
        
        // 二值化
        cv::Mat binary_frame;
        cv::threshold(image_add, binary_frame, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        // cv::Mat result;
        // cv::cvtColor(binary_frame, result, cv::COLOR_GRAY2BGR);
		img_seg = binary_frame;
	}

	void get_track_by_gary(const Mat &frame, Mat &img_seg)
	{
		cv::Mat image_gray;
        cv::cvtColor(frame, image_gray, cv::COLOR_BGR2GRAY);
        
        cv::Mat binary_frame;
        cv::threshold(image_gray, binary_frame, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
		img_seg = binary_frame;
        // cv::Mat result;
        // cv::cvtColor(binary_frame, result, cv::COLOR_GRAY2BGR);
	}

	
public:
	std::vector<Yellow> yellow;
	/**
	 * @brief 图像矫正参数初始化
	 *
	 */
	Preprocess()
	{
		// 读取xml中的相机标定参数
		cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); // 摄像机内参矩阵
		distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));	// 相机的畸变矩阵
		// FileStorage file;
		// if (file.open("../res/calibration/valid/calibration.xml", FileStorage::READ)) // 读取本地保存的标定文件
		// {
		// 	file["cameraMatrix"] >> cameraMatrix;
		// 	file["distCoeffs"] >> distCoeffs;
		// 	cout << "相机矫正参数初始化成功!" << endl;
		// 	enable = true;
		// }
		// else
		// {
		// 	cout << "打开相机矫正参数失败!!!" << endl;
		// 	enable = false;
		// }
	};


	/**
	 * @brief 分割赛道
	 * @param frame		输入原始帧
	 * @param img_seg	分割后的图像
	 * @param method	分割方法: 0-灰度, 1-方差, 2-自适应阈值分割
	 */
	void track_segment(const Mat &frame, Mat &img_seg, const int method=0)
	{
		if (method == 0)
		{
			get_track_by_gary(frame, img_seg);
		}
		else if (method == 1)
		{
			get_track_by_var(frame, img_seg, 0.7, 0.3);
		}
		else if (method == 2)
		{
		    get_track_by_seg(frame, img_seg, mid_point, mid_point);
		}
		else
		{
		    get_track_by_gary(frame, img_seg);
			cout << "未知的分割方法!!!" << endl;
		}
	}


	/**
	 * @brief 图像二值化
	 *
	 * @param frame			输入原始帧
	 * @param imageBinary 	二值化之后的图像
     * @param method        分割方法: 0-灰度, 1-方差, 2-自适应阈值分割
	 */
	void binaryzation(const Mat &frame, Mat& imageBinary, const int method=0)
	{
		// 分割方法: 0-灰度, 1-方差, 2-自适应阈值分割
		track_segment(frame, imageBinary, method);
		// Mat imageGray;
		// cvtColor(frame, imageGray, COLOR_BGR2GRAY); // RGB转灰度图
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// threshold(imageGray, imageBinary, 0, 255, THRESH_OTSU); // OTSU二值化方法
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}

	/**
	 * @brief 矫正图像
	 *
	 * @param imagesPath 图像路径
	 */
	Mat correction(Mat &image)
	{
		if (enable)
		{
			Size sizeImage; // 图像的尺寸
			sizeImage.width = image.cols;
			sizeImage.height = image.rows;

			Mat mapx = Mat(sizeImage, CV_32FC1);	// 经过矫正后的X坐标重映射参数
			Mat mapy = Mat(sizeImage, CV_32FC1);	// 经过矫正后的Y坐标重映射参数
			Mat rotMatrix = Mat::eye(3, 3, CV_32F); // 内参矩阵与畸变矩阵之间的旋转矩阵

			// 采用initUndistortRectifyMap+remap进行图像矫正
			initUndistortRectifyMap(cameraMatrix, distCoeffs, rotMatrix, cameraMatrix, sizeImage, CV_32FC1, mapx, mapy);
			Mat imageCorrect = image.clone();
			remap(image, imageCorrect, mapx, mapy, INTER_LINEAR);

			// 采用undistort进行图像矫正
			//  undistort(image, imageCorrect, cameraMatrix, distCoeffs);

			return imageCorrect;
		}
		else
		{
			return image;
		}
	}

	void find_yellow(Mat frame, int ylow_H,int ylow_S,int ylow_V,int yhigh_H,int yhigh_S,int yhigh_V)
	{
		yellow.clear();
		Yellow point;
		cv::Mat hsv_frame;
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV_FULL);
		//printf("-------------begin search-----------\n");
        // 定义黄色的HSV范围
        cv::Scalar yellow_lower(ylow_H, ylow_S, ylow_V);
        cv::Scalar yellow_upper(yhigh_H, yhigh_S, yhigh_V);

        // 创建掩码以检测黄色
        cv::Mat yellow_mask;
        cv::inRange(hsv_frame, yellow_lower, yellow_upper, yellow_mask);

        // 查找黄色区域的轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(yellow_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		

        // 设置面积阈值
        int min_area = 80;  // 可根据需要调整
		for (const auto& contour : contours) 
		{   
			//printf("------------------in search--------------\n");
            cv::Rect rect  = cv::boundingRect(contour);
            int area = rect.width * rect.height;
            if (area > min_area) {
				//printf("-----------------search Right------------\n");
                point.x = rect.x;
                point.y= rect.y;
				point.height  = rect.height;
				point.width = rect.width;
				yellow.push_back(point);
                //printf("x: %d , y: %d , height : %d , width: %d \n",point.x,point.y,point.height,point.width);
            }

	}
	
}

	void drawbox(Mat img )
	{
		for (int i = 0; i < yellow.size(); i++)
			{

				cv::Rect rect(yellow[i].x, yellow[i].y, yellow[i].width, yellow[i].height);
				cv::rectangle(img, rect, cv::Scalar(0, 255, 255), 1);
			}
	}

private:
	bool enable = false; // 图像矫正使能：初始化完成
	Mat cameraMatrix;	 // 摄像机内参矩阵
	Mat distCoeffs;		 // 相机的畸变矩阵
};
