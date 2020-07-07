#pragma once

#include <opencv2\core\core.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\imgcodecs\imgcodecs.hpp>

class ImgProc
{
public:
	ImgProc();
	~ImgProc();

	void thresh(cv::Mat &src, cv::Mat &dst, int threshold, int threshFlag);
	void adptThresh(cv::Mat &src, cv::Mat &dst, double maxValue, int adaptiveMethod, int threshType, int blocksize, double C);
	void erode(const cv::Mat &src, cv::Mat &dst, int sizeX, int sizeY, int structType);
	void dilate(const cv::Mat &src, cv::Mat &dst, int sizeX, int sizeY, int structType);
	void morph(const cv::Mat &src, cv::Mat &dst, int operation, int sizeX, int sizeY, int structType);
	void fftShift(cv::Mat &src, cv::Mat &dst);
	void getPowerSpectrum(cv::Mat &src, cv::Mat &magI);
	void synthesizeRadianFilter(cv::Mat &inout, cv::Point center, int radius);
	void synthesizeCrossFilter(cv::Mat &inout, cv::Point center, int size, int thickness);
	void filter2DFreq(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);
	void findRect(cv::Mat &src, std::vector<std::vector<cv::Point>> &rectangles);
};

