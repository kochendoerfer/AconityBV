#include "ImgProc.h"



ImgProc::ImgProc()
{
}


ImgProc::~ImgProc()
{
}

void ImgProc::thresh(cv::Mat &src, cv::Mat &dst, int threshold, int threshFlag)
{
	if (threshFlag & cv::THRESH_OTSU == cv::THRESH_OTSU)
	{
		threshold = 0;
	}
	cv::threshold(src, dst, threshold, 255, threshFlag);
}

void ImgProc::adptThresh(cv::Mat &src, cv::Mat &dst, double maxValue, int adaptiveMethod, int threshType, int blocksize, double C)
{
	cv::adaptiveThreshold(src, dst, maxValue, adaptiveMethod, threshType, blocksize, C);
}

void ImgProc::erode(const cv::Mat &src, cv::Mat &dst, int sizeX, int sizeY, int structType)
{
	cv::Mat element = cv::getStructuringElement(structType,
		cv::Size(2 * sizeX + 1, 2 * sizeY + 1),
		cv::Point(sizeX, sizeY));

	cv::erode(src, dst, element);
}

void ImgProc::dilate(const cv::Mat &src, cv::Mat &dst, int sizeX, int sizeY, int structType)
{
	cv::Mat element = cv::getStructuringElement(structType,
		cv::Size(2 * sizeX + 1, 2 * sizeY + 1),
		cv::Point(sizeX, sizeY));

	cv::dilate(src, dst, element);
}

void ImgProc::morph(const cv::Mat &src, cv::Mat &dst, int operation, int sizeX, int sizeY, int structType)
{
	cv::Mat element = cv::getStructuringElement(structType,
		cv::Size(2 * sizeX + 1, 2 * sizeY + 1),
		cv::Point(sizeX, sizeY));
	cv::morphologyEx(src, dst, operation, element);
}

void ImgProc::fftShift(cv::Mat &src, cv::Mat &dst)
{
	dst = src.clone();
	int cx = dst.cols / 2;
	int cy = dst.rows / 2;
	cv::Mat q0(dst, cv::Rect(0, 0, cx, cy));
	cv::Mat q1(dst, cv::Rect(cx, 0, cx, cy));
	cv::Mat q2(dst, cv::Rect(0, cy, cx, cy));
	cv::Mat q3(dst, cv::Rect(cx, cy, cx, cy));
	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void ImgProc::getPowerSpectrum(cv::Mat &I, cv::Mat &magI)
{
	cv::Mat padded;                            //expand input image to optimal size
	int m = cv::getOptimalDFTSize(I.rows);
	int n = cv::getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexI;
	cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	magI = planes[0];
	cv::pow(magI, 2, magI);
	magI += cv::Scalar::all(1);                    // switch to logarithmic scale
	cv::log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
}

void ImgProc::synthesizeRadianFilter(cv::Mat& inout, cv::Point center, int radius)
{
	cv::circle(inout, center, radius, 1, -1, 8);
}

void ImgProc::synthesizeCrossFilter(cv::Mat &inout, cv::Point center, int size, int thickness)
{
	cv::drawMarker(inout, center, 1, cv::MARKER_CROSS, size, thickness);
}

void ImgProc::filter2DFreq(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel)
{
	cv::Mat planes[2] = { cv::Mat_<float>(src.clone()), cv::Mat::zeros(src.size(), CV_32F) };
	cv::Mat complexI;
	cv::merge(planes, 2, complexI);
	cv::dft(complexI, complexI, cv::DFT_SCALE);
	cv::Mat planesH[2] = { cv::Mat_<float>(kernel.clone()), cv::Mat::zeros(kernel.size(), CV_32F) };
	cv::Mat complexH;
	cv::merge(planesH, 2, complexH);
	cv::Mat complexIH;
	cv::mulSpectrums(complexI, complexH, complexIH, 0);
	cv::idft(complexIH, complexIH);
	cv::split(complexIH, planes);
	dst = planes[0];
}

double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
}

void ImgProc::findRect(cv::Mat &src, std::vector<std::vector<cv::Point>> &rectangles)
{
	std::vector < std::vector<cv::Point> > contours;
	cv::findContours(src, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	std::vector<cv::Point> approx;
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true)*0.002, true);
		if (approx.size() == 4 &&
			fabs(cv::contourArea(approx)) > 1000 &&
			cv::isContourConvex(approx))
		{
			double maxCosine = 0;
			for (size_t j = 2; j < 5; j++)
			{
				double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
				maxCosine = MAX(maxCosine, cosine);
			}
			// if cosines of all angles are small
			// (all angles are ~90 degree) then write quandrange
			// vertices to resultant sequence
			if (maxCosine < 0.3)
				rectangles.push_back(approx);
		}
	}
}