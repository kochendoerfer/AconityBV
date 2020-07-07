#include "ImgProc.h"
#include <iostream>

void findMarker(std::string srcPath, std::string dstPath)
{
	//filtering
	///load image
	///type is CV_8U
	cv::Mat src = cv::imread(srcPath, cv::IMREAD_UNCHANGED);
	ImgProc proc;
	///generate filter kernel
	cv::Mat kernel = cv::Mat(src.size(), CV_32F, cv::Scalar(0));
	int size = 10;
	proc.synthesizeCrossFilter(kernel, cv::Point(int(kernel.cols / 2), int(kernel.rows / 2)), MIN(kernel.rows, kernel.cols), size);
	proc.fftShift(kernel, kernel);
	///filtering
	cv::Mat filtered;
	src.convertTo(src, CV_32F);
	proc.filter2DFreq(src, filtered, kernel);
	filtered.convertTo(filtered, CV_8U);
	cv::normalize(filtered, filtered, 0, 255, cv::NORM_MINMAX);
	//cv::imshow("filtered", filtered);
	//morphological operations
	cv::Mat thresh, morphX, morphY;
	//threshold first
	proc.thresh(filtered, thresh, 90, cv::THRESH_BINARY);
	thresh = (255 - thresh);
	//cv::imshow("thresh", thresh);
	//bring out rect in x-direction
	proc.erode(thresh, morphY, 2, 19, cv::MORPH_CROSS);
	proc.dilate(morphY, morphY, 3, morphY.rows, cv::MORPH_RECT);
	//bring out rect in y-direction
	proc.erode(thresh, morphX, 17, 3, cv::MORPH_CROSS);
	proc.dilate(morphX, morphX, morphX.cols, 4, cv::MORPH_RECT);

	//find rectangles
	std::vector<std::vector<cv::Point>> rectX, rectY;
	proc.findRect(morphX, rectX);
	proc.findRect(morphY, rectY);
	cv::Point p1 = cv::Point(rectY[0][0].x, rectX[0][0].y);
	cv::Point p2 = cv::Point(rectY[0][0].x, rectX[0][1].y);
	cv::Point p3 = cv::Point(rectY[0][2].x, rectX[0][0].y);
	cv::Point p4 = cv::Point(rectY[0][2].x, rectX[0][1].y);
	cv::Point pm = cv::Point((p3.x + p2.x) / 2, (p3.y + p2.y) / 2);
	cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
	src.convertTo(src, CV_8UC3);
	cv::circle(src, p1, 2, cv::Scalar(0, 0, 255), 1);
	cv::circle(src, p2, 2, cv::Scalar(0, 0, 255), 1);
	cv::circle(src, p3, 2, cv::Scalar(0, 0, 255), 1);
	cv::circle(src, p4, 2, cv::Scalar(0, 0, 255), 1);
	cv::circle(src, pm, 5, cv::Scalar(0, 0, 255), 1);
	cv::imwrite(dstPath, src);

}

void doCLAHE(const cv::Mat &src, cv::Mat &dst, int blocksize)
{
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2, cv::Size(blocksize, blocksize));
	clahe->apply(src, dst);
}

void adaptiveThresh(const cv::Mat &src, cv::Mat &dst, int max, int threshType, int threshFlag, int blocksize, int C)
{
	cv::adaptiveThreshold(src, dst, max, threshType, threshFlag, blocksize, C);
}

int main(int argc, char* argv[])
{
	std::string imgName = "S1_M1_1000us.bmp";
	std::string srcpath = "H:/WorkDir/AufnahmenAconity/" + imgName;
	std::string dstpath = "H:/WorkDir/AufnahmenAconity/" + std::string("Res_") + imgName;
	cv::Mat src = cv::imread(srcpath, cv::IMREAD_GRAYSCALE);
	cv::Mat srcLines;
	src.copyTo(srcLines);
	cv::Mat clahe, adaptive, houghCol, thresh;
	//static threshold
	cv::threshold(src, thresh, 200, 255, cv::THRESH_BINARY);
	thresh.copyTo(adaptive);
	////adaptive threshold
	//doCLAHE(src, clahe, 25);
	//adaptiveThresh(clahe, adaptive, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 51, 10);
	//adaptive = 255 - adaptive;
	
	//proc.erode(adaptive, adaptive, 2, 2, cv::MORPH_RECT);
	cv::cvtColor(adaptive, houghCol, CV_GRAY2BGR);
	cv::cvtColor(src, src, CV_GRAY2BGR);
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(adaptive, lines, 1, CV_PI / 180, 50, 70, 70);

	//reject invalid lines
	//lines have to be either horizontal or vertical and be around the middle area
	int tolerance = 5;
	int x = src.cols;
	int y = src.rows;
	double factor = 0.15;
	auto it = lines.begin();
	while (it != lines.end())
	{
		if (abs(it->val[0] - it->val[2]) > tolerance
			&& abs(it->val[1] - it->val[3]) > tolerance)
			it = lines.erase(it);
		else
			it++;
	}

	//sort lines in horizontal and vertical lines
	//get average of lines
	std::vector<cv::Vec4i>hlines, vlines;
	int yAvg = 0;
	int yCount = 0;
	int xAvg = 0;
	int xCount = 0;
	for (size_t i = 0; i < lines.size(); i++)
	{
		//horizontal lines
		if (abs(lines[i][0] - lines[i][2]) > abs(lines[i][1] - lines[i][3]))
		{
			if (lines[i][1] < y / 2 + y * factor
				&& lines[i][1] > y / 2 - y * factor)
			{
				hlines.push_back(lines.at(i));
				yAvg += (lines[i][1] + lines[i][3]) / 2;
				yCount++;
				//cv::line(houghCol, cv::Point(lines[i][0], lines[i][1]),
				//	cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0), 1, 8);
			}
		}
		//vertical lines
		else if (abs(lines[i][0] - lines[i][2]) < abs(lines[i][1] - lines[i][3]))
		{
			if (lines[i][0] < x / 2 + x * factor
				&& lines[i][0] > x / 2 - x * factor)
			{
				vlines.push_back(lines.at(i));
				xAvg += (lines[i][0] + lines[i][2]) / 2;
				xCount++;
				//cv::line(houghCol, cv::Point(lines[i][0], lines[i][1]),
				//	cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 1, 8);
			}
		}
	}
	xAvg /= xCount;
	yAvg /= yCount;

	cv::circle(houghCol, cv::Point(xAvg, yAvg), 3, cv::Scalar(0, 0, 255));

	//draw middle lines
	/*cv::line(houghCol, cv::Point(x/2, 0),
		cv::Point(x/2,y), cv::Scalar(0, 255, 0), 1, 8);
	cv::line(houghCol, cv::Point(x / 2 + x * factor, 0),
		cv::Point(x / 2 + x * factor, y), cv::Scalar(255, 0, 0), 1, 8);
	cv::line(houghCol, cv::Point(x / 2 - x * factor, 0),
		cv::Point(x / 2 - x * factor, y), cv::Scalar(255, 0, 0), 1, 8);

	cv::line(houghCol, cv::Point(0, y/2),
		cv::Point(x, y/2), cv::Scalar(0, 255, 0), 1, 8);
	cv::line(houghCol, cv::Point(0, y/2 - y * factor),
		cv::Point(x, y/2 - y * factor), cv::Scalar(255, 0, 0), 1, 8);
	cv::line(houghCol, cv::Point(0, y / 2 + y * factor),
		cv::Point(x, y / 2 + y * factor), cv::Scalar(255, 0, 0), 1, 8);*/

	//print lines in img
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	cv::line(houghCol, cv::Point(lines[i][0], lines[i][1]),
	//		cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 1, 8);
	//}

	/*
	int vmin = INT_MAX;
	int vmax = -1;
	int hmin = INT_MAX;
	int hmax = -1;
	//find min max in y-direction off horizontal lines
	for (size_t i = 0; i < hlines.size(); i++)
	{
		int avg = (hlines[i][1] + hlines[i][3]) / 2;
		hmin = (hmin < avg) ? hmin : avg;
		hmax = (hmax > avg) ? hmax : avg;
	}
	int hDist = (hmax - hmin) * 0.1;
	//and min max in x-direction off vertical lines
	for (size_t i = 0; i < vlines.size(); i++)
	{
		int avg = (vlines[i][0] + vlines[i][2]) / 2;
		vmin = (vmin < avg) ? vmin : avg;
		vmax = (vmax > avg) ? vmax : avg;

	}
	int vDist = (vmax - vmin) * 0.1;

	//sort horizontal lines into up and down
	int hUp = 0;
	int hDown = 0;
	int hUpC = 0;
	int hDownC = 0;
	for (size_t i = 0; i < hlines.size(); i++)
	{
		int avg = (hlines[i][1] + hlines[i][3]) / 2;
		if (avg <= hmin + hDist
			&& avg >= hmin - hDist)
		{
			hUp += avg;
			hUpC++;
		}
		else if (avg <= hmax + hDist
			&& avg >= hmax - hDist)
		{
			hDown += avg;
			hDownC++;
		}
	}
	hUp = hUp / hUpC;
	hDown = hDown / hDownC;
	
	//sort vertical lines into left and right
	int vLeft = 0;
	int vRight = 0;
	int vLeftC = 0;
	int vRightC = 0;
	for (size_t i = 0; i < vlines.size(); i++)
	{
		int avg = (vlines[i][0] + vlines[i][2]) / 2;
		if (avg <= vmin + vDist
			&& avg >= vmin - vDist)
		{
			vLeft += avg;
			vLeftC++;
		}
		else if (avg <= vmax + vDist
			&& avg >= vmax - vDist)
		{
			vRight += avg;
			vRightC++;
		}
	}
	vLeft = vLeft / vLeftC;
	vRight = vRight / vRightC;

	//draw lines
	cv::line(src, cv::Point(vLeft, 0),
		cv::Point(vLeft, src.rows),
		cv::Scalar(0, 0, 255), 1, 8);
	cv::line(src, cv::Point(vRight, 0),
		cv::Point(vRight, src.rows),
		cv::Scalar(255, 0, 0), 1, 8);
	
	cv::line(src, cv::Point(0, hUp),
		cv::Point(src.cols, hUp),
		cv::Scalar(0, 255, 0), 1, 8);
	cv::line(src, cv::Point(0, hDown),
		cv::Point(src.cols, hDown),
		cv::Scalar(255, 0, 255), 1, 8);
	cv::Point mid((vLeft + vRight) / 2, (hUp + hDown) / 2);
	cv::circle(src, mid, 5, cv::Scalar(0, 0, 255));
	std::cout << mid;*/

	cv::imshow("borders and marker", src);
	cv::imshow("thresh 200", thresh);
	cv::imshow("all hough lines", houghCol);
	cv::waitKey(0);

	//cv::imwrite(dstpath, src);
	return 0;
}