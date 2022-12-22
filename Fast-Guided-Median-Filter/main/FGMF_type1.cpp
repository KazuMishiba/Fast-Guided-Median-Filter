#pragma once
#include "FGMF_type1.h"




//Thread数を指定
cv::Mat FGMF1::filter2DInterface(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax) {
	//入力画像チャンネル数
	const int IchannelNum = I.channels();
	//ガイド画像チャンネル数
	const int GchannelNum = G.channels();

	//return FGMF1::filter2DI1G1_MultiThread(I, G, threadNum, radius, eps2, Imax);

	
	if (threadNum == 1)
	{
		//signle thread
		if (IchannelNum == 1 && GchannelNum == 1)
			return FGMF1::filter2DI1G1_SingleThread(I, G, radius, eps2, Imax);
		else if (IchannelNum == 1 && GchannelNum == 3)
			return FGMF1::filter2DI1G3_SingleThread(I, G, radius, eps2, Imax);
		else if (IchannelNum == 3 && GchannelNum == 1)
			return FGMF1::filter2DI3G1_SingleThread(I, G, radius, eps2, Imax);
		else if (IchannelNum == 3 && GchannelNum == 3)
			return FGMF1::filter2DI3G3_SingleThread(I, G, radius, eps2, Imax);
	}
	else
	{
		//multi thread
		if (IchannelNum == 1 && GchannelNum == 1)
			return FGMF1::filter2DI1G1_MultiThread(I, G, threadNum, radius, eps2, Imax);
		else if (IchannelNum == 1 && GchannelNum == 3)
			return FGMF1::filter2DI1G3_MultiThread(I, G, threadNum, radius, eps2, Imax);
		else if (IchannelNum == 3 && GchannelNum == 1)
			return FGMF1::filter2DI3G1_MultiThread(I, G, threadNum, radius, eps2, Imax);
		else if (IchannelNum == 3 && GchannelNum == 3)
			return FGMF1::filter2DI3G3_MultiThread(I, G, threadNum, radius, eps2, Imax);
	}
	
	return cv::Mat();
}


//Single Thread
//I1
cv::Mat FGMF1::filter2DI1G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF1::filter2DI1_SingleThread<gSum, fg_node, int, float>(I, G, radius, eps2, Imax);
}
cv::Mat FGMF1::filter2DI1G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF1::filter2DI1_SingleThread<g3Sum, fg3_node, cv::Vec3i, cv::Vec3f>(I, G, radius, eps2, Imax);
}
template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF1::filter2DI1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	cv::Mat result = cv::Mat(I.size(), I.depth());
	FGMF1::filter2D<GSum, FG, GTYPE, CTYPE>(I, G, result, 0, I.rows - 1, 0, I.cols - 1, radius, eps2, Imax);
	return result;
}
//I3
cv::Mat FGMF1::filter2DI3G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF1::filter2DI3_SingleThread<gSum, fg_node, int, float>(I, G, radius, eps2, Imax);
}
cv::Mat FGMF1::filter2DI3G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF1::filter2DI3_SingleThread<g3Sum, fg3_node, cv::Vec3i, cv::Vec3f>(I, G, radius, eps2, Imax);
}
template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF1::filter2DI3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<cv::Mat> Is;
	split(I, Is);
	std::vector<cv::Mat> results(3);
	results[0] = cv::Mat(I.size(), CV_32S);
	results[1] = cv::Mat(I.size(), CV_32S);
	results[2] = cv::Mat(I.size(), CV_32S);
	cv::Mat cx;
	if (I.channels() == 1)
		cx = cv::Mat(I.size(), CV_32FC1);
	else
		cx = cv::Mat(I.size(), CV_32FC3);
	//cv::Mat cx = cv::Mat(I.size(), cx_cv32);
	cv::Mat dx = cv::Mat(I.size(), CV_32FC1);
	FGMF1::filter2D_saveCD<GSum, FG, GTYPE, CTYPE>(Is[0], G, cx, dx, results[0], 0, I.rows - 1, 0, I.cols - 1, radius, eps2, Imax);
	for (int k = 1; k <= 2; k++)
		FGMF1::filter2D_useCD<GSum, FG, GTYPE, CTYPE>(Is[k], G, cx, dx, results[k], 0, I.rows - 1, 0, I.cols - 1, radius, eps2, Imax);
	merge(results, result);
	return result;
}

//Multi Thread
//cols (幅分割)
//I1
cv::Mat FGMF1::filter2DI1G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	return 	FGMF1::filter2DI1_MultiThread<gSum, fg_node, int, float>(I, G, threadNum, radius, eps2, Imax);
}
cv::Mat FGMF1::filter2DI1G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	return 	FGMF1::filter2DI1_MultiThread<g3Sum, fg3_node, cv::Vec3i, cv::Vec3f>(I, G, threadNum, radius, eps2, Imax);
}
template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF1::filter2DI1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<int> dim0Start(threadNum);
	std::vector<int> dim0End(threadNum);
	calculatePosForCols(I.size(), threadNum, dim0Start, dim0End);
#pragma omp parallel for
	for (int i = 0; i < threadNum; i++)
		FGMF1::filter2D<GSum, FG, GTYPE, CTYPE>(I, G, result, 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);
	return result;
}
//I3
cv::Mat FGMF1::filter2DI3G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	return 	FGMF1::filter2DI3_MultiThread<gSum, fg_node, int, float>(I, G, threadNum, radius, eps2, Imax);
}
cv::Mat FGMF1::filter2DI3G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	return 	FGMF1::filter2DI3_MultiThread<g3Sum, fg3_node, cv::Vec3i, cv::Vec3f>(I, G, threadNum, radius, eps2, Imax);
}
template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF1::filter2DI3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<int> dim0Start(threadNum);
	std::vector<int> dim0End(threadNum);
	calculatePosForCols(I.size(), threadNum, dim0Start, dim0End);
	std::vector<cv::Mat> Is;
	split(I, Is);
	std::vector<cv::Mat> results(3);
	results[0] = cv::Mat(I.size(), CV_32S);
	results[1] = cv::Mat(I.size(), CV_32S);
	results[2] = cv::Mat(I.size(), CV_32S);
	cv::Mat cx;
	if (I.channels() == 1)
		cx = cv::Mat(I.size(), CV_32FC1);
	else
		cx = cv::Mat(I.size(), CV_32FC3);
	//cv::Mat cx = cv::Mat(I.size(), cx_cv32);
	cv::Mat dx = cv::Mat(I.size(), CV_32FC1);

#pragma omp parallel for
	for (int i = 0; i < threadNum; i++)
		FGMF1::filter2D_saveCD<GSum, FG, GTYPE, CTYPE>(Is[0], G, cx, dx, results[0], 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);
	for (int k = 1; k <= 2; k++) {
#pragma omp parallel for
		for (int i = 0; i < threadNum; i++)
			FGMF1::filter2D_useCD<GSum, FG, GTYPE, CTYPE>(Is[k], G, cx, dx, results[k], 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);

	}
	merge(results, result);
	return result;
}