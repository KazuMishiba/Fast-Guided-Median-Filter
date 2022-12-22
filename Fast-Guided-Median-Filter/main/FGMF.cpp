#include "FGMF.h"

cv::Mat FGMF::filter2DInterfaceTest(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax) {
	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<int> dim0Start(threadNum);
	std::vector<int> dim0End(threadNum);
	calculatePosForCols(I.size(), threadNum, dim0Start, dim0End);
#pragma omp parallel for
	for (int i = 0; i < threadNum; i++)
		FGMF::filter2Dtest<gSum, fgSumUpToIndex, fg, int, float>(I, G, result, 0, I.rows - 1, 0, I.cols - 1, radius, eps2, Imax);
	return result;



}

//Thread数を指定
cv::Mat FGMF::filter2DInterface(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax) {
	//入力画像チャンネル数
	const int IchannelNum = I.channels();
	//ガイド画像チャンネル数
	const int GchannelNum = G.channels();

	if (threadNum == 1)
	{
		//signle thread
		if (IchannelNum == 1 && GchannelNum == 1)
			return FGMF::filter2DI1G1_SingleThread(I, G, radius, eps2, Imax);
		else if (IchannelNum == 1 && GchannelNum == 3)
			return FGMF::filter2DI1G3_SingleThread(I, G, radius, eps2, Imax);
		else if (IchannelNum == 3 && GchannelNum == 1)
			return FGMF::filter2DI3G1_SingleThread(I, G, radius, eps2, Imax);
		else if (IchannelNum == 3 && GchannelNum == 3)
			return FGMF::filter2DI3G3_SingleThread(I, G, radius, eps2, Imax);
	}
	else
	{
		//multi thread
		if (IchannelNum == 1 && GchannelNum == 1)
			return FGMF::filter2DI1G1_MultiThread(I, G, threadNum, radius, eps2, Imax);
		else if (IchannelNum == 1 && GchannelNum == 3)
			return FGMF::filter2DI1G3_MultiThread(I, G, threadNum, radius, eps2, Imax);
		else if (IchannelNum == 3 && GchannelNum == 1)
			return FGMF::filter2DI3G1_MultiThread(I, G, threadNum, radius, eps2, Imax);
		else if (IchannelNum == 3 && GchannelNum == 3)
			return FGMF::filter2DI3G3_MultiThread(I, G, threadNum, radius, eps2, Imax);
	}
 	return cv::Mat();
}




//Single Thread
//I1
cv::Mat FGMF::filter2DI1G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI1_SingleThread<gSum, fgSumUpToIndex, fg, int, float>(I, G, radius, eps2, Imax);
}
cv::Mat FGMF::filter2DI1G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI1_SingleThread<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(I, G, radius, eps2, Imax);
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF::filter2DI1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	cv::Mat result = cv::Mat(I.size(), I.depth());
	FGMF::filter2D<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(I, G, result, 0, I.rows - 1, 0, I.cols - 1, radius, eps2, Imax);
	return result;
}
//I3
cv::Mat FGMF::filter2DI3G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI3_SingleThread<gSum, fgSumUpToIndex, fg, int, float>(CV_32FC1, I, G, radius, eps2, Imax);
}
cv::Mat FGMF::filter2DI3G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI3_SingleThread<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(CV_32FC3, I, G, radius, eps2, Imax);
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF::filter2DI3_SingleThread(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<cv::Mat> Is;
	split(I, Is);
	std::vector<cv::Mat> results(3);
	results[0] = cv::Mat(I.size(), CV_32S);
	results[1] = cv::Mat(I.size(), CV_32S);
	results[2] = cv::Mat(I.size(), CV_32S);
	cv::Mat cx = cv::Mat(I.size(), cx_cv32);
	cv::Mat dx = cv::Mat(I.size(), CV_32FC1);
	FGMF::filter2D_saveCD<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(Is[0], G, cx, dx, results[0], 0, I.rows - 1, 0, I.cols - 1, radius, eps2, Imax);
	for (int k = 1; k <= 2; k++)
		FGMF::filter2D_useCD<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(Is[k], G, cx, dx, results[k], 0, I.rows - 1, 0, I.cols - 1, radius, eps2, Imax);
	merge(results, result);
	return result;
}


//

//Multi Thread
//cols (幅分割)
//I1
cv::Mat FGMF::filter2DI1G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI1_MultiThread<gSum, fgSumUpToIndex, fg, int, float>(I, G, threadNum, radius, eps2, Imax);
}
cv::Mat FGMF::filter2DI1G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI1_MultiThread<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(I, G, threadNum, radius, eps2, Imax);
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF::filter2DI1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<int> dim0Start(threadNum);
	std::vector<int> dim0End(threadNum);
	calculatePosForCols(I.size(), threadNum, dim0Start, dim0End);
#pragma omp parallel for
	for (int i = 0; i < threadNum; i++)
		FGMF::filter2D<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(I, G, result, 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);
	return result;
}
//I3
cv::Mat FGMF::filter2DI3G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI3_MultiThread<gSum, fgSumUpToIndex, fg, int, float>(CV_32FC1, I, G, threadNum, radius, eps2, Imax);
}
cv::Mat FGMF::filter2DI3G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI3_MultiThread<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(CV_32FC3, I, G, threadNum, radius, eps2, Imax);
}


template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF::filter2DI3_MultiThread(int cx_cv32, cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{

	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<int> dim0Start(threadNum);
	std::vector<int> dim0End(threadNum);
	calculatePosForCols(I.size(), threadNum, dim0Start, dim0End);

	std::vector<cv::Mat> Is;
	cv::split(I, Is);
	std::vector<cv::Mat> results(3);
	results[0] = cv::Mat(I.size(), CV_32S);
	results[1] = cv::Mat(I.size(), CV_32S);
	results[2] = cv::Mat(I.size(), CV_32S);

	cv::Mat cx = cv::Mat(I.size(), cx_cv32);
	cv::Mat dx = cv::Mat(I.size(), CV_32FC1);
#pragma omp parallel for
	for (int i = 0; i < threadNum; i++)
		FGMF::filter2D_saveCD<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(Is[0], G, cx, dx, results[0], 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);

	/*
	cv::imshow("cccpu", cv::abs(cx) * 25600);
	cv::imshow("ddcpu", cv::abs(dx) * 256);
	cv::waitKey(0);
	*/


	for (int k = 1; k <= 2; k++)
#pragma omp parallel for
		for (int i = 0; i < threadNum; i++)
			FGMF::filter2D_useCD<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(Is[k], G, cx, dx, results[k], 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);

	/*
	for (int k = 0; k <= 2; k++)
#pragma omp parallel for
		for (int i = 0; i < threadNum; i++)
			FGMF::filter2D<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(Is[k], G, results[k], 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);
	*/
	cv::merge(results, result);
	return result;
}



//I3G1
void FGMF::calculateCxDxOnGPU(cv::Mat& GMat, int radius, float eps2, SizeInfo& sizeInfo, cv::Mat& cxMat, cv::Mat& dxMat)
{
	/*
	//mat GMat を int* Gにコピー
	int* G;
	Utility::allocateDeviceMemory(G, GMat, sizeInfo);
	//Utility::showDevice(G, sizeInfo, "test", false, 255);
	//float* cx, dxを確保
	float* cx;
	float* dx;
	Utility::allocateDeviceMemory(cx, sizeInfo);
	Utility::allocateDeviceMemory(dx, sizeInfo);
	//int4* sumG, tempを確保
	int4* sumG;
	int4* temp;
	Utility::allocateDeviceMemory(sumG, sizeInfo);
	Utility::allocateDeviceMemory(temp, sizeInfo);
	cu_calculateCxDxFromG(sizeInfo, NULL, G, radius, eps2, cx, dx, sumG, temp);
	//cx,dxをcxMat, dxMatにコピー
	cxMat = Utility::downloadLinearArrayAsMat(cx, sizeInfo);
	dxMat = Utility::downloadLinearArrayAsMat(dx, sizeInfo);
	*/



	/*
	cv::imshow("cc", cv::abs(cxMat) * 25600);
	cv::imshow("dd", cv::abs(dxMat) * 256);
	cv::waitKey(0);
	*/
}


cv::Mat FGMF::gpuTestCxDx(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax, SizeInfo& sizeInfo) {

	int threadNum = 12;
	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<int> dim0Start(threadNum);
	std::vector<int> dim0End(threadNum);
	calculatePosForCols(I.size(), threadNum, dim0Start, dim0End);

	std::vector<cv::Mat> Is;
	cv::split(I, Is);
	std::vector<cv::Mat> results(3);
	results[0] = cv::Mat(I.size(), CV_32S);
	results[1] = cv::Mat(I.size(), CV_32S);
	results[2] = cv::Mat(I.size(), CV_32S);

	cv::Mat cx = cv::Mat(I.size(), CV_32FC1);
	cv::Mat dx = cv::Mat(I.size(), CV_32FC1);

	FGMF::calculateCxDxOnGPU(G, radius, eps2, sizeInfo, cx, dx);

	for (int k = 0; k <= 2; k++)
	{
#pragma omp parallel for
		for (int i = 0; i < threadNum; i++)
			FGMF::filter2D_useCD<gSum, fgSumUpToIndex, fg, int, float>(Is[k], G, cx, dx, results[k], 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);
	}
	/*
	for (int k = 0; k <= 2; k++)
#pragma omp parallel for
		for (int i = 0; i < threadNum; i++)
			FGMF::filter2D<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(Is[k], G, results[k], 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);
	*/
	cv::merge(results, result);
	return result;

}



/*
cv::Mat FGMF::filter2DI3G3_MultiThread2(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI3_MultiThread2<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(CV_32FC3, I, G, radius, eps2, Imax);
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF::filter2DI3_MultiThread2(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	int colNum = 12;

	cv::Mat result = cv::Mat(I.size(), CV_32SC3);
	std::vector<int> dim0Start(colNum);
	std::vector<int> dim0End(colNum);
	calculatePosForCols(I.size(), colNum, dim0Start, dim0End);

#pragma omp parallel for
	for (int i = 0; i < colNum; i++)
		FGMF::filter2DColor<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(I, G, result, 0, I.rows - 1, dim0Start[i], dim0End[i], radius, eps2, Imax);
	return result;
}



cv::Mat FGMF::filter2DI3G3_MultiThread3(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	return 	FGMF::filter2DI3_MultiThread3<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(CV_32FC3, I, G, radius, eps2, Imax);
}
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF::filter2DI3_MultiThread3(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax)
{
	int colNum = 12;

	cv::Mat result = cv::Mat(I.size(), CV_32SC3);
	std::vector<int> dim0Start(colNum);
	std::vector<int> dim0End(colNum);
	calculatePosForCols(I.size(), colNum, dim0Start, dim0End);

#pragma omp parallel for
	for (int i = 0; i < colNum; i++)
		FGMF::filter2DColor<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(I, G, result, dim0Start[i], dim0End[i], radius, eps2, Imax);
	return result;
}
*/











#if 0
//チャンネル方向を3次元目として３Dフィルタリングを適用する
cv::Mat FGMF::filter2DColor(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	int radius_depth = 2;// I.channels() - 1;
	//チャンネルをフレームとして分離する
	std::vector<cv::Mat> Is;
	split(I, Is);
	std::vector<cv::Mat> Gs;
	split(G, Gs);


	//
	std::vector< std::unique_ptr<cv::Mat>> pIs, pGs;
	/*
	std::vector<cv::Mat*> ptrIs, ptrGs;
	//for (int i = 0; i < Is.size(); i++)
	for (int i = 0; i < 1; i++)
	{
		ptrIs.push_back(&Is[i]);
		ptrGs.push_back(&Gs[i]);
	}*/
	/*
	//for (int i = 0; i < Is.size(); i++)
	for (int i = 0; i < 1; i++)
	{
		//cv::Mat* pI = &Is[i];
		pIs.emplace_back(move(&Is[i]));
		pGs.emplace_back(move(&Gs[i]));
		//cv::Mat* pG = &Gs[i];
		//pGs.push_back(unique_ptr<cv::Mat>(move(ptrGs[i])));
	}
	std::vector<unique_ptr<cv::Mat>> pResults = FGMF::filter3DWindow2<gSum, fgSumUpToIndex, fg, int, float>(pIs, pGs, threadNum, radius, radius_depth, eps2, Imax);
	//結果の画像化
	std::vector<cv::Mat> results;
	//for (int i = 0; i < Is.size(); i++)
	for (int i = 0; i < 1; i++)
	{
		results.push_back(*pResults[i]);
	}
	*/
	
	
	std::vector<cv::Mat> Is2, Gs2;
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Is2.push_back(Is[0]);
			Gs2.push_back(Gs[0]);
		}
	}
	std::vector<cv::Mat> results = FGMF::filter3DI1<gSum, fgSumUpToIndex, fg, int, float>(Is2, Gs2, threadNum, radius, radius_depth, eps2, Imax);
	std::vector<cv::Mat> results2(3);
	results2[0] = results[0];
	results2[1] = results[1];
	results2[2] = results[2];

	cv::Mat result;
	merge(results2, result);

	result.convertTo(result, CV_8U);
	cv::imshow("Result (aaa)", result);
	cv::waitKey(0);
	return result;
	//return cv::Mat();
}
#endif