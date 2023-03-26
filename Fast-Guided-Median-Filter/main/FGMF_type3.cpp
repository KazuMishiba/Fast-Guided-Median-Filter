#include "FGMF_type3.h"

#include "GuidedFilter.h"



inline void FGMF3::testBlockGrid()
{
	cu_testBlockGrid();
}


// 自作二値化関数
void my_threshold
(
	const cv::Mat& src, cv::Mat& dst,
	double thresh, double max_value
)
{
	int x, y = 0;
	unsigned char intensity = 0;
	for (y = 0; y < src.rows; y++)
	{
		for (x = 0; x < src.cols; x++)
		{
			intensity = src.at<unsigned char>(y, x);
			if (intensity < (unsigned char)thresh)
			{
				dst.at<unsigned char>(y, x) = 0;
			}
			else
			{
				dst.at<unsigned char>(y, x) = (unsigned char)max_value;
			}
		}
	}
}

// cv::parallel_for_利用コード
class TestParallelLoopBody : public cv::ParallelLoopBody
{
private:
	cv::Mat _src;
	cv::Mat _dst;
	double _thresh;
	double _max_value;
public:
	TestParallelLoopBody
	(
		const cv::Mat& src, cv::Mat& dst,
		double thresh, double max_value
	)
		: _src(src), _dst(dst), _thresh(thresh), _max_value(max_value) { }
	void operator() (const cv::Range& range) const
	{
		int row0 = range.start;
		int row1 = range.end;
		cv::Mat srcStripe = _src.rowRange(row0, row1);
		cv::Mat dstStripe = _dst.rowRange(row0, row1);
		my_threshold(srcStripe, dstStripe, _thresh, _max_value);
	}
};

void FGMF3::testOpencvParallel()
{


}




/*
//テスト用 multi channelデバッグ用
void FGMF3::filter2DGPU_Test(DeviceArray<int>*& I, DeviceArray<int>*& G, DeviceArray<int>*& result, int radius, float eps2, int Imax, SizeInfo& sizeInfo)
{
	//g3用
	float4* cxdx;
	Utility::allocateDeviceMemory(cxdx, sizeInfo);
	DeviceArray<int>* sumG = new DeviceArray<int>(3, sizeInfo);
	DeviceArray<int>* tempG = new DeviceArray<int>(3, sizeInfo);
	DeviceArray<int>* sumGG = new DeviceArray<int>(6, sizeInfo);
	DeviceArray<int>* tempGG = new DeviceArray<int>(6, sizeInfo);
	int pixelNumInWindow = (radius * 2 + 1) * (radius * 2 + 1);
	//cx, dx計算
	cu_calculateCx3DxFromG(sizeInfo, NULL, G, radius, pixelNumInWindow, eps2, cxdx, sumG, sumGG, tempG, tempGG);
	//median計算
	//cu_filter2D(sizeInfo, NULL, radius, Imax, result, I, G, cxdx);


	//multi用
	int yn = G->arrayLength;
	DeviceArray<float>* cxdx2 = new DeviceArray<float>(yn + 1, sizeInfo);
	DeviceArray<int>* sumG2 = new DeviceArray<int>(yn, sizeInfo);
	DeviceArray<int>* tempG2 = new DeviceArray<int>(yn, sizeInfo);
	DeviceArray<int>* sumGG2 = new DeviceArray<int>((yn + 1)*yn / 2, sizeInfo);
	DeviceArray<int>* tempGG2 = new DeviceArray<int>((yn + 1)*yn / 2, sizeInfo);
	//int pixelNumInWindow = (radius * 2 + 1) * (radius * 2 + 1);
	//cx, dx計算
	cu_calculateCxXDxFromG(sizeInfo, NULL, G, radius, pixelNumInWindow, eps2, cxdx2, sumG2, sumGG2, tempG2, tempGG2);


	//median計算
	cu_filter2DTest(sizeInfo, NULL, radius, Imax, result, I, G, cxdx, cxdx2);


	//メモリ開放
	cudaFree(cxdx);
	delete sumG;
	delete tempG;
	delete sumGG;
	delete tempGG;
	//メモリ開放
	delete cxdx2;
	delete sumG2;
	delete tempG2;
	delete sumGG2;
	delete tempGG2;
}

*/



#if 0

cv::Mat FGMF3::filter2DGPU_shared1(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax, SizeInfo& sizeInfo)
{
	//check validation
	assert(I.depth() == CV_32S);
	assert(G.depth() == CV_32S);

	//cudaStream_t stream;
	//cudaStreamCreate(&stream);
	cudaStream_t stream, stream2;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&stream2);
	//cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	//cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

	float2* cxdx_device;
	int2* sumG_device, *temp_device;
	int* G_device;
	int* F_device;
	Utility::allocateDeviceMemory(F_device, I, sizeInfo, stream);
	Utility::allocateDeviceMemory(G_device, G, sizeInfo);
	Utility::allocateDeviceMemory(cxdx_device, sizeInfo);
	Utility::allocateDeviceMemory(sumG_device, sizeInfo);
	Utility::allocateDeviceMemory(temp_device, sizeInfo);
	//cudaStreamSynchronize(stream2);
	//cx, dx計算
	cu_calculateDC(sizeInfo, NULL, G_device, radius, eps2, cxdx_device, sumG_device, temp_device);

	//結果格納用
	int* result_device;
	Utility::allocateDeviceMemory(result_device, sizeInfo);

	cudaStreamSynchronize(stream);
	//median計算
	cu_filter2D_shared1(sizeInfo, NULL, radius, Imax, result_device, F_device, G_device, cxdx_device);

	cv::Mat result(sizeInfo.height_, sizeInfo.width_, CV_32SC1);
	//cudaHostRegister(result.data, I.total() * I.elemSize(), cudaHostRegisterDefault);
	//result = Utility::downloadLinearArrayAsMat(result_device, sizeInfo);
	cudaMemcpy2D(result.data, result.step, result_device, sizeInfo.pitch<int>(), sizeInfo.width_ * sizeof(int), sizeInfo.height_, cudaMemcpyDefault);

	//メモリ開放
	cudaFree(result_device);
	cudaFree(cxdx_device);
	cudaFree(sumG_device);
	cudaFree(temp_device);
	cudaFree(F_device);
	cudaFree(G_device);

	return result;
}

//page lock memory使用
cv::Mat FGMF3::filter2DGPU_shared2(int* I, int* G, int radius, float eps2, int Imax, SizeInfo& sizeInfo)
{
	//cu_testBlockGrid();

	cudaStream_t stream, stream2;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&stream2);
	//cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	//cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

	float2* cxdx_device;
	int2* sumG_device, *temp_device;
	int* G_device;
	int* F_device;
	//Utility::allocateDeviceMemory(F_device, I, sizeInfo, stream);
	//Utility::allocateDeviceMemory(G_device, G, sizeInfo, stream2);
	Utility::allocateDeviceMemory(F_device, I, sizeInfo);
	Utility::allocateDeviceMemory(G_device, G, sizeInfo);

	//Utility::showDevice(F_device, sizeInfo, "F", false, 256);
	//Utility::showDevice(G_device, sizeInfo, "G", false, 256);

	Utility::allocateDeviceMemory(cxdx_device, sizeInfo);
	Utility::allocateDeviceMemory(sumG_device, sizeInfo);
	Utility::allocateDeviceMemory(temp_device, sizeInfo);
	cudaStreamSynchronize(stream2);
	//cx, dx計算
	cu_calculateDC(sizeInfo, NULL, G_device, radius, eps2, cxdx_device, sumG_device, temp_device);

	//結果格納用
	int* result_device;
	Utility::allocateDeviceMemory(result_device, sizeInfo);

	cudaStreamSynchronize(stream);
	//median計算
	cu_filter2D_shared1(sizeInfo, NULL, radius, Imax, result_device, F_device, G_device, cxdx_device);
	cv::Mat result = Utility::downloadLinearArrayAsMat(result_device, sizeInfo);

	//メモリ開放
	cudaFree(result_device);
	cudaFree(cxdx_device);
	cudaFree(sumG_device);
	cudaFree(temp_device);
	cudaFree(F_device);
	cudaFree(G_device);

	return result;
	//return cv::Mat();
}



//fのみ(セルフガイド)
void FGMF3::filter2DGPU_selfGuide(int*& I, int*& result, int radius, float eps2, int Imax, SizeInfo& sizeInfo)
{
	float2* cxdx;
	int2* sumG, *temp;
	Utility::allocateDeviceMemory(cxdx, sizeInfo);
	Utility::allocateDeviceMemory(sumG, sizeInfo);
	Utility::allocateDeviceMemory(temp, sizeInfo);
	//cx, dx計算
	cu_calculateDC(sizeInfo, NULL, I, radius, eps2, cxdx, sumG, temp);
	//median計算
	cu_filter2D_selfGuide(sizeInfo, NULL, radius, Imax, result, I, cxdx);

	//メモリ開放
	cudaFree(cxdx);
	cudaFree(sumG);
	cudaFree(temp);
}



#endif