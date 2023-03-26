#include "FGMF_GPU_Or_ver0.h"
#include "FGMF_GPU_Or_ver0.cuh"

namespace FGMF_GPU_Or_ver0
{
	/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			cu_memoryInfo();
			if (abort) exit(code);
		}
	}*/

	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange)
	{
		WMF wmf = WMF(f_img, g_img, radius, eps2, fRange);

		std::chrono::system_clock::time_point start, end;
		start = std::chrono::system_clock::now();

		wmf.apply_2d_filter();

		end = std::chrono::system_clock::now();
		double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Prop Win (after refactoring without data transfer): ";
		std::cout << time << " [ms] / " << 1 << "[times]" << std::endl;

		return wmf.downloadResultAsMat();
	}

	void WMF::apply_2d_filter()
	{
		if (channelNum_g_ == 1)
			filter2DGPU<float2>(f_device_, g_device_, result_device_, radius_, eps2_, fRange_, sizeInfo_);
		else if (channelNum_g_ == 3)
			filter2DGPU<float4>(f_device_, g_device_, result_device_, radius_, eps2_, fRange_, sizeInfo_);
	}

	cv::Mat filter_2d_multichannel(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange)
	{
		WMF wmf = WMF(f_img, g_img, radius, eps2, fRange);
		std::chrono::system_clock::time_point start, end;
		start = std::chrono::system_clock::now();

		wmf.apply_2d_filter_multichannel();

		end = std::chrono::system_clock::now();
		double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Prop Win (after refactoring without data transfer): ";
		std::cout << time << " [ms] / " << 1 << "[times]" << std::endl;

		return wmf.downloadResultAsMat();
	}

	void WMF::apply_2d_filter_multichannel()
	{
		filter2DGPU_MultiChannel(f_device_, g_device_, result_device_, radius_, eps2_, fRange_, sizeInfo_);
	}


	cv::Mat WMF::downloadResultAsMat()
	{
		return Helper::UtilityForCUDA::downloadLinearArrayAsMat(result_device_, sizeInfo_);
	}

	//////////////////////////////////////////////////////////////////////
	//2D
	template<typename DC_TYPE>
	void WMF::filter2DGPU(Helper::DeviceArray<int>*& f, Helper::DeviceArray<int>*& g, Helper::DeviceArray<int>*& result, int radius, float eps2, int fRange, Helper::SizeInfo& sizeInfo)
	{
		DC_TYPE* dc;
		Helper::UtilityForCUDA::allocateDeviceMemory(dc, sizeInfo);

		int pixelNumInWindow = (radius * 2 + 1) * (radius * 2 + 1);

		//cx, dxåvéZ
		cu_calculateDC(sizeInfo, NULL, g, radius, pixelNumInWindow, eps2, dc);

		//medianåvéZ
		if (f->arrayLength == 1 && g->arrayLength == 1)
			FGMF_GPU_Or_ver0::cu_filter2D<int2, float2, 1, 1>(sizeInfo, NULL, radius, fRange, result, f, g, (float2*)(dc));
		else if (f->arrayLength == 1 && g->arrayLength == 3)
			FGMF_GPU_Or_ver0::cu_filter2D<int4, float4, 1, 3>(sizeInfo, NULL, radius, fRange, result, f, g, (float4*)(dc));
		else if (f->arrayLength == 3 && g->arrayLength == 1)
			FGMF_GPU_Or_ver0::cu_filter2D<int2, float2, 3, 1>(sizeInfo, NULL, radius, fRange, result, f, g, (float2*)(dc));
		else if (f->arrayLength == 3 && g->arrayLength == 3)
			FGMF_GPU_Or_ver0::cu_filter2D<int4, float4, 3, 3>(sizeInfo, NULL, radius, fRange, result, f, g, (float4*)(dc));

		//ÉÅÉÇÉääJï˙
		cudaFree(dc);
	}

	//Multichannel
	void WMF::filter2DGPU_MultiChannel(Helper::DeviceArray<int>*& f, Helper::DeviceArray<int>*& g, Helper::DeviceArray<int>*& result, int radius, float eps2, int fRange, Helper::SizeInfo& sizeInfo)
	{
		
		int n = g->arrayLength;
		Helper::DeviceArray<float>* dc = new Helper::DeviceArray<float>(n + 1, sizeInfo);

		int pixelNumInWindow = (radius * 2 + 1) * (radius * 2 + 1);

		//cx, dxåvéZ
		cu_calculateDCx(sizeInfo, NULL, g, radius, pixelNumInWindow, eps2, dc);

		//medianåvéZ
		FGMF_GPU_Or_ver0::cu_filter2D_Multichannel(sizeInfo, NULL, radius, fRange, result, f, g, dc);

		//ÉÅÉÇÉääJï˙
		delete dc;
	}


}