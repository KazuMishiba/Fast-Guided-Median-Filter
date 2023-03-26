#include "FGMF_GPU_Or_ver2.h"
#include "FGMF_GPU_Or_ver2.cuh"

namespace FGMF_GPU_Or_ver2
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
		return wmf.apply_2d_filter();
	}


	cv::Mat WMF::apply_2d_filter()
	{
		std::chrono::system_clock::time_point start, end;
		start = std::chrono::system_clock::now();

		if (channelNum_g_ == 1)
		{
			filter2DGPU_G1(f_device_, g_device_, result_device_, radius_, eps2_, fRange_, sizeInfo_);
		}
		else if (channelNum_g_ == 3)
		{
			//filter2DGPU_G3(f_device_, g_device_, result_device_, radius_, eps2_, fRange_, sizeInfo_);
		}


		end = std::chrono::system_clock::now();
		double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Prop Win (after refactoring without data transfer): ";
		std::cout << time << " [ms] / " << 1 << "[times]" << std::endl;

		return Helper::UtilityForCUDA::downloadLinearArrayAsMat(result_device_, sizeInfo_);
	}


	//////////////////////////////////////////////////////////////////////
	//2D
	//I1G1
	void WMF::filter2DGPU_G1(Helper::DeviceArray<int>*& f, Helper::DeviceArray<int>*& g, Helper::DeviceArray<int>*& result, int radius, float eps2, int fRange, Helper::SizeInfo& sizeInfo)
	{
		float2* dc;
		int2* sumG, * temp;
		Helper::UtilityForCUDA::allocateDeviceMemory(dc, sizeInfo);
		Helper::UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo);
		Helper::UtilityForCUDA::allocateDeviceMemory(temp, sizeInfo);
		int pixelNumInWindow = (radius * 2 + 1) * (radius * 2 + 1);

		std::chrono::system_clock::time_point start, end;
		start = std::chrono::system_clock::now();

		//cx, dxåvéZ
		FGMF_GPU_Or_ver2::cu_calculateDC(sizeInfo, NULL, g->host[0], radius, pixelNumInWindow, eps2, dc, sumG, temp);

		end = std::chrono::system_clock::now();
		double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Prop Win (after refactoring without data transfer dc calculation): ";
		std::cout << time << " [ms] / " << 1 << "[times]" << std::endl;

		start = std::chrono::system_clock::now();

		//medianåvéZ
		if (f->arrayLength == 1)
		{
			FGMF_GPU_Or_ver2::cu_filter2DF1<int2, float2, 2>(sizeInfo, NULL, radius, fRange,result, f, g, dc);
		}
		else if (f->arrayLength == 3)
		{
			//FGMF_GPU_Or_ver2::cu_filter2DF3<int2, float2, 2>(sizeInfo, NULL, radius, fRange, result, f, g, dc);
		}

		end = std::chrono::system_clock::now();
		time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Prop Win (after refactoring without data transfer filtering): ";
		std::cout << time << " [ms] / " << 1 << "[times]" << std::endl;
		//ÉÅÉÇÉääJï˙
		cudaFree(dc);
		cudaFree(sumG);
		cudaFree(temp);
		//cudaDeviceSynchronize();
	}

#if 0
	void WMF::filter2DGPU_G3(Helper::DeviceArray<int>*& f, Helper::DeviceArray<int>*& g, Helper::DeviceArray<int>*& result, int radius, float eps2, int fRange, Helper::SizeInfo& sizeInfo)
	{
		float4* dc;
		Helper::UtilityForCUDA::allocateDeviceMemory(dc, sizeInfo);
		Helper::DeviceArray<int>* sumG = new Helper::DeviceArray<int>(3, sizeInfo);
		Helper::DeviceArray<int>* tempG = new Helper::DeviceArray<int>(3, sizeInfo);
		Helper::DeviceArray<int>* sumGG = new Helper::DeviceArray<int>(6, sizeInfo);
		Helper::DeviceArray<int>* tempGG = new Helper::DeviceArray<int>(6, sizeInfo);
		int pixelNumInWindow = (radius * 2 + 1) * (radius * 2 + 1);
		//cx, dxåvéZ
		FGMF_GPU_Or_ver2::cu_calculateDC3(sizeInfo, NULL, g, radius, pixelNumInWindow, eps2, dc, sumG, sumGG, tempG, tempGG);
		//medianåvéZ
		if (f->arrayLength == 1)
		{
			FGMF_GPU_Or_ver2::cu_filter2DF1<int4, float4, 4>(sizeInfo, NULL, radius, fRange, result, f, g, dc);
		}
		else if (f->arrayLength == 3)
		{
			//FGMF_GPU_Or_ver2::cu_filter2DF3<int4, float4, 4>(sizeInfo, NULL, radius, fRange, result, f, g, dc);
		}

		//ÉÅÉÇÉääJï˙
		cudaFree(dc);
		delete sumG;
		delete tempG;
		delete sumGG;
		delete tempGG;
	}
#endif

}