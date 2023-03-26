#include "FGMF_GPU_Or.h"
#include "FGMF_GPU_Or.cuh"

namespace FGMF_GPU_Or
{
	/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			cu_memoryInfo();
			if (abort) exit(code);
		}
	}*/

	void sizeinfoTest()
	{

		int width, height;
		size_t pitchF4, pitchF1, pitchF2, pitchUC1, pitchUC3, pitchI1, pitchI2, pitchI4;

		width = 98;
		height = 201;



		float4* dst_f4;
		gpuErrchk(cudaMallocPitch(&dst_f4, &pitchF4, width * sizeof(float4), height));
		cudaFree(dst_f4);
		//デバイスにリニアメモリをcudaMallocPitchで確保(float1)
		float* dst_f1;
		gpuErrchk(cudaMallocPitch(&dst_f1, &pitchF1, width * sizeof(float), height));
		cudaFree(dst_f1);
		//float2
		float2* dst_f2;
		gpuErrchk(cudaMallocPitch(&dst_f2, &pitchF2, width * sizeof(float2), height));
		cudaFree(dst_f2);
		//unsigned char*
		unsigned char* dst_uc1;
		gpuErrchk(cudaMallocPitch(&dst_uc1, &pitchUC1, width * sizeof(unsigned char), height));
		cudaFree(dst_uc1);
		//uchar3*
		uchar3* dst_uc3;
		gpuErrchk(cudaMallocPitch(&dst_uc3, &pitchUC3, width * sizeof(uchar3), height));
		cudaFree(dst_uc3);
		//int*
		int* dst_i1;
		gpuErrchk(cudaMallocPitch(&dst_i1, &pitchI1, width * sizeof(int), height));
		cudaFree(dst_i1);
		//int2*
		int2* dst_i2;
		gpuErrchk(cudaMallocPitch(&dst_i2, &pitchI2, width * sizeof(int2), height));
		cudaFree(dst_i2);
		//int4*
		int4* dst_i4;
		gpuErrchk(cudaMallocPitch(&dst_i4, &pitchI4, width * sizeof(int4), height));
		cudaFree(dst_i4);


		std::cout << "F4:" << pitchF4 << std::endl;




	}
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange)
	{
		WMF wmf = WMF(f_img, g_img, radius, eps2, fRange);
		return wmf.apply_2d_filter();
	}

	cv::Mat WMF::apply_2d_filter()
	{
		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);

		//g_imgをデバイスに転送
		Helper::UtilityForCUDA::allocateDeviceMemory(g_device_, sizeInfo_);
		Helper::UtilityForCUDA::uploadMatToDevice(g_img_, g_device_, sizeInfo_, stream1);//stream使ったほうがいいか？

		/*
		UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo_);
		UtilityForCUDA::allocateDeviceMemory(temp, sizeInfo_);
		int pixelNumInWindow = (radius_ * 2 + 1) * (radius * 2 + 1);
		//cx, dx計算
		cu_calculateCxDxFromG(sizeInfo_, NULL, G, radius, pixelNumInWindow, eps2_, cxdx, sumG, temp);
		//median計算
		cu_filter2D(sizeInfo, NULL, radius, Imax, result, I, G, cxdx);
		//メモリ開放
		cudaFree(cxdx);
		cudaFree(sumG);
		cudaFree(temp);
		*/
		
		int pixelNumInWindow = (radius_ * 2 + 1) * (radius_ * 2 + 1);
		cv::Mat result;
		//FとGのタイプで分岐
		if (channelNum_f_ == 1 && channelNum_g_ == 1) {
			//f,gのアップロード
			// f,gを結合
			cv::Mat fg_img;
			cv::Mat channels[] = { f_img_, g_img_ };
			cv::merge(channels, 2, fg_img);


			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();



			//dc確保
			Helper::UtilityForCUDA::allocateDeviceMemory(dc_device_, sizeInfo_);
			int2* sumG, * temp;
			Helper::UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo_);
			Helper::UtilityForCUDA::allocateDeviceMemory(temp, sizeInfo_);

			Helper::UtilityForCUDA::allocateDeviceMemory(fg_device_, fg_img, sizeInfo_, stream2);
			//cx, dx計算
			cu_calculateDC(sizeInfo_, stream1, g_device_, radius_, pixelNumInWindow, eps2_, dc_device_, sumG, temp);


			//結果の確保
			Helper::UtilityForCUDA::allocateDeviceMemory(result_device_, sizeInfo_);



			cudaFree(sumG);
			cudaFree(temp);


			
			cu_filter2D(sizeInfo_, NULL, radius_, fRange_, result_device_, fg_device_, dc_device_);
			result = Helper::UtilityForCUDA::downloadLinearArrayAsMat(result_device_, sizeInfo_);
			
			cudaFree(dc_device_);
			cudaFree(result_device_);

			std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
			double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::cout << "in: ";
			std::cout << time << " [ms] / " << 1 << "[times]" << std::endl;

		}
#if 0
			cv::Mat result(M_, N_, CV_32SC1);
			filtering<Window_calc_cd<int>, int, float>(f_img_, g_img_, result);
			return result;
		}
		else if (channelNum_f_ == 1 && channelNum_g_ == 3) {
			cv::Mat result(M_, N_, CV_32SC1);
			filtering<Window_calc_cd<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_img_, g_img_, result);
			return result;
		}
		else if (channelNum_f_ == 3 && channelNum_g_ == 1) {
			cv::Mat result;
			std::vector<cv::Mat> results = { cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1) };
			std::vector<cv::Mat> f_singles;
			cv::split(f_img_, f_singles);

			filtering<Window_calc_cd<int>, int, float>(f_singles[0], g_img_, results[0]);
			filtering<Window<int>, int, float>(f_singles[1], g_img_, results[1]);
			filtering<Window<int>, int, float>(f_singles[2], g_img_, results[2]);

			cv::merge(results, result);
			return result;
		}
		else if (channelNum_f_ == 3 && channelNum_g_ == 3) {
			cv::Mat result;
			std::vector<cv::Mat> results = { cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1) };
			std::vector<cv::Mat> f_singles;
			cv::split(f_img_, f_singles);

			filtering<Window_calc_cd<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[0], g_img_, results[0]);
			filtering<Window<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[1], g_img_, results[1]);
			filtering<Window<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[2], g_img_, results[2]);

			cv::merge(results, result);
			return result;
		}
#endif
		cudaFree(g_device_);
		return result;
	}
}
