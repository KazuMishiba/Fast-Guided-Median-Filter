#include "Helper.h"


namespace FGMF_GPU_Or
{
	//Common
	void cu_calculateSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* sumG, int2* temp);
	void cu_calculateSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G3, int radius, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG, Helper::DeviceArray<int>* tempG, Helper::DeviceArray<int>* tempGG);

	//For 2D
	void cu_calculateDC(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G, int radius, int pixelNumInWindow, float eps2, float2* dc);
	void cu_calculateDC(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G, int radius, int pixelNumInWindow, float eps2, float4* dc);
	void cu_calculateDCx(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* GX, int radius, int pixelNumInWindow, float eps2, Helper::DeviceArray<float>* dc);

	//For multidimensional data
	void cu_addSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int2* addSumG, int2* sumG);
	void cu_addSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* addSumG, Helper::DeviceArray<int>* addSumGG, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG);
	void cu_remSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int2* remSumG, int2* sumG);
	void cu_remSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* remSumG, Helper::DeviceArray<int>* remSumGG, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG);
	void cu_calculateDC(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int pixelNumInWindow, float eps2, float2* cxdx, int2* sumG);
	void cu_calculateDC3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G, int pixelNumInWindow, float eps2, float4* cxdx, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG);

}