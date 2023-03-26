#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "boxFilter.cuh"
#include "Helper.h"

#include "common.h"

namespace FGMF_GPU_Or
{

	//2D
	void cu_calculateDC(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, float2* dxcx, int2* sumG, int2* temp);
	void cu_calculateDC3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G3, int radius, int pixelNumInWindow, float eps2, float4* dxcx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);
	void cu_calculateCxXDxFromG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* GX, int radius, int pixelNumInWindow, float eps2, DeviceArray<float>* dxcx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);

	void cu_calculateSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G3, int radius, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);
	void cu_updateSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, DeviceArray<int>* addSumG, DeviceArray<int>* addSumGG, DeviceArray<int>* remSumG, DeviceArray<int>* remSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);
	void cu_addSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, DeviceArray<int>* addSumG, DeviceArray<int>* addSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);
	void cu_remSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* remSumG, DeviceArray<int>* remSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG);
	void cu_calculateCx3Dx(Helper::SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, int pixelNumInWindow, float eps2, float4* dxcx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG);

	//‘½ŽŸŒ³—p
	void cu_calculateSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* sumG, int2* temp);
	void cu_updateSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* addSumG, int2* remSumG, int2* sumG, int2* temp);
	void cu_addSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* addSumG, int2* sumG, int2* temp);
	void cu_remSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int2* remSumG, int2* sumG);
	void cu_calculateCxDx(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, float2* dxcx, int2* sumG);




	void cu_calculateSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G3, int radius, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);



	//refactoring—p
	void cu_calculateCxDxFromG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, DeviceArray<float>* dxcx, int2* sumG, int2* temp);

}