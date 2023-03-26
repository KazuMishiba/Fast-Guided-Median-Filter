#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//#include "boxFilter.cuh"
#include "Helper.h"

//using namespace Helper;

namespace FGMF_GPU_Or
{
	void cu_filter2D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int2* fg, float2* dxcx);
	void cu_filter2D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int4* fg, float4* dxcx);

#if 0
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, int* g, float2* dxcx);

	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, DeviceArray<int>* g, float4* dxcx);
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, DeviceArray<int>* f, int* g, float2* dxcx);
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, float4* dxcx);
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, DeviceArray<int>* g, DeviceArray<float>* dxcx);

	void cu_filter2DMultiChannel(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, DeviceArray<int>* g, DeviceArray<float>* dxcx);
	void cu_filter2DMultiChannel(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, DeviceArray<float>* dxcx);


	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, std::vector<int*>f, std::vector<int*> g, float2* dxcx);
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, std::vector<int*>f, std::vector<DeviceArray<int>*> g, float4* dxcx);
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, std::vector<DeviceArray<int>*>f, std::vector<int*> g, float2* dxcx);
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, std::vector<DeviceArray<int>*>f, std::vector<DeviceArray<int>*> g, float4* dxcx);

	void cu_testBlockGrid();

	//refactoring—p
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, int* g, DeviceArray<float>* dxcx);


	//multi test—p
	//void cu_filter2DTest(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, float4* dxcx, DeviceArray<float>* dxcx2);

#endif

}