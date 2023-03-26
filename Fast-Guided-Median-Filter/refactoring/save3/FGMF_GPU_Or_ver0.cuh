#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//#include "boxFilter.cuh"
#include "Utility.h"
#include "Utility.cuh"


namespace FGMF_GPU_Or_ver0
{

	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx);

	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, DeviceArray<int>* g, float4* cxdx);
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, int* g, float2* cxdx);
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, float4* cxdx);
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, DeviceArray<int>* g, DeviceArray<float>* cxdx);

	void cu_filter2DMultiChannel(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, DeviceArray<int>* g, DeviceArray<float>* cxdx);
	void cu_filter2DMultiChannel(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, DeviceArray<float>* cxdx);


	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, std::vector<int*>f, std::vector<int*> g, float2* cxdx);
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, std::vector<int*>f, std::vector<DeviceArray<int>*> g, float4* cxdx);
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, std::vector<DeviceArray<int>*>f, std::vector<int*> g, float2* cxdx);
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, std::vector<DeviceArray<int>*>f, std::vector<DeviceArray<int>*> g, float4* cxdx);

	void cu_testBlockGrid();

	//refactoring—p
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, DeviceArray<float>* cxdx);


	//multi test—p
	//void cu_filter2DTest(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, float4* cxdx, DeviceArray<float>* cxdx2);
}