#pragma one
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//#include "boxFilter.cuh"
#include "Helper.h"


namespace FGMF_GPU_Or_ver0
{
	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_F, int CHANNELNUM_G>
	void cu_filter2D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, DC_TYPE* dc);

	void cu_filter2D_Multichannel(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* dc);

#if 0
	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_FG>
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, DC_TYPE* dc);

	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_FG>
	void cu_filter2DF3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, DC_TYPE* dc);

	//void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, float2* dc);

	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx);

	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, Helper::DeviceArray<int>* g, float4* cxdx);
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, Helper::DeviceArray<int>* result_center, Helper::DeviceArray<int>* f, int* g, float2* cxdx);
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, Helper::DeviceArray<int>* result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, float4* cxdx);
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* cxdx);

	void cu_filter2DMultiChannel(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* cxdx);
	void cu_filter2DMultiChannel(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, Helper::DeviceArray<int>* result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* cxdx);
#endif
#if 0
	void cu_filter3D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, std::vector<int*>f, std::vector<int*> g, float2* cxdx);
	void cu_filter3D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, std::vector<int*>f, std::vector<Helper::DeviceArray<int>*> g, float4* cxdx);
	void cu_filter3D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, Helper::DeviceArray<int>* result_center, std::vector<Helper::DeviceArray<int>*>f, std::vector<int*> g, float2* cxdx);
	void cu_filter3D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, Helper::DeviceArray<int>* result_center, std::vector<Helper::DeviceArray<int>*>f, std::vector<Helper::DeviceArray<int>*> g, float4* cxdx);

	void cu_testBlockGrid();

	//refactoring—p
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, Helper::DeviceArray<float>* cxdx);
#endif

	//multi test—p
	//void cu_filter2DTest(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, Helper::DeviceArray<int>* result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, float4* cxdx, Helper::DeviceArray<float>* cxdx2);
}