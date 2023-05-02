#pragma once
#include "Helper.h"


namespace FGMF_GPU_Or
{
	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_F, int CHANNELNUM_G>
	void cu_filter2D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, DC_TYPE* dc);

	void cu_filter2D_Multichannel(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* dc);

	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_F, int CHANNELNUM_G>
	void cu_filterND(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius2D, int fRange, Helper::DeviceArray<int>*& result_center, std::vector<Helper::DeviceArray<int>*> f, std::vector<Helper::DeviceArray<int>*> g, DC_TYPE* dc);

}