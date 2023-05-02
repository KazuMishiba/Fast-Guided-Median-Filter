#include "FGMF_GPU_Or.cuh"

namespace FGMF_GPU_Or
{
	//Number of channels in g + 1
	__constant__ int channelNum_fg;
	//Number of channels in f
	__constant__ int channelNum_f;

	//Shared memory for histograms
	template<typename FG_TYPE>
	extern __shared__ FG_TYPE  sharedData[];

	
	// de_calculateHcum
	__device__ inline float
		de_calculateHcum(float2& dxcx, int2& W_fg_cum)
	{
		return dxcx.x * W_fg_cum.x + dxcx.y * W_fg_cum.y;
	}
	__device__ inline float
		de_calculateHcum(float4& dxcx, int4& W_fg_cum)
	{
		return dxcx.x * W_fg_cum.x + dxcx.y * W_fg_cum.y + dxcx.z * W_fg_cum.z + dxcx.w * W_fg_cum.w;
	}
	__device__ inline float
		de_calculateHcum(float& dxcx, int& W_fg_cum)
	{
		float* pdxcx = &dxcx;
		int* pW_fg_cum = &W_fg_cum;
		float h = 0.0f;
		for (int i = 0; i < channelNum_fg; i++)
			h += pdxcx[i] * pW_fg_cum[i];
		return h;
	}

	// de_update_W_fg
	__device__ inline float
		de_update_W_fg(float2& dxcx, int2* W_FG, int2& W_fg_cum, int W_k, int sign)
	{
		W_fg_cum.x += W_FG[W_k].x * sign;
		W_fg_cum.y += W_FG[W_k].y * sign;
		return de_calculateHcum(dxcx, W_fg_cum);
	}
	__device__ inline float
		de_update_W_fg(float4& dxcx, int4* W_FG, int4& W_fg_cum, int W_k, int sign)
	{
		W_fg_cum.x += W_FG[W_k].x * sign;
		W_fg_cum.y += W_FG[W_k].y * sign;
		W_fg_cum.z += W_FG[W_k].z * sign;
		W_fg_cum.w += W_FG[W_k].w * sign;
		return de_calculateHcum(dxcx, W_fg_cum);
	}
	__device__ inline float
		de_update_W_fg(float& dxcx, int* W_FG, int& W_fg_cum, int W_k, int sign)
	{
		for (int i = 0; i < channelNum_fg; i++)
			(&W_fg_cum)[i] += W_FG[W_k * channelNum_fg + i] * sign;
		return de_calculateHcum(dxcx, W_fg_cum);
	}

	// Search weighted median
	// 
	// Implemented a version of the program that avoids conditional branches (if-statements), 
	// due to their potential to cause computational delay in CUDA, where different threads 
	// may wait for each other when branching divergently. 
	// This allows us to achieve the same results more efficiently.
	template<typename FG_TYPE, typename DC_TYPE>
	__device__ inline int
		de_searchWeightedMedian(DC_TYPE& dxcx, FG_TYPE* W_FG, FG_TYPE* W_fg_cum, int& index)
	{
		float h = de_calculateHcum(dxcx, *W_fg_cum);
		int flagA = h < 0.5f;
		int flagB = flagA - 1;
		int sign = flagA * 2 - 1;
		while (true)
		{
			index += flagA;
			//if(histogram[index].x)
			{
				de_update_W_fg(dxcx, W_FG, *W_fg_cum, index, sign);
				h = de_calculateHcum(dxcx, *W_fg_cum);
				if ((h >= 0.5f) == flagA)
				{
					int result_center = index;
					index += flagB;
					return result_center;
				}
			}
			index += flagB;
		}
	}




	__device__ inline void
		de_addPixelToWindow(cudaTextureObject_t* texF, cudaTextureObject_t* texG, int s, int t, int2* W_FG, int2* W_fg_cum, int W_k)
	{
		int f = tex2D<int>(*texF, s, t);
		int g = tex2D<int>(*texG, s, t);
		atomicAdd(&W_FG[f].x, 1);
		atomicAdd(&W_FG[f].y, g);
		if (f <= W_k)
		{
			atomicAdd(&(*W_fg_cum).x, 1);
			atomicAdd(&(*W_fg_cum).y, g);
		}
	}

	__device__ inline void
		de_removePixelFromWindow(cudaTextureObject_t* texF, cudaTextureObject_t* texG, int s, int t, int2* W_FG, int2* W_fg_cum, int W_k)
	{
		int f = tex2D<int>(*texF, s, t);
		int g = tex2D<int>(*texG, s, t);
		atomicSub(&W_FG[f].x, 1);
		atomicSub(&W_FG[f].y, g);
		if (f <= W_k)
		{
			atomicSub(&(*W_fg_cum).x, 1);
			atomicSub(&(*W_fg_cum).y, g);
		}
	}

	__device__ inline void
		de_addPixelToWindow(cudaTextureObject_t* texF, cudaTextureObject_t* texG, int s, int t, int4* W_FG, int4* W_fg_cum, int W_k)
	{
		int f = tex2D<int>(*texF, s, t);
		int g0 = tex2D<int>(texG[0], s, t);
		int g1 = tex2D<int>(texG[1], s, t);
		int g2 = tex2D<int>(texG[2], s, t);
		atomicAdd(&W_FG[f].x, 1);
		atomicAdd(&W_FG[f].y, g0);
		atomicAdd(&W_FG[f].z, g1);
		atomicAdd(&W_FG[f].w, g2);
		if (f <= W_k)
		{
			atomicAdd(&(*W_fg_cum).x, 1);
			atomicAdd(&(*W_fg_cum).y, g0);
			atomicAdd(&(*W_fg_cum).z, g1);
			atomicAdd(&(*W_fg_cum).w, g2);
		}
	}

	__device__ inline void
		de_removePixelFromWindow(cudaTextureObject_t* texF, cudaTextureObject_t* texG, int s, int t, int4* W_FG, int4* W_fg_cum, int W_k)
	{
		int f = tex2D<int>(*texF, s, t);
		int g0 = tex2D<int>(texG[0], s, t);
		int g1 = tex2D<int>(texG[1], s, t);
		int g2 = tex2D<int>(texG[2], s, t);
		atomicSub(&W_FG[f].x, 1);
		atomicSub(&W_FG[f].y, g0);
		atomicSub(&W_FG[f].z, g1);
		atomicSub(&W_FG[f].w, g2);
		if (f <= W_k)
		{
			atomicSub(&(*W_fg_cum).x, 1);
			atomicSub(&(*W_fg_cum).y, g0);
			atomicSub(&(*W_fg_cum).z, g1);
			atomicSub(&(*W_fg_cum).w, g2);
		}
	}

	__device__ inline void
		de_addPixelToWindow(cudaTextureObject_t* texF, cudaTextureObject_t* texG, int s, int t, int* W_FG, int* W_fg_cum, int W_k)
	{
		int f, g;
		f = tex2D<int>(*texF, s, t);
		atomicAdd(&W_FG[f * channelNum_fg], 1);

		if (f <= W_k)
			atomicAdd(&W_fg_cum[0], 1);

		for (int i = 0; i < channelNum_fg - 1; i++)
		{
			g = tex2D<int>(texG[i], s, t);
			atomicAdd(&W_FG[f * channelNum_fg + i + 1], g);
			if (f <= W_k)
				atomicAdd(&W_fg_cum[i + 1], g);
		}
	}

	__device__ inline void
		de_removePixelFromWindow(cudaTextureObject_t* texF, cudaTextureObject_t* texG, int s, int t, int* W_FG, int* W_fg_cum, int W_k)
	{
		int f, g;
		f = tex2D<int>(*texF, s, t);
		atomicSub(&W_FG[f * channelNum_fg], 1);

		if (f <= W_k)
			atomicSub(&W_fg_cum[0], 1);

		for (int i = 0; i < channelNum_fg - 1; i++)
		{
			g = tex2D<int>(texG[i], s, t);
			atomicSub(&W_FG[f * channelNum_fg + i + 1], g);
			if (f <= W_k)
				atomicSub(&W_fg_cum[i + 1], g);
		}
	}



	// For grayscale and color image
	template<typename FG_TYPE, typename DC_TYPE>
	__global__ void
		de_filter2d(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, DC_TYPE* dc, size_t pitchI1, size_t pitchF_channelNum_fg)
	{
		// s-coordinate of the pixel to be processed by this thread block
		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		// s-coordinate of the pixel to be processed by this thread
		int s_handle = s_center + tid - radius;

		// Initialization
		FG_TYPE* W_fg_cum = &sharedData<FG_TYPE>[0];
		FG_TYPE* W_FG = &sharedData<FG_TYPE>[1];
		__shared__ int W_k;

		// Only the central thread is executed
		if (tid == radius)
		{
			// Initialize window
			memset(sharedData<FG_TYPE>, 0, sizeof(FG_TYPE) * (fRange + 1));
			W_k = tex2D<int>(*texF, s_center, 0);//current index
		}
		__syncthreads();

		// For the pixel in the first row
		// Histogram construction
		for (int t = -radius; t <= radius; t++)
			de_addPixelToWindow(texF, texG, s_handle, t, W_FG, W_fg_cum, W_k);
		__syncthreads();
		// Median search
		if (tid == radius)
		{
			DC_TYPE dxcx = *((DC_TYPE*)((char*)dc) + s_center);
			*((int*)((char*)result_center) + s_center) = de_searchWeightedMedian(dxcx, W_FG, W_fg_cum, W_k);
		}

		// For pixels after the second row
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			// Update window using sliding window approach
			de_addPixelToWindow(texF, texG, s_handle, t + radius, W_FG, W_fg_cum, W_k);
			de_removePixelFromWindow(texF, texG, s_handle, t - radius - 1, W_FG, W_fg_cum, W_k);

			__syncthreads();
			if (tid == radius)
			{
				DC_TYPE dxcx = *((DC_TYPE*)((char*)dc + t * pitchF_channelNum_fg) + s_center);
				// Search weighted median
				*((int*)((char*)result_center + t * pitchI1) + s_center) = de_searchWeightedMedian(dxcx, W_FG, W_fg_cum, W_k);
			}
		}
	}


	


	// For multichannel guide image
	__global__ void
		de_filter2D_multichannel(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, float** dc, size_t pitchI1, size_t pitchF1)
	{
		// s-coordinate of the pixel to be processed by this thread block
		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		// s-coordinate of the pixel to be processed by this thread
		int s_handle = s_center + tid - radius;

		// Initialization
		int* W_fg_cum = &sharedData<int>[0];
		int* W_FG = &sharedData<int>[channelNum_fg];
		__shared__ int W_k;

		float* dxcx;

		// Only the central thread is executed
		if (tid == radius)
		{
			dxcx = new float[channelNum_fg];
			// Initialize window
			memset(sharedData<int>, 0, sizeof(int) * channelNum_fg * (fRange + 1));
			W_k = tex2D<int>(*texF, s_center, 0);//current index
		}
		__syncthreads();

		// For the pixel in the first row
		// Histogram construction
		for (int t = -radius; t <= radius; t++)
			de_addPixelToWindow(texF, texG, s_handle, t, W_FG, W_fg_cum, W_k);
		__syncthreads();
		// Median search
		if (tid == radius)
		{
			for (int i = 0; i < channelNum_fg; i++)
				dxcx[i] = *((float*)((char*)dc[i]) + s_center);
			*((int*)((char*)result_center) + s_center) = de_searchWeightedMedian(*dxcx, W_FG, W_fg_cum, W_k);
		}

		// For pixels after the second row
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			// Update window using sliding window approach
			de_addPixelToWindow(texF, texG, s_handle, t + radius, W_FG, W_fg_cum, W_k);
			de_removePixelFromWindow(texF, texG, s_handle, t - radius - 1, W_FG, W_fg_cum, W_k);

			__syncthreads();
			if (tid == radius)
			{
				for (int i = 0; i < channelNum_fg; i++)
					dxcx[i] = *((float*)((char*)dc[i] + t * pitchF1) + s_center);
				// Search weighted median
				*((int*)((char*)result_center + t * pitchI1) + s_center) = de_searchWeightedMedian(*dxcx, W_FG, W_fg_cum, W_k);
			}
		}

		if (tid == radius)
			delete dxcx;
	}



	// For grayscale and color image
	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_F, int CHANNELNUM_G>
	void cu_filter2D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, DC_TYPE* dc)
	{
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int k = CHANNELNUM_G + 1;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

		for (int c = 0; c < CHANNELNUM_F; c++)
		{
			de_filter2d<FG_TYPE, DC_TYPE> << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* k, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[c], &(texF.device[c]), texG.device, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<DC_TYPE>());
		}
	}
	template void cu_filter2D<int2, float2, 1, 1>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, Helper::DeviceArray<int>*, Helper::DeviceArray<int>*, float2*);
	template void cu_filter2D<int4, float4, 1, 3>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, Helper::DeviceArray<int>*, Helper::DeviceArray<int>*, float4*);
	template void cu_filter2D<int2, float2, 3, 1>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, Helper::DeviceArray<int>*, Helper::DeviceArray<int>*, float2*);
	template void cu_filter2D<int4, float4, 3, 3>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, Helper::DeviceArray<int>*, Helper::DeviceArray<int>*, float4*);

	// For multichannel guide image
	void cu_filter2D_Multichannel(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* dc)
	{
		int channelNum_f = f->arrayLength;
		int channelNum_g = g->arrayLength;

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		const int k = channelNum_g + 1;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

		for (int c = 0; c < channelNum_f; c++)
		{
			de_filter2D_multichannel << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* k, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[c], &(texF.device[c]), texG.device, dc->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
		}
	}





	// For grayscale and color multidimensional image
	template<typename FG_TYPE, typename DC_TYPE>
	__global__ void
		de_filterNd(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfData, int currentChannel, DC_TYPE* dc, size_t pitchI1, size_t pitchF_channelNum_fg)
	{
		// s-coordinate of the pixel to be processed by this thread block
		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		// s-coordinate of the pixel to be processed by this thread
		int s_handle = s_center + tid - radius;

		// Initialization
		FG_TYPE* W_fg_cum = &sharedData<FG_TYPE>[0];
		FG_TYPE* W_FG = &sharedData<FG_TYPE>[1];
		int channelNum_g = channelNum_fg - 1;
		__shared__ int W_k;

		// Only the central thread is executed
		if (tid == radius)
		{
			// Initialize window
			memset(sharedData<FG_TYPE>, 0, sizeof(FG_TYPE) * (fRange + 1));
			W_k = tex2D<int>(texF[(numOfData / 2) * channelNum_f + currentChannel], s_center, 0);//current index
		}
		__syncthreads();

		// For the pixel in the first row
		// Histogram construction
		for (int i = 0; i < numOfData; i++)
			for (int t = -radius; t <= radius; t++)
				de_addPixelToWindow(&texF[i * channelNum_f + currentChannel], &texG[i * channelNum_g], s_handle, t, W_FG, W_fg_cum, W_k);

		__syncthreads();
		// Median search
		if (tid == radius)
		{
			DC_TYPE dxcx = *((DC_TYPE*)((char*)dc) + s_center);
			*((int*)((char*)result_center) + s_center) = de_searchWeightedMedian(dxcx, W_FG, W_fg_cum, W_k);
		}

		// For pixels after the second row
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			// Update window using sliding window approach
			for (int i = 0; i < numOfData; i++) {
				de_addPixelToWindow(&texF[i * channelNum_f + currentChannel], &texG[i * channelNum_g], s_handle, t + radius, W_FG, W_fg_cum, W_k);
				de_removePixelFromWindow(&texF[i * channelNum_f + currentChannel], &texG[i * channelNum_g], s_handle, t - radius - 1, W_FG, W_fg_cum, W_k);
			}

			__syncthreads();
			if (tid == radius)
			{
				DC_TYPE dxcx = *((DC_TYPE*)((char*)dc + t * pitchF_channelNum_fg) + s_center);
				// Search weighted median
				*((int*)((char*)result_center + t * pitchI1) + s_center) = de_searchWeightedMedian(dxcx, W_FG, W_fg_cum, W_k);
			}
		}
	}





	// For grayscale and color multidimensional image
	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_F, int CHANNELNUM_G>
	void cu_filterND(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius2D, int fRange, Helper::DeviceArray<int>*& result_center, std::vector<Helper::DeviceArray<int>*> f, std::vector<Helper::DeviceArray<int>*> g, DC_TYPE* dc)
	{
		int blockSize = radius2D * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int k = CHANNELNUM_G + 1;
		constexpr int k2 = CHANNELNUM_F;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));
		cudaMemcpyToSymbol(channelNum_f, &k2, sizeof(int));

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;

		int numOfData = f.size();

		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

		for (int currentChannel = 0; currentChannel < CHANNELNUM_F; currentChannel++)
		{
			de_filterNd<FG_TYPE, DC_TYPE> << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* k, stream >> > (sizeInfo.width_, sizeInfo.height_, radius2D, fRange, result_center->host[currentChannel], texF.device, texG.device, numOfData, currentChannel, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<DC_TYPE>());
		}
	}
	template void cu_filterND<int2, float2, 1, 1>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, std::vector<Helper::DeviceArray<int>*>, std::vector<Helper::DeviceArray<int>*>, float2*);
	template void cu_filterND<int4, float4, 1, 3>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, std::vector<Helper::DeviceArray<int>*>, std::vector<Helper::DeviceArray<int>*>, float4*);
	template void cu_filterND<int2, float2, 3, 1>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, std::vector<Helper::DeviceArray<int>*>, std::vector<Helper::DeviceArray<int>*>, float2*);
	template void cu_filterND<int4, float4, 3, 3>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, std::vector<Helper::DeviceArray<int>*>, std::vector<Helper::DeviceArray<int>*>, float4*);

}