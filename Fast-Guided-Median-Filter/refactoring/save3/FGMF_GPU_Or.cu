#include "FGMF_GPU_Or.cuh"
#include<opencv2/opencv.hpp>
#include <vector>

namespace FGMF_GPU_Or
{
	__constant__ int channelNum_g_plusOne;


	/*
	* c,d��DeviceArray���g���ƂP�`�����l���̏ꍇ��15%���x���Ȃ�B�K�C�h�R�`�����l�����Ƃ�茰���ɒx���Ȃ邾�낤�B
	* �t�ɁAc,d�̓e�N�X�`���������Ƀt�F�b�`���Ă����΂悢�C������B
	* ����ɂ�菭�Ȃ��Ƃ��P�C�R�`�����l���̓e���v���[�g�g���Ă�����C������B
	* f,g���܂Ƃ߂ăe�N�X�`���������Ƀt�F�b�`����΂悢�̂ł́H
	* 
	* �Ƃ肠�����_�����e���̎��������t�@�N�^�����O
	* ��葬���e�X�g�����Ƃ��Ĉȉ�
	*	1-1�Ȃ�f,g��float2�Ƃ��ăe�N�X�`���������Ƀt�F�b�`
	*	1-3�܂���3-1�Ȃ�f,g��float4�Ƃ��ăe�N�X�`���������Ƀt�F�b�`
	*	3-3�Ȃ�f,g���ꂼ���float4�Ƃ��ăe�N�X�`���������Ƀt�F�b�`
	*	����3�`�����l���̏ꍇ�́A�J�[�l�����œ����ɂR�`�����l��������
	* 
	* �����J�n����l�����
	* f,g��ǂݍ��ށB
	* c,d�̌v�Z
	*	g���f�o�C�X�ɓ]��
	*	���̑��������m�ہ@������intX, floatX��size info�͎擾�\
	* �������]���i�����c,d�̌v�Z���ɕ���Ŏ��s�\�Ȃ͂��j
	*	f,g���Z�b�g�ɂ��ăf�o�C�X�ɓ]���i1-1,1-3,3-1�̂Ƃ��j
	*	f,g���ꂼ����f�o�C�X�ɓ]���i3-3,X-Y�̂Ƃ��j
	* �����l�̌v�Z
	*/

	//de_calculateHcum
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
		de_calculateHcum(float* dxcx, int* W_fg_cum)
	{
		float h = 0.0f;
		for (int i = 0; i < channelNum_g_plusOne; i++)
			h += dxcx[i] * W_fg_cum[i];
		return h;
	}

	//de_update_W_fg
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
		for (int i = 0; i < channelNum_g_plusOne; i++)
			(&W_fg_cum)[i] += W_FG[W_k * channelNum_g_plusOne + i] * sign;
		return de_calculateHcum(&dxcx, &W_fg_cum);
	}

	//de_findMedian
	template<typename FG_TYPE, typename DC_TYPE>
	__device__ inline int
		de_findMedian(DC_TYPE& dxcx, FG_TYPE* W_FG, FG_TYPE* W_fg_cum, int& index)
	{
		float h = de_calculateHcum(dxcx, *W_fg_cum);
		int flagA = h < 0.5f;
		int flag2 = flagA - 1;
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
					index += flag2;
					return result_center;
				}
			}
			index += flag2;
		}
	}
	


	__device__ inline void
		de_addPixelToWindow(cudaTextureObject_t texFG1, int s, int t, int2* W_FG, int2* W_fg_cum, int W_k)
	{
		int2 fg = tex2D<int2>(texFG1, s, t);
		atomicAdd(&W_FG[fg.x].x, 1);
		atomicAdd(&W_FG[fg.x].y, fg.y);
		if (fg.x <= W_k)
		{
			atomicAdd(&W_fg_cum->x, 1);
			atomicAdd(&W_fg_cum->y, fg.y);
		}
	}

	__device__ inline void
		de_removePixelFromWindow(cudaTextureObject_t texFG1, int s, int t, int2* W_FG, int2* W_fg_cum, int W_k)
	{
		int2 fg = tex2D<int2>(texFG1, s, t);
		atomicSub(&W_FG[fg.x].x, 1);
		atomicSub(&W_FG[fg.x].y, fg.y);
		if (fg.x <= W_k)
		{
			atomicSub(&W_fg_cum->x, 1);
			atomicSub(&W_fg_cum->y, fg.y);
		}
	}

	__device__ inline void
		de_addPixelToWindow(cudaTextureObject_t texFG3, int s, int t, int4* W_FG3, int4* W_fg3_cum, int W_k)
	{
		int4 fg3 = tex2D<int4>(texFG3, s, t);
		atomicAdd(&W_FG3[fg3.x].x, 1);
		atomicAdd(&W_FG3[fg3.x].y, fg3.y);
		atomicAdd(&W_FG3[fg3.x].z, fg3.z);
		atomicAdd(&W_FG3[fg3.x].w, fg3.w);
		if (fg3.x <= W_k)
		{
			atomicAdd(&W_fg3_cum->x, 1);
			atomicAdd(&W_fg3_cum->y, fg3.y);
			atomicAdd(&W_fg3_cum->z, fg3.z);
			atomicAdd(&W_fg3_cum->w, fg3.w);
		}
	}

	__device__ inline void
		de_removePixelFromWindow(cudaTextureObject_t texFG3, int s, int t, int4* W_FG3, int4* W_fg3_cum, int W_k)
	{
		int4 fg3 = tex2D<int4>(texFG3, s, t);
		atomicSub(&W_FG3[fg3.x].x, 1);
		atomicSub(&W_FG3[fg3.x].y, fg3.y);
		atomicSub(&W_FG3[fg3.x].z, fg3.z);
		atomicSub(&W_FG3[fg3.x].w, fg3.w);
		if (fg3.x <= W_k)
		{
			atomicSub(&W_fg3_cum->x, 1);
			atomicSub(&W_fg3_cum->y, fg3.y);
			atomicSub(&W_fg3_cum->z, fg3.z);
			atomicSub(&W_fg3_cum->w, fg3.w);
		}
	}

	template<int CHANNNEL_NUM_G, typename FG_TYPE, typename DC_TYPE>
	__global__ void
		de_filter2DFG1_3(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t texFG1_3, DC_TYPE* dxcx, size_t pitchI1, size_t pitchF_channelNum_g_plusOne)
	{
		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		int s_handle = s_center + tid - radius;

		extern __shared__ FG_TYPE sharedData[];
		FG_TYPE* W_fg_cum = &sharedData[0];
		//int* W_FG = &sdata[2];
		FG_TYPE* W_FG = &sharedData[CHANNNEL_NUM_G+ 1];
		__shared__ int W_k;

		int f[1];
		int g[CHANNNEL_NUM_G];

		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			//�q�X�g�O����������, W_gf������
			memset(sharedData, 0, sizeof(FG_TYPE) * (fRange + 1));
			//for (int i = 0; i < (fRange + 1) * (CHANNNEL_NUM_G + 1); i++)
			//	sharedData[i] = 0;
			FG_TYPE fg = tex2D<FG_TYPE>(texFG1_3, s_center, 0);
			W_k = fg.x;//current index
		}

		__syncthreads();
		//1�ڃq�X�g�O�����`��
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		for (int t = -radius; t <= radius; t++)
			de_addPixelToWindow(texFG1_3, s_handle, t, W_FG, W_fg_cum, W_k);

		__syncthreads();
		//1�s�ڂ̒����l�v�Z
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			DC_TYPE _dxcx = *((DC_TYPE*)((char*)dxcx) + s_center);
			*((int*)((char*)result_center) + s_center) = de_findMedian(_dxcx, W_FG, W_fg_cum, W_k);
		}

		//2�s�ڈȍ~�̏���
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			//�q�X�g�O�����ɒǉ�
			de_addPixelToWindow(texFG1_3, s_handle, t + radius, W_FG, W_fg_cum, W_k);
			//�q�X�g�O��������폜
			de_removePixelFromWindow(texFG1_3, s_handle, t - radius - 1, W_FG, W_fg_cum, W_k);

			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				DC_TYPE _dxcx = *((DC_TYPE*)((char*)dxcx + t * pitchF_channelNum_g_plusOne) + s_center);
				//�����l�v�Z
				*((int*)((char*)result_center + t * pitchI1) + s_center) = de_findMedian(_dxcx, W_FG, W_fg_cum, W_k);
			}
		}
	}



	/*
	//gX
	__global__ void
		de_filter2D(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t* texGX, float** CxDx, size_t pitchI1, size_t pitchF1)
	{
		int x = blockIdx.x;
		if (x >= width)
			return;
		int tid = threadIdx.x;
		if (tid >= radius * 2 + 1)
			return;
		int xPos = x + tid - radius;

		//shared memory��histogramX��fgXSumUpToIndex�ɕ�����
		extern __shared__ int  buffers[];
		int* fgXSumUpToIndex = &buffers[0];
		int* histogramX = &buffers[channelNum_g_plusOne];
		//g1,g3�ł�fgXSumUpToIndex��f,g�̏��Ŏ����������Adxcx�Ə��Ԃ����킹�邽�߂ɁAg,...,g f�̏��ɂ���
		//histogram�����l
		//histogram��1�����ɕ���ł��āA�ebin�ɂ��āAg,...,g,f �̏��ɕ���ł���
		int n = channelNum_g_plusOne - 1;

		__shared__ int index;
		int f;
		int* g = new int[n];

		float* dxcx;

		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			dxcx = new float[channelNum_g_plusOne];
			//�q�X�g�O����������
			for (int i = 0; i < fRange * channelNum_g_plusOne; i++)
			{
				histogramX[i] = 0;
			}
			for (int i = 0; i < channelNum_g_plusOne; i++)
			{
				fgXSumUpToIndex[i] = 0;
			}
			index = tex2D<int>(texF, x, 0);//current index
		}
		//thread����
		__syncthreads();

		//1�ڃq�X�g�O�����`��
		for (int yy = -radius; yy <= radius; yy++)
		{
			//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
			{
				f = tex2D<int>(texF, xPos, yy);
				for (int i = 0; i < n; i++)
					g[i] = tex2D<int>(texGX[i], xPos, yy);
				for (int i = 0; i < n; i++)
					atomicAdd(&histogramX[f * channelNum_g_plusOne + i], g[i]);
				atomicAdd(&histogramX[f * channelNum_g_plusOne + n], 1);
				if (f <= index)
				{
					for (int i = 0; i < n; i++)
						atomicAdd(&fgXSumUpToIndex[i], g[i]);
					atomicAdd(&fgXSumUpToIndex[n], 1);
				}

			}
		}
		//thread����
		__syncthreads();


		//1�s�ڂ̒����l�v�Z
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			for (int i = 0; i < channelNum_g_plusOne; i++)
				dxcx[i] = *((float*)((char*)CxDx[i]) + x);
//			*((int*)((char*)result_center) + x) = de_findMedian(dxcx, histogramX, fgXSumUpToIndex, index);
		}
		//thread����
		__syncthreads();

		//2�s�ڈȍ~�̏���
		for (int y = 1; y < height; y++)
		{
			//�q�X�g�O�����ɒǉ�
			f = tex2D<int>(texF, xPos, y + radius);
			for (int i = 0; i < n; i++)
				g[i] = tex2D<int>(texGX[i], xPos, y + radius);
			for (int i = 0; i < n; i++)
				atomicAdd(&histogramX[f * channelNum_g_plusOne + i], g[i]);
			atomicAdd(&histogramX[f * channelNum_g_plusOne + n], 1);
			if (f <= index)
			{
				for (int i = 0; i < n; i++)
					atomicAdd(&fgXSumUpToIndex[i], g[i]);
				atomicAdd(&fgXSumUpToIndex[n], 1);
			}
			//�q�X�g�O��������폜
			f = tex2D<int>(texF, xPos, y - radius - 1);
			for (int i = 0; i < n; i++)
				g[i] = tex2D<int>(texGX[i], xPos, y - radius - 1);
			for (int i = 0; i < n; i++)
				atomicSub(&histogramX[f * channelNum_g_plusOne + i], g[i]);
			atomicSub(&histogramX[f * channelNum_g_plusOne + n], 1);
			if (f <= index)
			{
				for (int i = 0; i < n; i++)
					atomicSub(&fgXSumUpToIndex[i], g[i]);
				atomicSub(&fgXSumUpToIndex[n], 1);
			}

			//thread����
			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				for (int i = 0; i < channelNum_g_plusOne; i++)
					dxcx[i] = *((float*)((char*)CxDx[i] + y * pitchF1) + x);
				//�����l�v�Z
//				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(dxcx, histogramX, fgXSumUpToIndex, index);
			}
			//thread����
			__syncthreads();
		}


		delete g;
		if (tid == radius)
		{
			delete dxcx;
		}
		__syncthreads();
	}
	*/
	//I1G1
	void cu_filter2D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int2* fg, float2* dxcx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		cudaTextureObject_t texFG;
		Helper::UtilityForCUDA::setLinearArrayToTexture(fg, texFG, sizeInfo, filterMode);

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int channelNum_g = 1;
		constexpr int _channelNum_g_plusOne = channelNum_g + 1;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &_channelNum_g_plusOne, sizeof(int));
		de_filter2DFG1_3<channelNum_g, int2, float2> << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* _channelNum_g_plusOne, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texFG, dxcx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

		cudaDestroyTextureObject(texFG);
	}
#if 0
	//I1G3
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int4* fg, float4* dxcx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		cudaTextureObject_t texFG;
		UtilityForCUDA::setLinearArrayToTexture(fg, texFG, sizeInfo, filterMode);

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int channelNum_g = 3;
		constexpr int _channelNum_g_plusOne = channelNum_g + 1;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &_channelNum_g_plusOne, sizeof(int));
		de_filter2DFG1_3<channelNum_g, int4, float4> << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* _channelNum_g_plusOne, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texFG, dxcx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());

		cudaDestroyTextureObject(texFG);
	}
#endif

#if 0
	//I1GX
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, DeviceArray<int>* g, DeviceArray<float>* dxcx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		cudaTextureObject_t texF;
		TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
		UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		int n = g->arrayLength;
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		//shared memory�́A�q�X�g�O�����{uptoindex�K�v�ŁA�q�X�g�O������fRange * (n+1)�Auptoindex��n+1�K�v
		de_filter2D << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* (n + 1), stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF, texG.device, dxcx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());

		cudaDestroyTextureObject(texF);
	}
	//I3G1
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, DeviceArray<int>* f, int* g, float2* dxcx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		cudaTextureObject_t texG;
		TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);
		UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int k = 2;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		//stream
		cudaStream_t streams[3];
		cudaStreamCreate(&streams[0]);
		cudaStreamCreate(&streams[1]);
		cudaStreamCreate(&streams[2]);

		for (int i = 0; i < 3; i++)
		{
			de_filter2D << <gridSizeX, blockSize, fRange * sizeof(int) * 2, streams[i] >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF.host[i], texG, dxcx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
		}
		//cudaDeviceSynchronize();
	}
	//I3G3
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, float4* dxcx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
		TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int k = 4;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		//stream
		cudaStream_t streams[3];
		cudaStreamCreate(&streams[0]);
		cudaStreamCreate(&streams[1]);
		cudaStreamCreate(&streams[2]);

		for (int i = 0; i < 3; i++)
		{
			de_filter2D << <gridSizeX, blockSize, fRange * sizeof(int) * 4, streams[i] >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF.host[i], texG.device, dxcx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
		}
		//cudaDeviceSynchronize();
	}

	//I1GX
	void cu_filter2DMultiChannel(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, DeviceArray<int>* g, DeviceArray<float>* dxcx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
		cudaTextureObject_t texF;
		UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		int n = g->arrayLength;
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		de_filter2D << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* (n + 1), NULL >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF, texG.device, dxcx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
		cudaDestroyTextureObject(texF);
		//cudaDeviceSynchronize();
	}
	//IXGY
	void cu_filter2DMultiChannel(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, DeviceArray<float>* dxcx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
		TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		int n = g->arrayLength;
		int m = f->arrayLength;
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		for (int i = 0; i < m; i++)
		{
			de_filter2D << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* (n + 1), NULL >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF.host[i], texG.device, dxcx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
		}
		//cudaDeviceSynchronize();
	}






	//////////////////////////////////////////////////
	//3D
	
	//G1
	__device__ inline int
		de_findMedian(float2& dxcx, int2* histogram, int2& W_gf, int& index)
	{
		float h = dxcx.x * W_gf.y + dxcx.y * W_gf.x;
		int flagA = h < 0.5f;
		int flag2 = flagA - 1;
		int sign = flagA * 2 - 1;
		while (true)
		{
			index += flagA;
			//if(histogram[index].x)
			{
				W_gf.x += histogram[index].x * sign;
				W_gf.y += histogram[index].y * sign;
				h = dxcx.x * W_gf.y + dxcx.y * W_gf.x;
				if ((h >= 0.5f) == flagA)
				{
					int result_center = index;
					index += flag2;
					return result_center;
				}
			}
			index += flag2;
		}
	}
	__device__ inline int
		de_findMedian(float4& dxcx, int4* histogram, int4& W_gf, int& index)
	{
		float h = dxcx.x * W_gf.y + dxcx.y * W_gf.z + dxcx.z * W_gf.w + dxcx.w * W_gf.x;
		int flagA = h < 0.5f;
		int flag2 = flagA - 1;
		int sign = flagA * 2 - 1;
		while (true)
		{
			index += flagA;
			//if(histogram[index].x)
			{
				W_gf.x += histogram[index].x * sign;
				W_gf.y += histogram[index].y * sign;
				W_gf.z += histogram[index].z * sign;
				W_gf.w += histogram[index].w * sign;
				h = dxcx.x * W_gf.y + dxcx.y * W_gf.z + dxcx.z * W_gf.w + dxcx.w * W_gf.x;
				if ((h >= 0.5f) == flagA)
				{
					int result_center = index;
					index += flag2;
					return result_center;
				}
			}
			index += flag2;
		}
	}




	//����͔C�ӂ̎����ł��g�������i�|�C���^���K�؂Ȃ�j
	//g1
	__global__ void
		de_filter3D(int width_, int height_, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float2* CxDx, size_t pitchI1, size_t pitchF2)
	{
		
		int x = blockIdx.x;
		if (x >= width_)
			return;
		int tid = threadIdx.x;
		if (tid >= radius * 2 + 1)
			return;
		int xPos = x + tid - radius;

		extern __shared__ int2 histogram2[];

		__shared__ int index;
		__shared__ int2 W_gf;//index�ȉ��a

		int f, g;

		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			//�q�X�g�O����������
			for (int i = 0; i < fRange; i++)
			{
				histogram2[i] = make_int2(0, 0);
			}
			W_gf = make_int2(0, 0);
			index = tex2D<int>(texF[numOfFrames / 2], x, 0);//current index �{���͏������S�t���[��index���w�肵����(���ꂾ�ƒ[�̂Ƃ��ɁA�������S�ł͂Ȃ��Ȃ�)
		}
		//thread����
		__syncthreads();

		//1�ڃq�X�g�O�����`��
		for (int k = 0; k < numOfFrames; k++)
		{
			for (int yy = -radius; yy <= radius; yy++)
			{
				//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
				{
					f = tex2D<int>(texF[k], xPos, yy);
					g = tex2D<int>(texG[k], xPos, yy);
					atomicAdd(&histogram2[f].x, 1);
					atomicAdd(&histogram2[f].y, g);
					if (f <= index)
					{
						atomicAdd(&W_gf.x, 1);
						atomicAdd(&W_gf.y, g);
					}
				}
			}
		}
		//thread����
		__syncthreads();
		//1�s�ڂ̒����l�v�Z
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			float2 dxcx = *((float2*)((char*)CxDx) + x);
			*((int*)((char*)result_center) + x) = de_findMedian(dxcx, histogram2, W_gf, index);
		}
		//thread����
		__syncthreads();

		//2�s�ڈȍ~�̏���
		for (int y = 1; y < height_; y++)
		{
			for (int k = 0; k < numOfFrames; k++)
			{
				//�q�X�g�O�����ɒǉ�
				f = tex2D<int>(texF[k], xPos, y + radius);
				g = tex2D<int>(texG[k], xPos, y + radius);
				atomicAdd(&histogram2[f].x, 1);
				atomicAdd(&histogram2[f].y, g);
				if (f <= index)
				{
					atomicAdd(&W_gf.x, 1);
					atomicAdd(&W_gf.y, g);
				}
				//�q�X�g�O��������폜
				f = tex2D<int>(texF[k], xPos, y - radius - 1);
				g = tex2D<int>(texG[k], xPos, y - radius - 1);
				atomicSub(&histogram2[f].x, 1);
				atomicSub(&histogram2[f].y, g);
				if (f <= index)
				{
					atomicSub(&W_gf.x, 1);
					atomicSub(&W_gf.y, g);
				}
			}

			//thread����
			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				float2 dxcx = *((float2*)((char*)CxDx + y * pitchF2) + x);
				//�����l�v�Z
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(dxcx, histogram2, W_gf, index);
			}
			//thread����
			__syncthreads();
		}
	}

	//g3
	__global__ void
		de_filter3D(int width_, int height_, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float4* CxDx, size_t pitchI1, size_t pitchF4)
	{
		int x = blockIdx.x;
		if (x >= width_)
			return;
		int tid = threadIdx.x;
		if (tid >= radius * 2 + 1)
			return;
		int xPos = x + tid - radius;

		extern __shared__ int4 histogram4[];
		__shared__ int index;
		__shared__ int4 W_gf;//index�ȉ��a

		int f;
		int g[3];

		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			//�q�X�g�O����������
			for (int i = 0; i < fRange; i++)
			{
				histogram4[i] = make_int4(0, 0, 0, 0);
			}
			W_gf = make_int4(0, 0, 0, 0);
			index = tex2D<int>(texF[numOfFrames / 2], x, 0);//current index �{���͏������S�t���[��index���w�肵����(���ꂾ�ƒ[�̂Ƃ��ɁA�������S�ł͂Ȃ��Ȃ�)
		}
		//thread����
		__syncthreads();

		//1�ڃq�X�g�O�����`��
		for (int k = 0, n = 0; k < numOfFrames; k++, n += 3)
		{
			for (int yy = -radius; yy <= radius; yy++)
			{
				//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
				{
					f = tex2D<int>(texF[k], xPos, yy);
					g[0] = tex2D<int>(texG[n], xPos, yy);
					g[1] = tex2D<int>(texG[n + 1], xPos, yy);
					g[2] = tex2D<int>(texG[n + 2], xPos, yy);
					atomicAdd(&histogram4[f].x, 1);
					atomicAdd(&histogram4[f].y, g[0]);
					atomicAdd(&histogram4[f].z, g[1]);
					atomicAdd(&histogram4[f].w, g[2]);
					if (f <= index)
					{
						atomicAdd(&W_gf.x, 1);
						atomicAdd(&W_gf.y, g[0]);
						atomicAdd(&W_gf.z, g[1]);
						atomicAdd(&W_gf.w, g[2]);
					}
				}
			}
		}
		//thread����
		__syncthreads();
		//1�s�ڂ̒����l�v�Z
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			float4 dxcx = *((float4*)((char*)CxDx) + x);
			*((int*)((char*)result_center) + x) = de_findMedian(dxcx, histogram4, W_gf, index);
		}
		//thread����
		__syncthreads();

		//2�s�ڈȍ~�̏���
		for (int y = 1; y < height_; y++)
		{
			for (int k = 0, n = 0; k < numOfFrames; k++, n += 3)
			{
				//�q�X�g�O�����ɒǉ�
				f = tex2D<int>(texF[k], xPos, y + radius);
				g[0] = tex2D<int>(texG[n], xPos, y + radius);
				g[1] = tex2D<int>(texG[n + 1], xPos, y + radius);
				g[2] = tex2D<int>(texG[n + 2], xPos, y + radius);
				atomicAdd(&histogram4[f].x, 1);
				atomicAdd(&histogram4[f].y, g[0]);
				atomicAdd(&histogram4[f].z, g[1]);
				atomicAdd(&histogram4[f].w, g[2]);
				if (f <= index)
				{
					atomicAdd(&W_gf.x, 1);
					atomicAdd(&W_gf.y, g[0]);
					atomicAdd(&W_gf.z, g[1]);
					atomicAdd(&W_gf.w, g[2]);
				}
				//�q�X�g�O��������폜
				f = tex2D<int>(texF[k], xPos, y - radius - 1);
				g[0] = tex2D<int>(texG[n], xPos, y - radius - 1);
				g[1] = tex2D<int>(texG[n + 1], xPos, y - radius - 1);
				g[2] = tex2D<int>(texG[n + 2], xPos, y - radius - 1);
				atomicSub(&histogram4[f].x, 1);
				atomicSub(&histogram4[f].y, g[0]);
				atomicSub(&histogram4[f].z, g[1]);
				atomicSub(&histogram4[f].w, g[2]);
				if (f <= index)
				{
					atomicSub(&W_gf.x, 1);
					atomicSub(&W_gf.y, g[0]);
					atomicSub(&W_gf.z, g[1]);
					atomicSub(&W_gf.w, g[2]);
				}
			}

			//thread����
			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				float4 dxcx = *((float4*)((char*)CxDx + y * pitchF4) + x);
				//�����l�v�Z
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(dxcx, histogram4, W_gf, index);
			}
			//thread����
			__syncthreads();
		}
	}


	//I1G1
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, std::vector<int*>f, std::vector<int*> g, float2* dxcx)
	{
		int numOfFrames = f.size();

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
		TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		de_filter3D << <gridSizeX, blockSize, fRange * sizeof(int2), stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF.device, texG.device, numOfFrames, dxcx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
	}

	//I1G3
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, std::vector<int*>f, std::vector<DeviceArray<int>*> g, float4* dxcx)
	{
		int numOfFrames = f.size();

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
		TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		de_filter3D << <gridSizeX, blockSize, fRange * sizeof(int4), stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF.device, texG.device, numOfFrames, dxcx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
	}

	//I3G1
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, std::vector<DeviceArray<int>*>f, std::vector<int*> g, float2* dxcx)
	{
		int numOfFrames = f.size();

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);

		//f�ɂ��Ċe�`�����l����device memory��z��Ɋi�[
		std::vector<std::vector<int*>> fs(3, std::vector<int*>(numOfFrames));
		for (int i = 0; i < numOfFrames; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				fs[j][i] = f[i]->host[j];
			}
		}

		cudaStream_t streams[3];
		TextureArray<int>* texF[3];
		for (int i = 0; i < 3; i++)
		{
			texF[i] = new TextureArray<int>(fs[i], filterMode, sizeInfo);
			cudaStreamCreate(&streams[i]);
		}

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		for (int i = 0; i < 3; i++)
		{
			de_filter3D << <gridSizeX, blockSize, fRange * sizeof(int) * 2, streams[i] >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF[i]->device, texG.device, numOfFrames, dxcx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
		}
		cudaDeviceSynchronize();
		for (int i = 0; i < 3; i++)
		{
			delete texF[i];
		}
	}

	//I3G3
	void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, DeviceArray<int>* result_center, std::vector<DeviceArray<int>*>f, std::vector<DeviceArray<int>*> g, float4* dxcx)
	{
		int numOfFrames = f.size();

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);

		//f�ɂ��Ċe�`�����l����device memory��z��Ɋi�[
		std::vector<std::vector<int*>> fs(3, std::vector<int*>(numOfFrames));
		for (int i = 0; i < numOfFrames; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				fs[j][i] = f[i]->host[j];
			}
		}

		cudaStream_t streams[3];
		TextureArray<int>* texF[3];
		for (int i = 0; i < 3; i++)
		{
			texF[i] = new TextureArray<int>(fs[i], filterMode, sizeInfo);
			cudaStreamCreate(&streams[i]);
		}

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		for (int i = 0; i < 3; i++)
		{
			de_filter3D << <gridSizeX, blockSize, fRange * sizeof(int) * 4, streams[i] >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF[i]->device, texG.device, numOfFrames, dxcx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
		}
		cudaDeviceSynchronize();
		for (int i = 0; i < 3; i++)
		{
			delete texF[i];
		}
	}





	__global__ void
		de_pixel(int width_, int height_, int* dst, size_t pitchI1)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < 0 || x >= width_ || y < 0 || y >= height_)
			return;

		//	*((int*)((char*)dst + y * pitchI1) + x) = (y + x) * 256;
			//*((int*)((char*)dst + y * pitchI1) + x) = (threadIdx.x) * 256;
		printf("%d ", *((int*)((char*)dst + y * pitchI1) + x));
	}


	//block, grid test
	void cu_testBlockGrid()
	{

	}

#endif

}