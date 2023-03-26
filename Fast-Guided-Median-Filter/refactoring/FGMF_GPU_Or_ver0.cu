#include "FGMF_GPU_Or_ver0.cuh"
#include<opencv2/opencv.hpp>
#include <vector>

namespace FGMF_GPU_Or_ver0
{
	__constant__ int channelNum_fg;
	//���s������save3��ver0���畜�A


	/*
	* c,d��Helper::DeviceArray���g���ƂP�`�����l���̏ꍇ��15%���x���Ȃ�B�K�C�h�R�`�����l�����Ƃ�茰���ɒx���Ȃ邾�낤�B
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
	*/
	/*
	* ����atomic���Z�q�̌v�Z���x�������ł���A�R�`�����l�������ɏ��������Ƃ���ő��x�̌���͌����Ȃ������B
	* �����Ȃ�ƁAatomic���Z�q�����S�ɔr�����������̕��������\��������B
	* ���肤��P�ڂ̕��@�́A�P�X���b�h�P��ŏ������邱�Ƃ����A�����shared memory�������Ɍ͊�����͂��Ȃ̂Ō����ɂ͓���B
	* ���̕��@�Ƃ��čl�����̂́A�q�X�g�O�����r�����̃X���b�h���P�u���b�N�ŋN�����A�e�X���b�h���P��E�B���h�E���̃f�[�^��ǂݍ��݁A�����̒S������r���̒l��������q�X�g�O�����ɒǉ�����A�Ƃ������@���Ƃ�B
	* ���̂Ƃ��A�E�B���h�E���̃f�[�^���Ƃ肠�����e�X���b�h������ɓǂނ��Ƃɂ��邪�A������e�X���b�h�ɕ���ɍs�킹��shared memory�ɓ������@�����肤��B�����Ɏ��Ԃ�������̂ŁA����͌�񂵂ɂ���B
	*/

	/*
	template<typename CD_TYPE, typename FG_TYPE>
	__device__ inline int
		de_calculateHcum(const CD_TYPE& cxdx, FG_TYPE& fgSumUpToIndex, int& index)
	{
		return cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.x;
	}
	template<typename CD_TYPE, typename FG_TYPE>
	__device__ inline int
		de_calculateHcum(const CD_TYPE& cxdx, const FG_TYPE* histogram, FG_TYPE& fgSumUpToIndex, int& index)
	{
		return cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.x;
	}
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
		de_calculateHcum(float& dxcx, int& W_fg_cum)
	{
		float* pdxcx = &dxcx;
		int* pW_fg_cum = &W_fg_cum;
		float h = 0.0f;
		for (int i = 0; i < channelNum_fg; i++)
			h += pdxcx[i] * pW_fg_cum[i];
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
		for (int i = 0; i < channelNum_fg; i++)
			(&W_fg_cum)[i] += W_FG[W_k * channelNum_fg + i] * sign;
		return de_calculateHcum(dxcx, W_fg_cum);
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
		//int* g = new int[channelNum_fg - 1];
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
		//int* g = new int[channelNum_fg - 1];
		f = tex2D<int>(*texF, s, t);
		atomicAdd(&W_FG[f * channelNum_fg], 1);

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







	template<typename FG_TYPE>
	extern __shared__ FG_TYPE  sharedData[];

	template<typename FG_TYPE, typename DC_TYPE>
	__global__ void
		de_filter2d(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, DC_TYPE* dc, size_t pitchI1, size_t pitchF_channelNum_fg)
	{
		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		int s_handle = s_center + tid - radius;

		FG_TYPE* W_fg_cum = &sharedData<FG_TYPE>[0];
		FG_TYPE* W_FG = &sharedData<FG_TYPE>[channelNum_fg];
		__shared__ int W_k;


		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			//�q�X�g�O����������, fgSumUpToIndex������
			memset(sharedData<FG_TYPE>, 0, sizeof(FG_TYPE) * (fRange + 1));
			W_k = tex2D<int>(*texF, s_center, 0);//current index
		}

		__syncthreads();
		//1�ڃq�X�g�O�����`��
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		for (int t = -radius; t <= radius; t++)
			de_addPixelToWindow(texF, texG, s_handle, t, W_FG, W_fg_cum, W_k);

		__syncthreads();
		//1�s�ڂ̒����l�v�Z
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			DC_TYPE dxcx = *((DC_TYPE*)((char*)dc) + s_center);
			*((int*)((char*)result_center) + s_center) = de_findMedian(dxcx, W_FG, W_fg_cum, W_k);
		}

		//2�s�ڈȍ~�̏���
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			//�q�X�g�O�����ɒǉ�
			de_addPixelToWindow(texF, texG, s_handle, t + radius, W_FG, W_fg_cum, W_k);
			//�q�X�g�O��������폜
			de_removePixelFromWindow(texF, texG, s_handle, t - radius - 1, W_FG, W_fg_cum, W_k);

			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				DC_TYPE dxcx = *((DC_TYPE*)((char*)dc + t * pitchF_channelNum_fg) + s_center);
				//�����l�v�Z
				*((int*)((char*)result_center + t * pitchI1) + s_center) = de_findMedian(dxcx, W_FG, W_fg_cum, W_k);
			}
		}
	}


	


	//gX
	__global__ void
		de_filter2D_multichannel(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, float** dc, size_t pitchI1, size_t pitchF1)
	{
		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		int s_handle = s_center + tid - radius;

		int* W_fg_cum = &sharedData<int>[0];
		int* W_FG = &sharedData<int>[channelNum_fg];
		__shared__ int W_k;

		int a = channelNum_fg;
		float* dxcx;

		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			dxcx = new float[channelNum_fg];
			//�q�X�g�O����������, fgSumUpToIndex������
			memset(sharedData<int>, 0, sizeof(int) * channelNum_fg * (fRange + 1) * channelNum_fg);
			W_k = tex2D<int>(*texF, s_center, 0);//current index
		}

		__syncthreads();
		//1�ڃq�X�g�O�����`��
		for (int t = -radius; t <= radius; t++)
			de_addPixelToWindow(texF, texG, s_handle, t, W_FG, W_fg_cum, W_k);

		__syncthreads();
		//1�s�ڂ̒����l�v�Z
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			for (int i = 0; i < channelNum_fg; i++)
				dxcx[i] = *((float*)((char*)dc[i]) + s_center);
			*((int*)((char*)result_center) + s_center) = de_findMedian(*dxcx, W_FG, W_fg_cum, W_k);
		}
		//&float int, int, int
		__syncthreads();

		//2�s�ڈȍ~�̏���
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			//�q�X�g�O�����ɒǉ�
			de_addPixelToWindow(texF, texG, s_handle, t + radius, W_FG, W_fg_cum, W_k);
			//�q�X�g�O��������폜
			de_removePixelFromWindow(texF, texG, s_handle, t - radius - 1, W_FG, W_fg_cum, W_k);

			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				for (int i = 0; i < channelNum_fg; i++)
					dxcx[i] = *((float*)((char*)dc[i] + t * pitchF1) + s_center);
				//�����l�v�Z
				*((int*)((char*)result_center + t * pitchI1) + s_center) = de_findMedian(*dxcx, W_FG, W_fg_cum, W_k);
			}
		}

		if (tid == radius)
		{
			delete dxcx;
		}
	}








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
			//de_filter2D_multichannel << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* k, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[c], &(texF.device[c]), texG.device, dc->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
		}
	}


#if 0

	//I1GX
	void cu_filter2DMultiChannel(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* dc)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);
		cudaTextureObject_t texF;
		Helper::UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		int n = g->arrayLength;
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));
		de_filter2D << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* (n + 1), NULL >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF, texG.device, dc->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
		cudaDestroyTextureObject(texF);
		//cudaDeviceSynchronize();
	}
	//IXGY
	void cu_filter2DMultiChannel(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>* result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* dc)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		int n = g->arrayLength;
		int m = f->arrayLength;
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));
		for (int i = 0; i < m; i++)
		{
			de_filter2D << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* (n + 1), NULL >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF.host[i], texG.device, dc->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
		}
		//cudaDeviceSynchronize();
	}

#endif



#if 0
	//////////////////////////////////////////////////
	//3D
	
	//G1
	__device__ inline int
		de_findMedian(const float2& dc, const int2* W_FG, int2& W_fg_cum, int& W_k)
	{
		float h = dc.s * W_fg_cum.t + dc.t * W_fg_cum.s;
		const int flagA = h < 0.5f;
		const int flag2 = flagA - 1;
		const int sign = flagA * 2 - 1;
		while (true)
		{
			W_k += flagA;
			//if(histogram[index].x)
			{
				W_fg_cum.s += W_FG[W_k].s * sign;
				W_fg_cum.t += W_FG[W_k].t * sign;
				h = dc.s * W_fg_cum.t + dc.t * W_fg_cum.s;
				if ((h >= 0.5f) == flagA)
				{
					int result_center = W_k;
					W_k += flag2;
					return result_center;
				}
			}
			W_k += flag2;
		}
	}
	__device__ inline int
		de_findMedian(const float4& dc, const int4* W_FG, int4& W_fg_cum, int& W_k)
	{
		float h = dc.s * W_fg_cum.t + dc.t * W_fg_cum.z + dc.z * W_fg_cum.w + dc.w * W_fg_cum.s;
		const int flagA = h < 0.5f;
		const int flag2 = flagA - 1;
		const int sign = flagA * 2 - 1;
		while (true)
		{
			W_k += flagA;
			//if(histogram[index].x)
			{
				W_fg_cum.s += W_FG[W_k].s * sign;
				W_fg_cum.t += W_FG[W_k].t * sign;
				W_fg_cum.z += W_FG[W_k].z * sign;
				W_fg_cum.w += W_FG[W_k].w * sign;
				h = dc.s * W_fg_cum.t + dc.t * W_fg_cum.z + dc.z * W_fg_cum.w + dc.w * W_fg_cum.s;
				if ((h >= 0.5f) == flagA)
				{
					int result_center = W_k;
					W_k += flag2;
					return result_center;
				}
			}
			W_k += flag2;
		}
	}




	//����͔C�ӂ̎����ł��g�������i�|�C���^���K�؂Ȃ�j
	//g1
	__global__ void
		de_filter3D(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float2* dc, size_t pitchI1, size_t pitchF2)
	{
		//�����Ώۂ̒��S���W�� blockIdx.x �ɂȂ�B
		/*
		* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
		* ���Sthread��idx��radius
		*/

		int s = blockIdx.s;
		if (s >= width)
			return;
		int tid = threadIdx.s;
		if (tid >= radius * 2 + 1)
			return;
		int s_handle = s + tid - radius;

		extern __shared__ int2 histogram2[];

		__shared__ int W_k;
		__shared__ int2 W_fg_cum;//index�ȉ��a

		int f, g;

		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			//�q�X�g�O����������
			for (int i = 0; i < fRange; i++)
			{
				histogram2[i] = make_int2(0, 0);
			}
			W_fg_cum = make_int2(0, 0);
			W_k = tex2D<int>(texF[numOfFrames / 2], s, 0);//current index �{���͏������S�t���[��index���w�肵����(���ꂾ�ƒ[�̂Ƃ��ɁA�������S�ł͂Ȃ��Ȃ�)
		}
		//thread����
		__syncthreads();

		//1�ڃq�X�g�O�����`��
		for (int k = 0; k < numOfFrames; k++)
		{
			for (int t = -radius; t <= radius; t++)
			{
				//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
				{
					f = tex2D<int>(texF[k], s_handle, t);
					g = tex2D<int>(texG[k], s_handle, t);
					atomicAdd(&histogram2[f].s, 1);
					atomicAdd(&histogram2[f].t, g);
					if (f <= W_k)
					{
						atomicAdd(&W_fg_cum.s, 1);
						atomicAdd(&W_fg_cum.t, g);
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
			float2 dc = *((float2*)((char*)dc) + s);
			*((int*)((char*)result_center) + s) = de_findMedian(dc, histogram2, W_fg_cum, W_k);
		}
		//thread����
		__syncthreads();

		//2�s�ڈȍ~�̏���
		for (int t = 1; t < height; t++)
		{
			for (int k = 0; k < numOfFrames; k++)
			{
				//�q�X�g�O�����ɒǉ�
				f = tex2D<int>(texF[k], s_handle, t + radius);
				g = tex2D<int>(texG[k], s_handle, t + radius);
				atomicAdd(&histogram2[f].s, 1);
				atomicAdd(&histogram2[f].t, g);
				if (f <= W_k)
				{
					atomicAdd(&W_fg_cum.s, 1);
					atomicAdd(&W_fg_cum.t, g);
				}
				//�q�X�g�O��������폜
				f = tex2D<int>(texF[k], s_handle, t - radius - 1);
				g = tex2D<int>(texG[k], s_handle, t - radius - 1);
				atomicSub(&histogram2[f].s, 1);
				atomicSub(&histogram2[f].t, g);
				if (f <= W_k)
				{
					atomicSub(&W_fg_cum.s, 1);
					atomicSub(&W_fg_cum.t, g);
				}
			}

			//thread����
			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				float2 dc = *((float2*)((char*)dc + t * pitchF2) + s);
				//�����l�v�Z
				*((int*)((char*)result_center + t * pitchI1) + s) = de_findMedian(dc, histogram2, W_fg_cum, W_k);
			}
			//thread����
			__syncthreads();
		}
	}

	//g3
	__global__ void
		de_filter3D(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float4* dc, size_t pitchI1, size_t pitchF4)
	{
		int s = blockIdx.s;
		if (s >= width)
			return;
		int tid = threadIdx.s;
		if (tid >= radius * 2 + 1)
			return;
		int s_handle = s + tid - radius;

		extern __shared__ int4 histogram4[];
		__shared__ int W_k;
		__shared__ int4 W_fg_cum;//index�ȉ��a

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
			W_fg_cum = make_int4(0, 0, 0, 0);
			W_k = tex2D<int>(texF[numOfFrames / 2], s, 0);//current index �{���͏������S�t���[��index���w�肵����(���ꂾ�ƒ[�̂Ƃ��ɁA�������S�ł͂Ȃ��Ȃ�)
		}
		//thread����
		__syncthreads();

		//1�ڃq�X�g�O�����`��
		for (int k = 0, n = 0; k < numOfFrames; k++, n += 3)
		{
			for (int t = -radius; t <= radius; t++)
			{
				//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
				{
					f = tex2D<int>(texF[k], s_handle, t);
					g[0] = tex2D<int>(texG[n], s_handle, t);
					g[1] = tex2D<int>(texG[n + 1], s_handle, t);
					g[2] = tex2D<int>(texG[n + 2], s_handle, t);
					atomicAdd(&histogram4[f].s, 1);
					atomicAdd(&histogram4[f].t, g[0]);
					atomicAdd(&histogram4[f].z, g[1]);
					atomicAdd(&histogram4[f].w, g[2]);
					if (f <= W_k)
					{
						atomicAdd(&W_fg_cum.s, 1);
						atomicAdd(&W_fg_cum.t, g[0]);
						atomicAdd(&W_fg_cum.z, g[1]);
						atomicAdd(&W_fg_cum.w, g[2]);
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
			float4 dc = *((float4*)((char*)dc) + s);
			*((int*)((char*)result_center) + s) = de_findMedian(dc, histogram4, W_fg_cum, W_k);
		}
		//thread����
		__syncthreads();

		//2�s�ڈȍ~�̏���
		for (int t = 1; t < height; t++)
		{
			for (int k = 0, n = 0; k < numOfFrames; k++, n += 3)
			{
				//�q�X�g�O�����ɒǉ�
				f = tex2D<int>(texF[k], s_handle, t + radius);
				g[0] = tex2D<int>(texG[n], s_handle, t + radius);
				g[1] = tex2D<int>(texG[n + 1], s_handle, t + radius);
				g[2] = tex2D<int>(texG[n + 2], s_handle, t + radius);
				atomicAdd(&histogram4[f].s, 1);
				atomicAdd(&histogram4[f].t, g[0]);
				atomicAdd(&histogram4[f].z, g[1]);
				atomicAdd(&histogram4[f].w, g[2]);
				if (f <= W_k)
				{
					atomicAdd(&W_fg_cum.s, 1);
					atomicAdd(&W_fg_cum.t, g[0]);
					atomicAdd(&W_fg_cum.z, g[1]);
					atomicAdd(&W_fg_cum.w, g[2]);
				}
				//�q�X�g�O��������폜
				f = tex2D<int>(texF[k], s_handle, t - radius - 1);
				g[0] = tex2D<int>(texG[n], s_handle, t - radius - 1);
				g[1] = tex2D<int>(texG[n + 1], s_handle, t - radius - 1);
				g[2] = tex2D<int>(texG[n + 2], s_handle, t - radius - 1);
				atomicSub(&histogram4[f].s, 1);
				atomicSub(&histogram4[f].t, g[0]);
				atomicSub(&histogram4[f].z, g[1]);
				atomicSub(&histogram4[f].w, g[2]);
				if (f <= W_k)
				{
					atomicSub(&W_fg_cum.s, 1);
					atomicSub(&W_fg_cum.t, g[0]);
					atomicSub(&W_fg_cum.z, g[1]);
					atomicSub(&W_fg_cum.w, g[2]);
				}
			}

			//thread����
			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				float4 dc = *((float4*)((char*)dc + t * pitchF4) + s);
				//�����l�v�Z
				*((int*)((char*)result_center + t * pitchI1) + s) = de_findMedian(dc, histogram4, W_fg_cum, W_k);
			}
			//thread����
			__syncthreads();
		}
	}


	//I1G1
	void cu_filter3D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, std::vector<int*>f, std::vector<int*> g, float2* dc)
	{
		int numOfFrames = f.size();

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		de_filter3D << <gridSizeX, blockSize, fRange * sizeof(int2), stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF.device, texG.device, numOfFrames, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
	}

	//I1G3
	void cu_filter3D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, std::vector<int*>f, std::vector<Helper::DeviceArray<int>*> g, float4* dc)
	{
		int numOfFrames = f.size();

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		de_filter3D << <gridSizeX, blockSize, fRange * sizeof(int4), stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF.device, texG.device, numOfFrames, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
	}

	//I3G1
	void cu_filter3D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>* result_center, std::vector<Helper::DeviceArray<int>*>f, std::vector<int*> g, float2* dc)
	{
		int numOfFrames = f.size();

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

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
		Helper::TextureArray<int>* texF[3];
		for (int i = 0; i < 3; i++)
		{
			texF[i] = new Helper::TextureArray<int>(fs[i], filterMode, sizeInfo);
			cudaStreamCreate(&streams[i]);
		}

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		for (int i = 0; i < 3; i++)
		{
			de_filter3D << <gridSizeX, blockSize, fRange * sizeof(int) * 2, streams[i] >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF[i]->device, texG.device, numOfFrames, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
		}
		cudaDeviceSynchronize();
		for (int i = 0; i < 3; i++)
		{
			delete texF[i];
		}
	}

	//I3G3
	void cu_filter3D(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>* result_center, std::vector<Helper::DeviceArray<int>*>f, std::vector<Helper::DeviceArray<int>*> g, float4* dc)
	{
		int numOfFrames = f.size();

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

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
		Helper::TextureArray<int>* texF[3];
		for (int i = 0; i < 3; i++)
		{
			texF[i] = new Helper::TextureArray<int>(fs[i], filterMode, sizeInfo);
			cudaStreamCreate(&streams[i]);
		}

		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		for (int i = 0; i < 3; i++)
		{
			de_filter3D << <gridSizeX, blockSize, fRange * sizeof(int) * 4, streams[i] >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF[i]->device, texG.device, numOfFrames, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
		}
		cudaDeviceSynchronize();
		for (int i = 0; i < 3; i++)
		{
			delete texF[i];
		}
	}





	__global__ void
		de_pixel(int width, int height, int* dst, size_t pitchI1)
	{
		int s = blockIdx.s * blockDim.s + threadIdx.s;
		int t = blockIdx.t * blockDim.t + threadIdx.t;
		if (s < 0 || s >= width || t < 0 || t >= height)
			return;

		//	*((int*)((char*)dst + y * pitchI1) + x) = (y + x) * 256;
			//*((int*)((char*)dst + y * pitchI1) + x) = (threadIdx.x) * 256;
		printf("%d ", *((int*)((char*)dst + t * pitchI1) + s));
	}


	//block, grid test
	void cu_testBlockGrid()
	{

	}
#endif
}