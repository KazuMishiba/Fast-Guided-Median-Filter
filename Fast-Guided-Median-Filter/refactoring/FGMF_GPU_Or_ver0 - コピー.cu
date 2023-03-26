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
		de_calculateHcum(float* dxcx, int* W_fg_cum)
	{
		float h = 0.0f;
		for (int i = 0; i < channelNum_fg; i++)
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
		for (int i = 0; i < channelNum_fg; i++)
			(&W_fg_cum)[i] += W_FG[W_k * channelNum_fg + i] * sign;
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

	template<typename FG_TYPE>
	extern __shared__ FG_TYPE  sharedData[];

	template<typename FG_TYPE, typename DC_TYPE>
	__global__ void
		de_filter2dF1(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, DC_TYPE* dc, size_t pitchI1, size_t pitchF_channelNum_fg)
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

	template<typename FG_TYPE, typename DC_TYPE>
	__global__ void
		de_filter2dF3(int width, int height, int radius, int fRange, int** result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, DC_TYPE* dc, size_t pitchI1, size_t pitchF_channelNum_fg)
	{
		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		int s_handle = s_center + tid - radius;
		
		FG_TYPE* W_fg_cum0 = &sharedData<FG_TYPE>[0];
		FG_TYPE* W_FG0 = &sharedData<FG_TYPE>[channelNum_fg];
		FG_TYPE* W_fg_cum1 = &sharedData<FG_TYPE>[(fRange + 1)];
		FG_TYPE* W_FG1 = &sharedData<FG_TYPE>[(fRange + 1) + channelNum_fg];
		FG_TYPE* W_fg_cum2 = &sharedData<FG_TYPE>[(fRange + 1) * 2];
		FG_TYPE* W_FG2 = &sharedData<FG_TYPE>[(fRange + 1) * 2 + channelNum_fg];

		FG_TYPE* W_fg_cum[3] = { W_fg_cum0, W_fg_cum1,W_fg_cum2 };
		FG_TYPE* W_FG[3] = { W_FG0, W_FG1, W_FG2};
		__shared__ int W_k[3];

		__shared__ DC_TYPE dxcx;



		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			//�q�X�g�O����������, fgSumUpToIndex������
			memset(sharedData<FG_TYPE>, 0, sizeof(FG_TYPE) * (fRange + 1) * 3);
			W_k[0] = tex2D<int>(texF[0], s_center, 0);//current index
			W_k[1] = tex2D<int>(texF[1], s_center, 0);//current index
			W_k[2] = tex2D<int>(texF[2], s_center, 0);//current index
		}

		__syncthreads();
		//1�ڃq�X�g�O�����`��
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		for (int t = -radius; t <= radius; t++)
		{
			de_addPixelToWindow(&texF[0], texG, s_handle, t, W_FG[0], W_fg_cum[0], W_k[0]);
			de_addPixelToWindow(&texF[1], texG, s_handle, t, W_FG[1], W_fg_cum[1], W_k[1]);
			de_addPixelToWindow(&texF[2], texG, s_handle, t, W_FG[2], W_fg_cum[2], W_k[2]);
		}

		__syncthreads();
		//1�s�ڂ̒����l�v�Z
		if (tid == radius)
		{
			dxcx = *((DC_TYPE*)((char*)dc) + s_center);
		}
		__syncthreads();
		if (tid >= radius - 1 && tid <= radius + 1)
		{
			int id = tid - radius + 1;
			*((int*)((char*)result_center[id]) + s_center) = de_findMedian(dxcx, W_FG[id], W_fg_cum[id], W_k[id]);
		}

		//2�s�ڈȍ~�̏���
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			//�q�X�g�O�����ɒǉ�
			de_addPixelToWindow(&texF[0], texG, s_handle, t + radius, W_FG[0], W_fg_cum[0], W_k[0]);
			de_addPixelToWindow(&texF[1], texG, s_handle, t + radius, W_FG[1], W_fg_cum[1], W_k[1]);
			de_addPixelToWindow(&texF[2], texG, s_handle, t + radius, W_FG[2], W_fg_cum[2], W_k[2]);
			//�q�X�g�O��������폜
			de_removePixelFromWindow(&texF[0], texG, s_handle, t - radius - 1, W_FG[0], W_fg_cum[0], W_k[0]);
			de_removePixelFromWindow(&texF[1], texG, s_handle, t - radius - 1, W_FG[1], W_fg_cum[1], W_k[1]);
			de_removePixelFromWindow(&texF[2], texG, s_handle, t - radius - 1, W_FG[2], W_fg_cum[2], W_k[2]);

			__syncthreads();
			//���S�X���b�h�̂ݎ��s
			if (tid == radius)
			{
				dxcx = *((DC_TYPE*)((char*)dc + t * pitchF_channelNum_fg) + s_center);
			}
			__syncthreads();
			if (tid >= radius - 1 && tid <= radius + 1)
			{
				int id = tid - radius + 1;
				*((int*)((char*)result_center[id] + t * pitchI1) + s_center) = de_findMedian(dxcx, W_FG[id], W_fg_cum[id], W_k[id]);
			}
		}
	}

	


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
		int* histogramX = &buffers[channelNum_fg];
		//g1,g3�ł�fgXSumUpToIndex��f,g�̏��Ŏ����������Acxdx�Ə��Ԃ����킹�邽�߂ɁAg,...,g f�̏��ɂ���
		//histogram�����l
		//histogram��1�����ɕ���ł��āA�ebin�ɂ��āAg,...,g,f �̏��ɕ���ł���
		const int n = channelNum_fg - 1;

		__shared__ int index;
		int f;
		int* g = new int[n];

		float* cxdx;

		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			cxdx = new float[channelNum_fg];
			//�q�X�g�O����������
			for (int i = 0; i < fRange * channelNum_fg; i++)
			{
				histogramX[i] = 0;
			}
			for (int i = 0; i < channelNum_fg; i++)
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
					atomicAdd(&histogramX[f * channelNum_fg + i], g[i]);
				atomicAdd(&histogramX[f * channelNum_fg + n], 1);
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
			for (int i = 0; i < channelNum_fg; i++)
				cxdx[i] = *((float*)((char*)CxDx[i]) + x);
//			*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index);
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
				atomicAdd(&histogramX[f * channelNum_fg + i], g[i]);
			atomicAdd(&histogramX[f * channelNum_fg + n], 1);
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
				atomicSub(&histogramX[f * channelNum_fg + i], g[i]);
			atomicSub(&histogramX[f * channelNum_fg + n], 1);
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
				for (int i = 0; i < channelNum_fg; i++)
					cxdx[i] = *((float*)((char*)CxDx[i] + y * pitchF1) + x);
				//�����l�v�Z
//				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index);
			}
			//thread����
			__syncthreads();
		}


		delete g;
		if (tid == radius)
		{
			delete cxdx;
		}
		__syncthreads();
	}






	//�V����
	template<typename FG_TYPE, typename DC_TYPE>
	__global__ void
		de_filter2dF1new(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, DC_TYPE* dc, size_t pitchI1, size_t pitchF_channelNum_fg)
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















/*
	//I1G1 refactoring
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, int* g, Helper::DeviceArray<float>* cxdx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		cudaTextureObject_t texF, texG;
		Helper::UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h
		Helper::UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texG��g���o�C���h

	//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		de_filter2D << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* (1 + 1), stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF, texG, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());

		cudaDestroyTextureObject(texF);
		cudaDestroyTextureObject(texG);
	}
	*/

	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_FG>
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, DC_TYPE* dc)
	{
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int k = CHANNELNUM_FG;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

		de_filter2dF1<FG_TYPE, DC_TYPE> << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* k, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[0], texF.device, texG.device, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<DC_TYPE>());

	}
	template void cu_filter2DF1<int2, float2, 2>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, Helper::DeviceArray<int>*, Helper::DeviceArray<int>*, float2*);
	template void cu_filter2DF1<int4, float4, 4>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, Helper::DeviceArray<int>*, Helper::DeviceArray<int>*, float4*);


	template<typename FG_TYPE, typename DC_TYPE, int CHANNELNUM_FG>
	void cu_filter2DF3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, DC_TYPE* dc)
	{
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int k = CHANNELNUM_FG;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

		de_filter2dF3<FG_TYPE, DC_TYPE> << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* k * 3, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->device, texF.device, texG.device, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<DC_TYPE>());

	}
	template void cu_filter2DF3<int2, float2, 2>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, Helper::DeviceArray<int>*, Helper::DeviceArray<int>*, float2*);
	/*
	template void cu_filter2DF3<int4, float4, 4>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>*&, Helper::DeviceArray<int>*, Helper::DeviceArray<int>*, float4*);
	*/


	//template void cu_filter2DF1<int2, float2, 2>(Helper::SizeInfo&, cudaStream_t, int, int, Helper::DeviceArray<int>&, Helper::DeviceArray<int>, Helper::DeviceArray<int>, float2);
#if 0
	//I1G1
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, float2* dc)
	{
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int k = 2;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

		de_filter2dF1<int2, float2> << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* k, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[0], texF.device, texG.device, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	}

	//I1G3
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>*& result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, float4* dc)
	{
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		constexpr int k = 4;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));

		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);

		de_filter2dF1<int4, float4> << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* k, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[0], texF.device, texG.device, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
	}
	//I1GX
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, int* result_center, int* f, Helper::DeviceArray<int>* g, Helper::DeviceArray<float>* dc)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		cudaTextureObject_t texF;
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);
		Helper::UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		int n = g->arrayLength;
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));
		//shared memory�́A�q�X�g�O�����{uptoindex�K�v�ŁA�q�X�g�O������fRange * (n+1)�Auptoindex��n+1�K�v
		de_filter2D << <gridSizeX, blockSize, (fRange + 1) * sizeof(int)* (n + 1), stream >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center, texF, texG.device, dc->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());

		cudaDestroyTextureObject(texF);
	}
	//I3G1
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>* result_center, Helper::DeviceArray<int>* f, int* g, float2* dc)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		cudaTextureObject_t texG;
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);
		Helper::UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		int k = 2;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));
		//stream
		cudaStream_t streams[3];
		cudaStreamCreate(&streams[0]);
		cudaStreamCreate(&streams[1]);
		cudaStreamCreate(&streams[2]);

		for (int i = 0; i < 3; i++)
		{
			//��Ŏ���
//			de_filter2D<int2, float2> << <gridSizeX, blockSize, fRange * sizeof(int) * 2, streams[i] >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF.host[i], texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
		}
		//cudaDeviceSynchronize();
	}
	//I3G3
	void cu_filter2DF1(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int radius, int fRange, Helper::DeviceArray<int>* result_center, Helper::DeviceArray<int>* f, Helper::DeviceArray<int>* g, float4* dc)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		Helper::TextureArray<int> texG = Helper::TextureArray<int>(g, filterMode, sizeInfo);
		Helper::TextureArray<int> texF = Helper::TextureArray<int>(f, filterMode, sizeInfo);

		//�c������ shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width_;
		int k = 4;
		cudaMemcpyToSymbol(channelNum_fg, &k, sizeof(int));
		//stream
		cudaStream_t streams[3];
		cudaStreamCreate(&streams[0]);
		cudaStreamCreate(&streams[1]);
		cudaStreamCreate(&streams[2]);

		for (int i = 0; i < 3; i++)
		{
			de_filter2D << <gridSizeX, blockSize, fRange * sizeof(int) * 4, streams[i] >> > (sizeInfo.width_, sizeInfo.height_, radius, fRange, result_center->host[i], texF.host[i], texG.device, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
		}
		//cudaDeviceSynchronize();
	}

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
		de_filter3D(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float2* CxDx, size_t pitchI1, size_t pitchF2)
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
		int xPos = s + tid - radius;

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
			for (int yy = -radius; yy <= radius; yy++)
			{
				//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
				{
					f = tex2D<int>(texF[k], xPos, yy);
					g = tex2D<int>(texG[k], xPos, yy);
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
			float2 dc = *((float2*)((char*)CxDx) + s);
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
				f = tex2D<int>(texF[k], xPos, t + radius);
				g = tex2D<int>(texG[k], xPos, t + radius);
				atomicAdd(&histogram2[f].s, 1);
				atomicAdd(&histogram2[f].t, g);
				if (f <= W_k)
				{
					atomicAdd(&W_fg_cum.s, 1);
					atomicAdd(&W_fg_cum.t, g);
				}
				//�q�X�g�O��������폜
				f = tex2D<int>(texF[k], xPos, t - radius - 1);
				g = tex2D<int>(texG[k], xPos, t - radius - 1);
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
				float2 dc = *((float2*)((char*)CxDx + t * pitchF2) + s);
				//�����l�v�Z
				*((int*)((char*)result_center + t * pitchI1) + s) = de_findMedian(dc, histogram2, W_fg_cum, W_k);
			}
			//thread����
			__syncthreads();
		}
	}

	//g3
	__global__ void
		de_filter3D(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float4* CxDx, size_t pitchI1, size_t pitchF4)
	{
		int s = blockIdx.s;
		if (s >= width)
			return;
		int tid = threadIdx.s;
		if (tid >= radius * 2 + 1)
			return;
		int xPos = s + tid - radius;

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
			for (int yy = -radius; yy <= radius; yy++)
			{
				//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
				{
					f = tex2D<int>(texF[k], xPos, yy);
					g[0] = tex2D<int>(texG[n], xPos, yy);
					g[1] = tex2D<int>(texG[n + 1], xPos, yy);
					g[2] = tex2D<int>(texG[n + 2], xPos, yy);
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
			float4 dc = *((float4*)((char*)CxDx) + s);
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
				f = tex2D<int>(texF[k], xPos, t + radius);
				g[0] = tex2D<int>(texG[n], xPos, t + radius);
				g[1] = tex2D<int>(texG[n + 1], xPos, t + radius);
				g[2] = tex2D<int>(texG[n + 2], xPos, t + radius);
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
				f = tex2D<int>(texF[k], xPos, t - radius - 1);
				g[0] = tex2D<int>(texG[n], xPos, t - radius - 1);
				g[1] = tex2D<int>(texG[n + 1], xPos, t - radius - 1);
				g[2] = tex2D<int>(texG[n + 2], xPos, t - radius - 1);
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
				float4 dc = *((float4*)((char*)CxDx + t * pitchF4) + s);
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