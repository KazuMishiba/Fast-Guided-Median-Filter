#include "FGMF_type3.cuh"
#include<opencv2/opencv.hpp>
#include <vector>

extern __shared__ int4 histogram4[];
extern __shared__ int2 histogram2[];

#define TX 790
#define TY 142

//G1
__device__ inline int
de_findMedian(const float2& cxdx, const int2* histogram, int2& fgSumUpToIndex, int& index)
{
	float h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.x;
	const int flagA = h < 0.5f;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	while (true)
	{
		index += flagA;
		//if(histogram[index].x)
		{
			fgSumUpToIndex.x += histogram[index].x * sign;
			fgSumUpToIndex.y += histogram[index].y * sign;
			h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.x;
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

//G3
__device__ inline int
de_findMedian(const float4& cxdx, const int4* histogram, int4& fgSumUpToIndex, int& index)
{
	float h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.z  + cxdx.z * fgSumUpToIndex.w + cxdx.w * fgSumUpToIndex.x;
	const int flagA = h < 0.5f;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	while (true)
	{
		index += flagA;
		//if(histogram[index].x)
		{
			fgSumUpToIndex.x += histogram[index].x * sign;
			fgSumUpToIndex.y += histogram[index].y * sign;
			fgSumUpToIndex.z += histogram[index].z * sign;
			fgSumUpToIndex.w += histogram[index].w * sign;
			h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.z + cxdx.z * fgSumUpToIndex.w + cxdx.w * fgSumUpToIndex.x;
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

//GX n��G�̃`�����l����
__device__ inline int
de_findMedian( float*& cxdx,  int* histogram, int *& fgSumUpToIndex, int& index, int n)
{
	int saveIndex = index;



	float h = 0.0f;
	for (int i = 0; i <= n; i++) {
		h += cxdx[i] * fgSumUpToIndex[i];
	}
	const int flagA = h < 0.5f;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	const int k = n + 1;

	
	while (true)
	{
		index += flagA;
		//if(histogram[index].x)//�����̓R�����g�O���Ă���������
		{
			for (int i = 0; i <= n; i++)
				fgSumUpToIndex[i] += histogram[index * k + i] * sign;
			h = 0.0f;
			for (int i = 0; i <= n; i++)
				h += cxdx[i] * fgSumUpToIndex[i];
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





//�c�����X�L������ shared�������g�p��
/*
�P�u���b�N������P�񏈗�����B
�P�u���b�N�ɑ΂��āA�E�B���h�E���a���̃X���b�h���N������B
�q�X�g�O������shared�������Ɏ����A�u���b�N���ŋ��L����B
�e�X���b�h�̖����́A�e�N�X�`������������Ή�����ʒu��f,g���T���v�����O���āA�A�g�~�b�N���Z�Ńq�X�g�O�����ɒǉ��A�폜���邱�Ƃł���B
�u���b�N�����C���̃X���b�h�������A���̃X���b�h�͎������g���܂ފe�X���b�h�̃q�X�g�O�����\�z��҂��Ē����l���v�Z����
*/
__global__ void
de_filter2D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//�����Ώۂ̒��S���W�� blockIdx.x �ɂȂ�B
	/*
	* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
	* ���Sthread��idx��radius
	*/

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + tid - radius;

	__shared__ int index;
	__shared__ int2 fgSumUpToIndex;//index�ȉ��a

	int f, g;

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram2[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
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
			g = tex2D<int>(texG, xPos, yy);
			atomicAdd(&histogram2[f].x, 1);
			atomicAdd(&histogram2[f].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g);
			}
		}
	}
	//thread����
	__syncthreads();
	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		float2 cxdx = *((float2*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram2, fgSumUpToIndex, index);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		//�q�X�g�O�����ɒǉ�
		f = tex2D<int>(texF, xPos, y + radius);
		g = tex2D<int>(texG, xPos, y + radius);
		atomicAdd(&histogram2[f].x, 1);
		atomicAdd(&histogram2[f].y, g);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, g);
		}
		//�q�X�g�O��������폜
		f = tex2D<int>(texF, xPos, y - radius - 1);
		g = tex2D<int>(texG, xPos, y - radius - 1);
		atomicSub(&histogram2[f].x, 1);
		atomicSub(&histogram2[f].y, g);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, g);
		}

		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			float2 cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//�����l�v�Z
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram2, fgSumUpToIndex, index);
		}
		//thread����
		__syncthreads();
	}
}




//g3
__global__ void
de_filter2D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t* texG3, float4* CxDx, size_t pitchI1, size_t pitchF4)
{
	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + tid - radius;

	
	__shared__ int index;
	__shared__ int4 fgSumUpToIndex;//index�ȉ��a
	int f;
	int g[3];

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram4[i] = make_int4(0, 0, 0, 0);
		}
		fgSumUpToIndex = make_int4(0, 0, 0, 0);
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
			g[0] = tex2D<int>(texG3[0], xPos, yy);
			g[1] = tex2D<int>(texG3[1], xPos, yy);
			g[2] = tex2D<int>(texG3[2], xPos, yy);
			atomicAdd(&histogram4[f].x, 1);
			atomicAdd(&histogram4[f].y, g[0]);
			atomicAdd(&histogram4[f].z, g[1]);
			atomicAdd(&histogram4[f].w, g[2]);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g[0]);
				atomicAdd(&fgSumUpToIndex.z, g[1]);
				atomicAdd(&fgSumUpToIndex.w, g[2]);
			}
			
		}
	}
	//thread����
	__syncthreads();
	
	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		float4 cxdx = *((float4*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		//�q�X�g�O�����ɒǉ�
		f = tex2D<int>(texF, xPos, y + radius);
		g[0] = tex2D<int>(texG3[0], xPos, y + radius);
		g[1] = tex2D<int>(texG3[1], xPos, y + radius);
		g[2] = tex2D<int>(texG3[2], xPos, y + radius);
		atomicAdd(&histogram4[f].x, 1);
		atomicAdd(&histogram4[f].y, g[0]);
		atomicAdd(&histogram4[f].z, g[1]);
		atomicAdd(&histogram4[f].w, g[2]);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, g[0]);
			atomicAdd(&fgSumUpToIndex.z, g[1]);
			atomicAdd(&fgSumUpToIndex.w, g[2]);
		}
		//�q�X�g�O��������폜
		f = tex2D<int>(texF, xPos, y - radius - 1);
		g[0] = tex2D<int>(texG3[0], xPos, y - radius - 1);
		g[1] = tex2D<int>(texG3[1], xPos, y - radius - 1);
		g[2] = tex2D<int>(texG3[2], xPos, y - radius - 1);
		atomicSub(&histogram4[f].x, 1);
		atomicSub(&histogram4[f].y, g[0]);
		atomicSub(&histogram4[f].z, g[1]);
		atomicSub(&histogram4[f].w, g[2]);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, g[0]);
			atomicSub(&fgSumUpToIndex.z, g[1]);
			atomicSub(&fgSumUpToIndex.w, g[2]);
		}

		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			float4 cxdx = *((float4*)((char*)CxDx + y * pitchF4) + x);
			//�����l�v�Z
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
		}
		//thread����
		__syncthreads();
	}
	
}


//gX
__global__ void
de_filter2D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t* texGX, float** CxDx, size_t pitchI1, size_t pitchF1, int n)
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
	int *fgXSumUpToIndex = &buffers[0];
	int *histogramX = &buffers[n+1];
	//g1,g3�ł�fgXSumUpToIndex��f,g�̏��Ŏ����������Acxdx�Ə��Ԃ����킹�邽�߂ɁAg,...,g f�̏��ɂ���
	//histogram�����l
	//histogram��1�����ɕ���ł��āA�ebin�ɂ��āAg,...,g,f �̏��ɕ���ł���

	__shared__ int index;
	int f;
	int *g = new int[n];

	const int k = n + 1;
	float *cxdx;

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		cxdx = new float[k];
		//�q�X�g�O����������
		for (int i = 0; i < Imax * k; i++)
		{
			histogramX[i] = 0;
		}
		for (int i = 0; i < k; i++)
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
				atomicAdd(&histogramX[f*k+i], g[i]);
			atomicAdd(&histogramX[f*k + n], 1);
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
		for (int i = 0; i < k; i++)
			cxdx[i] = *((float*)((char*)CxDx[i]) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);		
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
			atomicAdd(&histogramX[f*k + i], g[i]);
		atomicAdd(&histogramX[f*k + n], 1);
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
			atomicSub(&histogramX[f*k + i], g[i]);
		atomicSub(&histogramX[f*k + n], 1);
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
			for (int i = 0; i < k; i++)
				cxdx[i] = *((float*)((char*)CxDx[i] + y * pitchF1) + x);
			//�����l�v�Z
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);
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


//I1G1
void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texF, texG;
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h
	UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texG��g���o�C���h

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter2D << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}
//I1G3
void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, DeviceArray<int>* g, float4* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texF;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter2D << <gridSizeX, blockSize, Imax * sizeof(int) * 4, stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG.device, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());

	cudaDestroyTextureObject(texF);
}
//I1GX
void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, DeviceArray<int>* g, DeviceArray<float>* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texF;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	int n = g->arrayLength;
	//shared memory�́A�q�X�g�O�����{uptoindex�K�v�ŁA�q�X�g�O������Imax * (n+1)�Auptoindex��n+1�K�v
	de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int) * (n+1), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG.device, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>(), n);

	cudaDestroyTextureObject(texF);
}
//I3G1
void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, int* g, float2* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texG;
	TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);
	UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	//stream
	cudaStream_t streams[3];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);
	cudaStreamCreate(&streams[2]);

	for (int i = 0; i < 3; i++)
	{
		de_filter2D << <gridSizeX, blockSize, Imax * sizeof(int) * 2, streams[i] >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF.host[i], texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
	}
	//cudaDeviceSynchronize();
}
//I3G3
void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, float4* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	//stream
	cudaStream_t streams[3];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);
	cudaStreamCreate(&streams[2]);

	for (int i = 0; i < 3; i++)
	{
		de_filter2D << <gridSizeX, blockSize, Imax * sizeof(int) * 4, streams[i] >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF.host[i], texG.device, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
	}
	//cudaDeviceSynchronize();
}

//I1GX
void cu_filter2DMultiChannel(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, DeviceArray<int>* g, DeviceArray<float>* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	cudaTextureObject_t texF;
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	int n = g->arrayLength;
	de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int) * (n + 1), NULL >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG.device, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>(), n);
	cudaDestroyTextureObject(texF);
	//cudaDeviceSynchronize();
}
//IXGY
void cu_filter2DMultiChannel(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, DeviceArray<float>* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	int n = g->arrayLength;
	int m = f->arrayLength;
	for (int i = 0; i < m; i++)
	{
		de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int) * (n + 1), NULL >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF.host[i], texG.device, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>(), n);
	}
	//cudaDeviceSynchronize();
}





//////////////////////////////////////////////////
//1�s1�X���b�h�����p�@I1G1�̂�
/*
//�L���b�V���g�p
void cu_filter2D_Cache(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texF, texG;
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h
	UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texG��g���o�C���h

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter2D_Cache << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}

//shared memory�g�p
void cu_filter2D_Shared(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texF, texG;
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h
	UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texG��g���o�C���h

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter2D << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}
*/


//////////////////////////////////////////////////
//3D

//����͔C�ӂ̎����ł��g�������i�|�C���^���K�؂Ȃ�j
//g1
__global__ void
de_filter3D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//�����Ώۂ̒��S���W�� blockIdx.x �ɂȂ�B
	/*
	* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
	* ���Sthread��idx��radius
	*/

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + tid - radius;

	__shared__ int index;
	__shared__ int2 fgSumUpToIndex;//index�ȉ��a

	int f, g;

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram2[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
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
					atomicAdd(&fgSumUpToIndex.x, 1);
					atomicAdd(&fgSumUpToIndex.y, g);
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
		float2 cxdx = *((float2*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram2, fgSumUpToIndex, index);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
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
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g);
			}
			//�q�X�g�O��������폜
			f = tex2D<int>(texF[k], xPos, y - radius - 1);
			g = tex2D<int>(texG[k], xPos, y - radius - 1);
			atomicSub(&histogram2[f].x, 1);
			atomicSub(&histogram2[f].y, g);
			if (f <= index)
			{
				atomicSub(&fgSumUpToIndex.x, 1);
				atomicSub(&fgSumUpToIndex.y, g);
			}
		}

		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			float2 cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//�����l�v�Z
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram2, fgSumUpToIndex, index);
		}
		//thread����
		__syncthreads();
	}
}

//g3
__global__ void
de_filter3D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float4* CxDx, size_t pitchI1, size_t pitchF4)
{
	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + tid - radius;

	__shared__ int index;
	__shared__ int4 fgSumUpToIndex;//index�ȉ��a

	int f;
	int g[3];

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram4[i] = make_int4(0, 0, 0, 0);
		}
		fgSumUpToIndex = make_int4(0, 0, 0, 0);
		index = tex2D<int>(texF[numOfFrames / 2], x, 0);//current index �{���͏������S�t���[��index���w�肵����(���ꂾ�ƒ[�̂Ƃ��ɁA�������S�ł͂Ȃ��Ȃ�)
	}
	//thread����
	__syncthreads();

	//1�ڃq�X�g�O�����`��
	for (int k = 0, n = 0; k < numOfFrames; k++, n+=3)
	{
		for (int yy = -radius; yy <= radius; yy++)
		{
			//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
			{
				f = tex2D<int>(texF[k], xPos, yy);
				g[0] = tex2D<int>(texG[n], xPos, yy);
				g[1] = tex2D<int>(texG[n+1], xPos, yy);
				g[2] = tex2D<int>(texG[n+2], xPos, yy);
				atomicAdd(&histogram4[f].x, 1);
				atomicAdd(&histogram4[f].y, g[0]);
				atomicAdd(&histogram4[f].z, g[1]);
				atomicAdd(&histogram4[f].w, g[2]);
				if (f <= index)
				{
					atomicAdd(&fgSumUpToIndex.x, 1);
					atomicAdd(&fgSumUpToIndex.y, g[0]);
					atomicAdd(&fgSumUpToIndex.z, g[1]);
					atomicAdd(&fgSumUpToIndex.w, g[2]);
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
		float4 cxdx = *((float4*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		for (int k = 0, n = 0; k < numOfFrames; k++, n+=3)
		{
			//�q�X�g�O�����ɒǉ�
			f = tex2D<int>(texF[k], xPos, y + radius);
			g[0] = tex2D<int>(texG[n], xPos, y + radius);
			g[1] = tex2D<int>(texG[n+1], xPos, y + radius);
			g[2] = tex2D<int>(texG[n+2], xPos, y + radius);
			atomicAdd(&histogram4[f].x, 1);
			atomicAdd(&histogram4[f].y, g[0]);
			atomicAdd(&histogram4[f].z, g[1]);
			atomicAdd(&histogram4[f].w, g[2]);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g[0]);
				atomicAdd(&fgSumUpToIndex.z, g[1]);
				atomicAdd(&fgSumUpToIndex.w, g[2]);
			}
			//�q�X�g�O��������폜
			f = tex2D<int>(texF[k], xPos, y - radius - 1);
			g[0] = tex2D<int>(texG[n], xPos, y - radius - 1);
			g[1] = tex2D<int>(texG[n+1], xPos, y - radius - 1);
			g[2] = tex2D<int>(texG[n+2], xPos, y - radius - 1);
			atomicSub(&histogram4[f].x, 1);
			atomicSub(&histogram4[f].y, g[0]);
			atomicSub(&histogram4[f].z, g[1]);
			atomicSub(&histogram4[f].w, g[2]);
			if (f <= index)
			{
				atomicSub(&fgSumUpToIndex.x, 1);
				atomicSub(&fgSumUpToIndex.y, g[0]);
				atomicSub(&fgSumUpToIndex.z, g[1]);
				atomicSub(&fgSumUpToIndex.w, g[2]);
			}
		}

		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			float4 cxdx = *((float4*)((char*)CxDx + y * pitchF4) + x);
			//�����l�v�Z
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
		}
		//thread����
		__syncthreads();
	}
}


//I1G1
void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, std::vector<int*>f, std::vector<int*> g, float2* cxdx)
{
	int numOfFrames = f.size();

	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter3D << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF.device, texG.device, numOfFrames, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
}

//I1G3
void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, std::vector<int*>f, std::vector<DeviceArray<int>*> g, float4* cxdx)
{
	int numOfFrames = f.size();

	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter3D << <gridSizeX, blockSize, Imax * sizeof(int4), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF.device, texG.device, numOfFrames, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
}

//I3G1
void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, std::vector<DeviceArray<int>*>f, std::vector<int*> g, float2* cxdx)
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
	int gridSizeX = sizeInfo.width;
	for (int i = 0; i < 3; i++)
	{
		de_filter3D << <gridSizeX, blockSize, Imax * sizeof(int) * 2, streams[i] >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF[i]->device, texG.device, numOfFrames, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
	}
	cudaDeviceSynchronize();
	for (int i = 0; i < 3; i++)
	{
		delete texF[i];
	}
}

//I3G3
void cu_filter3D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, std::vector<DeviceArray<int>*>f, std::vector<DeviceArray<int>*> g, float4* cxdx)
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
	int gridSizeX = sizeInfo.width;
	for (int i = 0; i < 3; i++)
	{
		de_filter3D << <gridSizeX, blockSize, Imax * sizeof(int) * 4, streams[i] >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF[i]->device, texG.device, numOfFrames, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
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
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

//	*((int*)((char*)dst + y * pitchI1) + x) = (y + x) * 256;
	//*((int*)((char*)dst + y * pitchI1) + x) = (threadIdx.x) * 256;
	printf("%d ", *((int*)((char*)dst + y * pitchI1) + x));
}


//block, grid test
void cu_testBlockGrid()
{

}

/*
�X���b�h�͍ŏ��̏����P�ʂł���A�Ⴆ�Ίe�s�ɂ��čs�����ɘA���I�ɏ���������ꍇ�A�X���b�h���͉摜�������ɂȂ�B
�Ⴆ�Ή摜������512�������Ƃ��悤�B
���̂Ƃ��A�q�X�g�O�����𑜓x��256���Ƃ���ƁA�����X���b�h�����ׂċN���ł����Ƃ���ƁA�q�X�g�O�����̋L�^��512*256*2(f,g)*4byte(int�^)�̃��������K�v�ƂȂ�B

���X���b�h�� = �O���b�h�� �~ �u���b�N��

�Ȃ̂ŁA��{�I�ɂ͓K���ɃO���b�h�����u���b�N�������߂�΂悢�B

�X���b�h�͂��������ɋN������̂��B


�A������X���b�h���A�����郁�����ɃA�N�Z�X����ƌ������ǂ��iCPU�Ƃ͈قȂ锭�z�j
�܂�A�e�X���b�h���s�����ɃA�N�Z�X����̂͌����������Ƃ������ƁB
�Ƃ������Ƃ́A��܂��͍s�P�ʂ̏���������ꍇ�́A�ǂ���ł������Ȃ������ɂ��ׂ��ł��邱�Ƃ�������B
����ɂ��ׂ�����̃X���b�h�i���A������X���b�h�j���A�����郁�����ɃA�N�Z�X���邱�ƂɂȂ�B
�i�R�A���X�A�N�Z�X�j

�u���b�N�T�C�Y�̌��ߕ�
�EOccupancy�Ȃ�ׂ��P�O�O���ɂ���
�E�u���b�N������̃X���b�h���͂Ȃ�ׂ�������
�E�������̓R�A���X�A�N�Z�X�B�Ȃ�ׂ���������B

�u���b�N�T�C�Y��128thread�̔{���ł���Ɨǂ�(���Ȃ��Ƃ��R�Q�̔{��)�B


�u���b�N��SMX(streaming multiprocessor extreme)��Ŏ��s�����B
SM�̒��ɂ�1�ȏ�̃u���b�N������A���������B
K40���ɂ���ƁASM��15����A�P������U�SK��32bit���W�X�^�������Ă���A�܂荇�v��3840KByte�̗e�ʂ̃��W�X�^�������Ă���B
�P�̃X���b�h���g���郌�W�X�^����Kelper�Ȃ�U�R�B

���̂��Ƃ���A�q�X�g�O���������W�X�^�Ɏ�������͖̂����ł���A���Ԃ�L1�L���b�V���Ƃ��Ɋm�ۂ���Ă���̂��낤�B
L1��shared�������͓����ꏊ�H�Ȃ̂ŁA�킴�킴shared���������g�������b�g�͂Ȃ��B
�Ǝv������A�����ϐ�������ƌ����ĕK������L1�ɃL���b�V������Ȃ���������Ȃ��炵���B
shared�������͖����I�ɂ��̏ꏊ�Ɋm�ۂł���Ƃ����_�ŁA�m���ɃA�N�Z�X����邱�Ƃ��������Ă�����̂�shared�������ɒu�����ق���������������Ȃ��B



shared�������̎g�p�B
�P�u���b�N���Ƃ�48�`64KB�܂Ŋm�ۂł���B
�q�X�g�O�����͂P�񂲂Ƃ�
256*2(f,g)*4byte(int�^)=2KB
�K�v�ŁA�u���b�N�T�C�Y��32���Ƃ��̎��_�ŃI�[�o�[����B
���̃T�C�Y������ɁA�q�X�g�O�����͊m�ۂ��Ă�L1�ɂ̓L���b�V������Ȃ��̂��B



�����ɂ���ɂ�host to device�̃������]�����܂Ƃ߂čs���B
�i�Ƃ����Ă�����͂���ȂɂȂ����H �摜���炢���j



���W�A���T���H���́A�T�����ɌŒ�l�������̂Ƒ��x�ς��Ȃ������B�܂胁�W�A���T���͍̂��̃A���S���Y���ł͂قƂ�ǎ��Ԃ�������Ȃ��B
�Z�k�ł������Ȃ̂́A�������A�N�Z�X��X���b�h�����Ȃǂ��炢���B

�e�N�X�`���A�N�Z�X��float�ł͂Ȃ�int�ɂ���(+0.5f���Ȃ�����int�^�j���S���e���Ȃ��B
histogram��int2����Ɨ������Q�ɕ��������S�����x�ς�炸�B

�����x�z�I�Ȃ񂾁H
���W�A���T�����AatomicAdd����������2/3�ɂȂ����B���ꂪ���E�����Aadd�������Ƃ������Ƃ̓q�X�g�O�����X�V���Ȃ����ƂɂȂ�̂ŁA�����͂����Ȃ��B
�Ȃ�ׂ�atomic���Z�����炷���@���l����B
�ˊ撣�����������B

*/


#if 0

__global__ void
de_gsum_y_fast(int width, int height, int radius, int4* sumG, cudaTextureObject_t texG, size_t pitchI4)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;


	int g;
	int _g;
	int sumg = 0.0f;
	int sumgg = 0.0f;

	//y=0
	for (int y = -radius; y <= radius; y++)
	{
		g = tex2D<int>(texG, float(x) + 0.5f, float(y) + 0.5f);
		sumg += g;
		sumgg += g * g;
	}
	*((int4*)((char*)sumG) + x) = make_int4(sumg, sumgg, 0, 0);

	for (int y = 1; y < height; y++)
	{
		g = tex2D<int>(texG, float(x) + 0.5f, float(y + radius) + 0.5f);
		_g = tex2D<int>(texG, float(x) + 0.5f, float(y - radius - 1) + 0.5f);
		sumg += g - _g;
		sumgg += g * g - _g * _g;
		*((int4*)((char*)sumG + y * pitchI4) + x) = make_int4(sumg, sumgg, 0, 0);
	}
}


//�Ӗ��Ȃ�����

//f,g������ ��L�������߂�
__global__ void
de_filter2D_shared3(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//�����Ώۂ̒��S���W�� blockDim.x * blockIdx.x + threadIdx.x �ɂȂ�B
	//
	/*
	* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
	* ���Sthread��idx��radius
	*/

	int diameter = radius * 2 + 1;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= width)
		return;
	//���S���W����̑��Έʒu��\��tid��threadIdx.y�ɂȂ�
	int tid = threadIdx.y;
	if (tid >= diameter)
		return;
	int xPos = x + tid - radius;

	__shared__ int2 histogram[8][256];//4�͌��ߑł�(base 128 ��r=15�̂Ƃ�)�������̂����ςɂ���
	__shared__ int index[8];
	__shared__ int2 fgSumUpToIndex[8];//index�ȉ��a
	//[4]�̂ǂ���g�p����̂��́AthreadIdx.x�ł��܂�
	int column = threadIdx.x;


	int f;
	int g;
	float2 cxdx;

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram[column][i] = make_int2(0, 0);
		}
		fgSumUpToIndex[column] = make_int2(0, 0);
		index[column] = tex2D<int>(texF, float(x) + 0.5f, 0.5f);//current index
		cxdx = *((float2*)((char*)CxDx) + x);
	}
	//thread����
	__syncthreads();

	//1�ڃq�X�g�O�����`��
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		{
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(yy) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(yy) + 0.5f);
			atomicAdd(&histogram[column][f].x, 1);
			atomicAdd(&histogram[column][f].y, g);
			if (f <= index[column])
			{
				atomicAdd(&fgSumUpToIndex[column].x, 1);
				atomicAdd(&fgSumUpToIndex[column].y, g);
			}
		}
	}
	//thread����
	__syncthreads();
	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[column], fgSumUpToIndex[column], index[column]);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		//�q�X�g�O�����ɒǉ�
		f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius) + 0.5f);
		g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius) + 0.5f);
		atomicAdd(&histogram[column][f].x, 1);
		atomicAdd(&histogram[column][f].y, g);
		if (f <= index[column])
		{
			atomicAdd(&fgSumUpToIndex[column].x, 1);
			atomicAdd(&fgSumUpToIndex[column].y, g);
		}
		//�q�X�g�O��������폜
		f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius - 1) + 0.5f);
		g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius - 1) + 0.5f);
		atomicSub(&histogram[column][f].x, 1);
		atomicSub(&histogram[column][f].y, g);
		if (f <= index[column])
		{
			atomicSub(&fgSumUpToIndex[column].x, 1);
			atomicSub(&fgSumUpToIndex[column].y, g);
		}

		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//�����l�v�Z
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[column], fgSumUpToIndex[column], index[column]);
		}
		//thread����
		__syncthreads();
	}
}



//f,g������
void cu_filter2D_shared3(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureObject_t texF, texG;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	//texF��f���o�C���h
	Utility::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);
	//texG��g���o�C���h
	Utility::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);

	//�c������ shared
	/*
	��L�������߂邱�Ƃ��l����B
	���̎����̏ꍇ�AblockSize���t�B���^���a�ɂȂ��Ă���B
	blockSize��32�̔{�����D�܂����A����128�̔{�����ǂ�(���R�͂悭�킩��Ȃ�)�B
	����āA�D�܂����w���blockSize�ɂȂ�悤�ɒ�������B
	*/
	int baseBlockSize = 128;
	const int diameter = radius * 2 + 1;
	//baseBlockSize��diameter�����������ꍇ�͒���
	baseBlockSize *= ((diameter / baseBlockSize) + 1);
	//diameter�ȏ�̍ŏ���2�ׂ̂���̐�
	int blockY = 1;
	while (true)
	{
		blockY *= 2;
		if (blockY >= diameter)
			break;
	}
	//baseBlockSize��blockY���������邩
	int blockX = baseBlockSize / blockY;
	dim3 blockSize = dim3(blockX, blockY, 1);
	//printf("%d %d\n", blockX, blockY);
	//int gridSizeX = sizeInfo.width;// ceil(sizeInfo.width / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockX);
	de_filter2D_shared3 << <gridSizeX, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}




//�ߋ�����


__global__ void
de_filter2D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//�����Ώۂ̒��S���W�� blockIdx.x �ɂȂ�B
	/*
	* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
	* ���Sthread��idx��radius
	*/

	int diameter = radius * 2 + 1;

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= diameter)
		return;
	int xPos = x + tid - radius;

	extern __shared__ int2 histogram[];
	__shared__ int index;
	__shared__ int2 fgSumUpToIndex;//index�ȉ��a


	int f;
	int g;
	float2 cxdx;

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
		index = tex2D<int>(texF, float(x) + 0.5f, 0.5f);//current index
		cxdx = *((float2*)((char*)CxDx) + x);
	}
	//thread����
	__syncthreads();

	//1�ڃq�X�g�O�����`��
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		{
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(yy) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(yy) + 0.5f);
			atomicAdd(&histogram[f].x, 1);
			atomicAdd(&histogram[f].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g);
			}
		}
	}
	//thread����
	__syncthreads();
	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx.x, cxdx.y, histogram, fgSumUpToIndex, index);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		//�q�X�g�O�����ɒǉ�
		f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius) + 0.5f);
		g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius) + 0.5f);
		atomicAdd(&histogram[f].x, 1);
		atomicAdd(&histogram[f].y, g);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, g);
		}
		//�q�X�g�O��������폜
		f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius - 1) + 0.5f);
		g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius - 1) + 0.5f);
		atomicSub(&histogram[f].x, 1);
		atomicSub(&histogram[f].y, g);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, g);
		}

		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//�����l�v�Z
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram, fgSumUpToIndex, index);
		}
		//thread����
		__syncthreads();
	}
}





__device__ inline int
de_findMedian(const float& cx, const float& dx, const int2* histogram, int2& fgSumUpToIndex, int& index, const float& half)
{
	const int halfSign = (half > 0) * 2 - 1;


	float h = (cx * fgSumUpToIndex.y + dx * fgSumUpToIndex.x) * halfSign;
	const int flagA = (h < half* halfSign);
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	while (true)
	{
		/*
		if (index >= 255 && !flagA)
			return 255;
		*/
		index += flagA;
		//if(histogram[index].x)
		{
			fgSumUpToIndex.x += histogram[index].x * sign;
			fgSumUpToIndex.y += histogram[index].y * sign;
			h = (cx * fgSumUpToIndex.y + dx * fgSumUpToIndex.x) * halfSign;
			if ((h >= half * halfSign) == flagA)
			{
				//�������̂ł���index��median
				int result_center = index;
				index += flag2;
				return result_center;
			}
			/*
			if (index >= 255)
				return 255;
				*/
		}
		index += flag2;
		/*
		if (index <= 0)
			return 0;
			*/
	}
}

//

__device__ inline int
de_findMedian(const float& cx, const float& dx, const int2* histogramForward, const int2* histogramBackward, int2& fgSumUpToIndexForward, int2& fgSumUpToIndexBackward, int& index, const float& half, const int forwardId, const float forwardWeight, const float backwardWeight)
{
	const int halfSign = (half > 0) * 2 - 1;

	float h = (cx * (fgSumUpToIndexForward.y * forwardWeight + fgSumUpToIndexBackward.y * backwardWeight) + dx * (fgSumUpToIndexForward.x * forwardWeight + fgSumUpToIndexBackward.x * backwardWeight)) * halfSign;
	//float h = (cx * fgSumUpToIndex.y + dx * fgSumUpToIndex.x) * halfSign;
	const int flagA = (h < half* halfSign);
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	while (true)
	{
		/*
		if (index >= 255 && !flagA)
			return 255;
		*/
		index += flagA;
		//if(histogram[index].x)
		{
			fgSumUpToIndexForward.x += histogramForward[index].x * sign;
			fgSumUpToIndexForward.y += histogramForward[index].y * sign;
			fgSumUpToIndexBackward.x += histogramBackward[index].x * sign;
			fgSumUpToIndexBackward.y += histogramBackward[index].y * sign;
			//h = (cx * fgSumUpToIndex.y + dx * fgSumUpToIndex.x) * halfSign;
			h = (cx * (fgSumUpToIndexForward.y * forwardWeight + fgSumUpToIndexBackward.y * backwardWeight) + dx * (fgSumUpToIndexForward.x * forwardWeight + fgSumUpToIndexBackward.x * backwardWeight)) * halfSign;
			if ((h >= half * halfSign) == flagA)
			{
				//�������̂ł���index��median
				int result_center = index;
				index += flag2;
				return result_center;
			}
			/*
			if (index >= 255)
				return 255;
				*/
		}
		index += flag2;
		/*
		if (index <= 0)
			return 0;
			*/
	}
}

__device__ inline int
de_findMedianDebug(const float& cx, const float& dx, const int2* histogram, int2& fgSumUpToIndex, int& index)
{
	float h = cx * fgSumUpToIndex.y + dx * fgSumUpToIndex.x;
	const int flagA = h < 0.5;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	printf("h: %f\n", h);
	while (true)
	{
		index += flagA;
		//if(histogram[index].x > 0)
		{
			fgSumUpToIndex.x += histogram[index].x * sign;
			fgSumUpToIndex.y += histogram[index].y * sign;
			h = cx * fgSumUpToIndex.y + dx * fgSumUpToIndex.x;
			printf("idx: %d, sumF: %d, sumG: %d, h: %f\n", index, fgSumUpToIndex.x, fgSumUpToIndex.y, h);
			if ((h >= 0.5) == flagA)
			{
				//�������̂ł���index��median
				int result_center = index;
				index += flag2;
				return result_center;
			}
		}
		index += flag2;

		if (index >= 255)
		{
			printf("## 255 over : %f\n", h);
			return 255;
		}
		if (index <= -1)
		{
			printf("## -1 under : %f\n", h);
			return -1;
		}
	}
}
//�m�F�p ��fx,y��cx�Adx���v�Z
__device__ inline int
de_calculateCxDxNow(int x, int y, cudaTextureObject_t texFG, int radius, float eps2, float& cx, float& dx)
{
	int2 pix = tex2D<int2>(texFG, float(x) + 0.5f, float(y) + 0.5f);

	int g_center = pix.y;
	int g;
	float pixNumInv = 1.0f / (float)((radius * 2 + 1) * (radius * 2 + 1));
	int gsum = 0;
	int ggsum = 0;
	for (int yy = -radius; yy <= radius; yy++)
	{
		for (int xx = -radius; xx <= radius; xx++)
		{
			pix = tex2D<int2>(texFG, float(x + xx) + 0.5f, float(y + yy) + 0.5f);
			g = pix.y;
			gsum += g;
			ggsum += g * g;
		}
	}
	float gave = gsum * pixNumInv;
	float vx = ggsum * pixNumInv - gave * gave + eps2;
	cx = (g_center - gave) * pixNumInv / vx;
	dx = pixNumInv - gave * cx;
}


__device__ inline int
de_findMedian_fmaf(const float& cx, const float& dx, const int2* histogram, int2& fgSumUpToIndex, int& index)
{
	//float h = cx * fgSumUpToIndex.y + dx * fgSumUpToIndex.x;
	float h = __fmaf_rd(cx, fgSumUpToIndex.y, dx * fgSumUpToIndex.x);
	const int flagA = h < 0.5f;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	while (true)
	{
		index += flagA;
		//if(histogram[index].x)
		{
			fgSumUpToIndex.x += histogram[index].x * sign;
			fgSumUpToIndex.y += histogram[index].y * sign;
			h = __fmaf_rd(cx, fgSumUpToIndex.y, dx * fgSumUpToIndex.x);
			if ((h >= 0.5f) == flagA)
			{
				//�������̂ł���index��median
				int result_center = index;
				index += flag2;
				return result_center;
			}
		}
		index += flag2;
	}
}

//�q�X�g�O�����̃T���v�����O���Ԉ���
/*
h�̘a��1�ł��邱�Ƃ��������Ȃ��Ȃ�̂�half=0.5�����藧�����A255�𒴂���ӏ����o�Ă��܂��̂ł��̂܂܂ł͎g���Ȃ��B
���v�l���X�V���Ă��������B
�ł������A�q�X�g�O�����X�V���a�Ȃ̂Ŗ͗l���ڗ��B
*/
__global__ void
de_filter2D_histogramSampling(int width, int height, int samplingRate, float weightingFactor, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//�����Ώۂ̒��S���W�� blockIdx.x �ɂȂ�B
	/*
	* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
	* ���Sthread��idx��radius
	*/

	//��f�̃T���v���͈͂́}radius�����A

	int centerPos = radius / samplingRate;

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + (tid - centerPos) * samplingRate;//

	extern __shared__ int2 histogram[];
	__shared__ int index;
	__shared__ int2 fgSumUpToIndex;//index�ȉ��a
	__shared__ int2 fgSumUpToAllIndex;//���v�l
	__shared__ float2 cxdx;


	int f;
	int g;

	//���S�X���b�h�̂ݎ��s
	if (tid == centerPos)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
		fgSumUpToAllIndex = make_int2(0, 0);
		index = tex2D<int>(texF, float(x) + 0.5f, 0.5f);//current index
		cxdx = *((float2*)((char*)CxDx) + x);
	}
	//thread����
	__syncthreads();

	//1�ڃq�X�g�O�����`��
	for (int yy = -radius; yy <= radius; yy += samplingRate)//
	{
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		{
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(yy) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(yy) + 0.5f);
			atomicAdd(&histogram[f].x, 1);
			atomicAdd(&histogram[f].y, g);
			atomicAdd(&fgSumUpToAllIndex.x, 1);
			atomicAdd(&fgSumUpToAllIndex.y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g);
			}
		}
	}
	//thread����
	__syncthreads();
	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == centerPos)
	{
		float adjustedCx = cxdx.x * weightingFactor;
		float adjustedDx = cxdx.y * weightingFactor;
		float half = (adjustedCx * fgSumUpToAllIndex.y + adjustedDx * fgSumUpToAllIndex.x) * 0.5f;
		*((int*)((char*)result_center) + x) = de_findMedian(adjustedCx, adjustedDx, histogram, fgSumUpToIndex, index, half);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		if (y % samplingRate == 0)
		{
			//�q�X�g�O�����X�V
			//�q�X�g�O�����ɒǉ�
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius) + 0.5f);
			atomicAdd(&histogram[f].x, 1);
			atomicAdd(&histogram[f].y, g);
			atomicAdd(&fgSumUpToAllIndex.x, 1);
			atomicAdd(&fgSumUpToAllIndex.y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g);
			}
			//�q�X�g�O��������폜
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius - samplingRate) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius - samplingRate) + 0.5f);
			atomicSub(&histogram[f].x, 1);
			atomicSub(&histogram[f].y, g);
			atomicSub(&fgSumUpToAllIndex.x, 1);
			atomicSub(&fgSumUpToAllIndex.y, g);
			if (f <= index)
			{
				atomicSub(&fgSumUpToIndex.x, 1);
				atomicSub(&fgSumUpToIndex.y, g);
			}

		}
		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == centerPos)
		{
			cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//�����l�v�Z
			float adjustedCx = cxdx.x * weightingFactor;
			float adjustedDx = cxdx.y * weightingFactor;
			float half = (adjustedCx * fgSumUpToAllIndex.y + adjustedDx * fgSumUpToAllIndex.x) * 0.5f;
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(adjustedCx, adjustedDx, histogram, fgSumUpToIndex, index, half);
		}
		//thread����
		__syncthreads();
	}
}


//��L�������̂��߂ɁA�q�X�g�O������2�p�ӂ��ďd�ݕt���ŗp����
__global__ void
de_filter2D_histogramSampling2(int width, int height, int samplingRate, float weightingFactor, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//�����Ώۂ̒��S���W�� blockIdx.x �ɂȂ�B
	/*
	* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
	* ���Sthread��idx��radius
	*/

	//��f�̃T���v���͈͂́}radius�����A

	int centerPos = radius / samplingRate;

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= centerPos * 2 + 1)
		return;
	int xPos = x + (tid - centerPos) * samplingRate;//

	__shared__ int2 histogram[2][256];
	__shared__ int index;
	__shared__ int2 fgSumUpToIndex[2];//index�ȉ��a
	__shared__ int2 fgSumUpToAllIndex[2];//���v�l
	__shared__ float2 cxdx;

	//���W�I�ɑ傫���ق��̃q�X�g�O�����C���f�b�N�X�i0 or 1�j
	__shared__ int forwardId;

	int f;
	int g;

	//���S�X���b�h�̂ݎ��s
	if (tid == centerPos)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram[0][i] = make_int2(0, 0);
			histogram[1][i] = make_int2(0, 0);
		}
		fgSumUpToIndex[0] = make_int2(0, 0);
		fgSumUpToIndex[1] = make_int2(0, 0);
		fgSumUpToAllIndex[0] = make_int2(0, 0);
		fgSumUpToAllIndex[1] = make_int2(0, 0);
		index = tex2D<int>(texF, float(x) + 0.5f, 0.5f);//current index
		cxdx = *((float2*)((char*)CxDx) + x);
	}
	//thread����
	__syncthreads();

	//1�ڃq�X�g�O�����`��
	for (int yy = -radius; yy <= radius; yy += samplingRate)//
	{
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		{
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(yy) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(yy) + 0.5f);
			atomicAdd(&histogram[0][f].x, 1);
			atomicAdd(&histogram[0][f].y, g);
			atomicAdd(&fgSumUpToAllIndex[0].x, 1);
			atomicAdd(&fgSumUpToAllIndex[0].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[0].x, 1);
				atomicAdd(&fgSumUpToIndex[0].y, g);
			}
		}
	}
	//thread����
	__syncthreads();
	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == centerPos)
	{
		float half = (cxdx.x * fgSumUpToAllIndex[0].y + cxdx.y * fgSumUpToAllIndex[0].x) * 0.5f;
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[0], fgSumUpToIndex[0], index, half);
		forwardId = 1;
	}
	//thread����
	__syncthreads();

	//���̃q�X�g�O�����`��
	for (int yy = -radius; yy <= radius; yy += samplingRate)//
	{
		{
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(yy + samplingRate) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(yy + samplingRate) + 0.5f);
			atomicAdd(&histogram[1][f].x, 1);
			atomicAdd(&histogram[1][f].y, g);
			atomicAdd(&fgSumUpToAllIndex[1].x, 1);
			atomicAdd(&fgSumUpToAllIndex[1].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[1].x, 1);
				atomicAdd(&fgSumUpToIndex[1].y, g);
			}
		}
	}


	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		if (y % samplingRate == 0)
		{
			if (tid == centerPos)
			{
				//forward��p���Ē����l�̌v�Z
				//float half = (cxdx.x * fgSumUpToAllIndex[forwardId].y + cxdx.y * fgSumUpToAllIndex[forwardId].x) * 0.5f;
				//*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], fgSumUpToIndex[forwardId], index, half);

				float forwardWeight = 1.0f;
				float backwardWeight = 0.0f;

				float half = (cxdx.x * (fgSumUpToAllIndex[forwardId].y * forwardWeight + fgSumUpToAllIndex[!forwardId].y * backwardWeight) + cxdx.y * (fgSumUpToAllIndex[forwardId].x * forwardWeight + fgSumUpToAllIndex[!forwardId].x * backwardWeight)) * 0.5f;
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], histogram[!forwardId], fgSumUpToIndex[forwardId], fgSumUpToIndex[!forwardId], index, half, forwardId, forwardWeight, backwardWeight);

				//forward�̓���ւ�
				forwardId = !forwardId;
			}

			//thread����
			__syncthreads();

			//�q�X�g�O�����X�V backward�̂�2�񑊓����X�V����forward, backward�����ւ���
			//�q�X�g�O�����ɒǉ�
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius) + 0.5f);
			atomicAdd(&histogram[forwardId][f].x, 1);
			atomicAdd(&histogram[forwardId][f].y, g);
			atomicAdd(&fgSumUpToAllIndex[forwardId].x, 1);
			atomicAdd(&fgSumUpToAllIndex[forwardId].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[forwardId].x, 1);
				atomicAdd(&fgSumUpToIndex[forwardId].y, g);
			}
			//�q�X�g�O��������폜
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius - samplingRate) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius - samplingRate) + 0.5f);
			atomicSub(&histogram[forwardId][f].x, 1);
			atomicSub(&histogram[forwardId][f].y, g);
			atomicSub(&fgSumUpToAllIndex[forwardId].x, 1);
			atomicSub(&fgSumUpToAllIndex[forwardId].y, g);
			if (f <= index)
			{
				atomicSub(&fgSumUpToIndex[forwardId].x, 1);
				atomicSub(&fgSumUpToIndex[forwardId].y, g);
			}

			//�q�X�g�O�����ɒǉ�
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius + samplingRate) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius + samplingRate) + 0.5f);
			atomicAdd(&histogram[forwardId][f].x, 1);
			atomicAdd(&histogram[forwardId][f].y, g);
			atomicAdd(&fgSumUpToAllIndex[forwardId].x, 1);
			atomicAdd(&fgSumUpToAllIndex[forwardId].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[forwardId].x, 1);
				atomicAdd(&fgSumUpToIndex[forwardId].y, g);
			}
			//�q�X�g�O��������폜
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius) + 0.5f);
			atomicSub(&histogram[forwardId][f].x, 1);
			atomicSub(&histogram[forwardId][f].y, g);
			atomicSub(&fgSumUpToAllIndex[forwardId].x, 1);
			atomicSub(&fgSumUpToAllIndex[forwardId].y, g);
			if (f <= index)
			{
				atomicSub(&fgSumUpToIndex[forwardId].x, 1);
				atomicSub(&fgSumUpToIndex[forwardId].y, g);
			}


		}
		else
		{
			//�d�ݕt�������l�̌v�Z
			//���S�X���b�h�̂ݎ��s
			if (tid == centerPos)
			{
				cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);

				//�d�݂̌v�Z
				int l1 = y % samplingRate;//backward����̋���
				//int l2 = samplingRate - l1;//forward�܂ł̋���
				float forwardWeight = l1 / (float)samplingRate;
				float backwardWeight = 1 - forwardWeight;

				//�����l�v�Z
				float half = (cxdx.x * (fgSumUpToAllIndex[forwardId].y * forwardWeight + fgSumUpToAllIndex[!forwardId].y * backwardWeight) + cxdx.y * (fgSumUpToAllIndex[forwardId].x * forwardWeight + fgSumUpToAllIndex[!forwardId].x * backwardWeight)) * 0.5f;
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], histogram[!forwardId], fgSumUpToIndex[forwardId], fgSumUpToIndex[!forwardId], index, half, forwardId, forwardWeight, backwardWeight);
			}


		}


		//thread����
		__syncthreads();
	}
}


//�����s���̐����o��̂�x�����͊Ԉ����Ȃ�
__global__ void
de_filter2D_histogramSampling3(int width, int height, int samplingRate, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//�����Ώۂ̒��S���W�� blockIdx.x �ɂȂ�B
	/*
	* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
	* ���Sthread��idx��radius
	*/

	//��f�̃T���v���͈͂́}radius�����A

	int centerPos = radius;

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= centerPos * 2 + 1)
		return;
	int xPos = x + (tid - centerPos);//

	__shared__ int2 histogram[2][256];
	__shared__ int index;
	__shared__ int2 fgSumUpToIndex[2];//index�ȉ��a
	__shared__ int2 fgSumUpToAllIndex[2];//���v�l
	__shared__ float2 cxdx;

	//���W�I�ɑ傫���ق��̃q�X�g�O�����C���f�b�N�X�i0 or 1�j
	__shared__ int forwardId;

	int f;
	int g;

	//���S�X���b�h�̂ݎ��s
	if (tid == centerPos)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram[0][i] = make_int2(0, 0);
			histogram[1][i] = make_int2(0, 0);
		}
		fgSumUpToIndex[0] = make_int2(0, 0);
		fgSumUpToIndex[1] = make_int2(0, 0);
		fgSumUpToAllIndex[0] = make_int2(0, 0);
		fgSumUpToAllIndex[1] = make_int2(0, 0);
		index = tex2D<int>(texF, float(x) + 0.5f, 0.5f);//current index
		cxdx = *((float2*)((char*)CxDx) + x);
	}
	//thread����
	__syncthreads();

	//1�ڃq�X�g�O�����`��
	for (int yy = -radius; yy <= radius; yy += samplingRate)//
	{
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		{
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(yy) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(yy) + 0.5f);
			atomicAdd(&histogram[0][f].x, 1);
			atomicAdd(&histogram[0][f].y, g);
			atomicAdd(&fgSumUpToAllIndex[0].x, 1);
			atomicAdd(&fgSumUpToAllIndex[0].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[0].x, 1);
				atomicAdd(&fgSumUpToIndex[0].y, g);
			}
		}
	}
	//thread����
	__syncthreads();
	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == centerPos)
	{
		float half = (cxdx.x * fgSumUpToAllIndex[0].y + cxdx.y * fgSumUpToAllIndex[0].x) * 0.5f;
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[0], fgSumUpToIndex[0], index, half);
		forwardId = 1;
	}
	//thread����
	__syncthreads();

	//���̃q�X�g�O�����`��
	for (int yy = -radius; yy <= radius; yy += samplingRate)//
	{
		{
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(yy + samplingRate) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(yy + samplingRate) + 0.5f);
			atomicAdd(&histogram[1][f].x, 1);
			atomicAdd(&histogram[1][f].y, g);
			atomicAdd(&fgSumUpToAllIndex[1].x, 1);
			atomicAdd(&fgSumUpToAllIndex[1].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[1].x, 1);
				atomicAdd(&fgSumUpToIndex[1].y, g);
			}
		}
	}


	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		if (y % samplingRate == 0)
		{
			if (tid == centerPos)
			{
				//forward��p���Ē����l�̌v�Z
				//float half = (cxdx.x * fgSumUpToAllIndex[forwardId].y + cxdx.y * fgSumUpToAllIndex[forwardId].x) * 0.5f;
				//*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], fgSumUpToIndex[forwardId], index, half);

				float forwardWeight = 1.0f;
				float backwardWeight = 0.0f;

				float half = (cxdx.x * (fgSumUpToAllIndex[forwardId].y * forwardWeight + fgSumUpToAllIndex[!forwardId].y * backwardWeight) + cxdx.y * (fgSumUpToAllIndex[forwardId].x * forwardWeight + fgSumUpToAllIndex[!forwardId].x * backwardWeight)) * 0.5f;
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], histogram[!forwardId], fgSumUpToIndex[forwardId], fgSumUpToIndex[!forwardId], index, half, forwardId, forwardWeight, backwardWeight);

				//forward�̓���ւ�
				forwardId = !forwardId;
			}

			//thread����
			__syncthreads();

			//�q�X�g�O�����X�V backward�̂�2�񑊓����X�V����forward, backward�����ւ���
			//�q�X�g�O�����ɒǉ�
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius) + 0.5f);
			atomicAdd(&histogram[forwardId][f].x, 1);
			atomicAdd(&histogram[forwardId][f].y, g);
			atomicAdd(&fgSumUpToAllIndex[forwardId].x, 1);
			atomicAdd(&fgSumUpToAllIndex[forwardId].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[forwardId].x, 1);
				atomicAdd(&fgSumUpToIndex[forwardId].y, g);
			}
			//�q�X�g�O��������폜
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius - samplingRate) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius - samplingRate) + 0.5f);
			atomicSub(&histogram[forwardId][f].x, 1);
			atomicSub(&histogram[forwardId][f].y, g);
			atomicSub(&fgSumUpToAllIndex[forwardId].x, 1);
			atomicSub(&fgSumUpToAllIndex[forwardId].y, g);
			if (f <= index)
			{
				atomicSub(&fgSumUpToIndex[forwardId].x, 1);
				atomicSub(&fgSumUpToIndex[forwardId].y, g);
			}

			//�q�X�g�O�����ɒǉ�
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius + samplingRate) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius + samplingRate) + 0.5f);
			atomicAdd(&histogram[forwardId][f].x, 1);
			atomicAdd(&histogram[forwardId][f].y, g);
			atomicAdd(&fgSumUpToAllIndex[forwardId].x, 1);
			atomicAdd(&fgSumUpToAllIndex[forwardId].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[forwardId].x, 1);
				atomicAdd(&fgSumUpToIndex[forwardId].y, g);
			}
			//�q�X�g�O��������폜
			f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius) + 0.5f);
			g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius) + 0.5f);
			atomicSub(&histogram[forwardId][f].x, 1);
			atomicSub(&histogram[forwardId][f].y, g);
			atomicSub(&fgSumUpToAllIndex[forwardId].x, 1);
			atomicSub(&fgSumUpToAllIndex[forwardId].y, g);
			if (f <= index)
			{
				atomicSub(&fgSumUpToIndex[forwardId].x, 1);
				atomicSub(&fgSumUpToIndex[forwardId].y, g);
			}


		}
		else
		{
			//�d�ݕt�������l�̌v�Z
			//���S�X���b�h�̂ݎ��s
			if (tid == centerPos)
			{
				cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);

				//�d�݂̌v�Z
				int l1 = y % samplingRate;//backward����̋���
				int l2 = samplingRate - l1;//forward�܂ł̋���
				float forwardWeight = l1 / (float)samplingRate;
				float backwardWeight = 1 - forwardWeight;

				//�F�d�݂��Ă݂�H
				g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - l1) + 0.5f);
				f = tex2D<int>(texG, float(xPos) + 0.5f, float(y + l2) + 0.5f);
				int k = tex2D<int>(texG, float(xPos) + 0.5f, float(y) + 0.5f);
				int d1 = abs(k - g);
				int d2 = abs(k - f);
				if (d1 != 0 && d2 != 0)
				{
					forwardWeight = d1 / (float)(d1 + d2);
				}
				else
				{
					forwardWeight = 0.5f;
				}
				backwardWeight = 1 - forwardWeight;


				//�����l�v�Z
				float half = (cxdx.x * (fgSumUpToAllIndex[forwardId].y * forwardWeight + fgSumUpToAllIndex[!forwardId].y * backwardWeight) + cxdx.y * (fgSumUpToAllIndex[forwardId].x * forwardWeight + fgSumUpToAllIndex[!forwardId].x * backwardWeight)) * 0.5f;
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], histogram[!forwardId], fgSumUpToIndex[forwardId], fgSumUpToIndex[!forwardId], index, half, forwardId, forwardWeight, backwardWeight);
			}


		}


		//thread����
		__syncthreads();
	}
}



//�q�X�g�O�����T���v�����O
void cu_filter2D_histogramSampling(SizeInfo& sizeInfo, cudaStream_t stream, int samplingRate, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureObject_t texF, texG;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	Utility::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texF��f���o�C���h
	Utility::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texG��g���o�C���h

	//���a��samplingRate�̔{���ɂȂ�悤�ɕύX����
	int adjustedRadius = radius - radius % samplingRate;
	int originalDiameter = (radius * 2 + 1);
	int sampledLength = (adjustedRadius / samplingRate) * 2 + 1;
	//�T���v�����O�l��
	/*
	int blockSize = sampledLength;//
	int gridSizeX = sizeInfo.width;
	de_filter2D_histogramSampling2 << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, samplingRate, weightingFactor, adjustedRadius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
	*/
	int blockSize = adjustedRadius * 2 + 1;//
	int gridSizeX = sizeInfo.width;
	de_filter2D_histogramSampling3 << <gridSizeX, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, samplingRate, adjustedRadius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());


	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}

//�]�����Ԃ�������f�݂̂ł����x��1%���������Ȃ�Ȃ�����
//f�̂݁i�Z���t�K�C�h�j
__global__ void
de_filter2D_selfGuide(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//�����Ώۂ̒��S���W�� blockIdx.x �ɂȂ�B
	/*
	* thread��0�`radius*2 �܂Ŏg�p����Ƃ��āA����ȏ��return����
	* ���Sthread��idx��radius
	*/

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + tid - radius;

	extern __shared__ int2 histogram[];
	__shared__ int index;
	__shared__ int2 fgSumUpToIndex;//index�ȉ��a

	int f;

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
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
			atomicAdd(&histogram[f].x, 1);
			atomicAdd(&histogram[f].y, f);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, f);
			}
		}
	}
	//thread����
	__syncthreads();
	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		float2 cxdx = *((float2*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram, fgSumUpToIndex, index);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		//�q�X�g�O�����ɒǉ�
		f = tex2D<int>(texF, xPos, y + radius);
		atomicAdd(&histogram[f].x, 1);
		atomicAdd(&histogram[f].y, f);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, f);
		}
		//�q�X�g�O��������폜
		f = tex2D<int>(texF, xPos, y - radius - 1);
		atomicSub(&histogram[f].x, 1);
		atomicSub(&histogram[f].y, f);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, f);
		}

		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			float2 cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram, fgSumUpToIndex, index);
		}
		//thread����
		__syncthreads();
	}
}



//f�̂�(�Z���t�K�C�h�j
void cu_filter2D_selfGuide(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, float2* cxdx)
{
	cudaTextureObject_t texF;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	//texF��f���o�C���h
	Utility::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter2D_selfGuide << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	cudaDestroyTextureObject(texF);
}




















//G3
__device__ inline int
de_findMedianDebug(const float4& cxdx, const int4* histogram, int4& fgSumUpToIndex, int& index, int x, int y)
{
	float h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.z + cxdx.z * fgSumUpToIndex.w + cxdx.w * fgSumUpToIndex.x;
	const int flagA = h < 0.5f;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	printf(" - g3 -\n");
	printf("(%d, %d): %f\n", x, y, h);
	printf("%f, %f, %f, %f\n", cxdx.x, cxdx.y, cxdx.z, cxdx.w);
	printf("%d, %d, %d, %d\n\n", fgSumUpToIndex.y, fgSumUpToIndex.z, fgSumUpToIndex.w, fgSumUpToIndex.x);
	while (true)
	{
		index += flagA;
		//if(histogram[index].x)
		{
			fgSumUpToIndex.x += histogram[index].x * sign;
			fgSumUpToIndex.y += histogram[index].y * sign;
			fgSumUpToIndex.z += histogram[index].z * sign;
			fgSumUpToIndex.w += histogram[index].w * sign;
			h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.z + cxdx.z * fgSumUpToIndex.w + cxdx.w * fgSumUpToIndex.x;
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

//GX n��G�̃`�����l����
__device__ inline int
de_findMedianDebug(float*& cxdx, int* histogram, int *& fgSumUpToIndex, int& index, int n, int x, int y)
{
	float h = 0.0f;
	for (int i = 0; i <= n; i++) {
		h += cxdx[i] * fgSumUpToIndex[i];
	}
	const int flagA = h < 0.5f;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	const int k = n + 1;

	printf(" - multi -\n");
	printf("(%d, %d): %f\n", x, y, h);
	printf("%f, %f, %f, %f\n", cxdx[0], cxdx[1], cxdx[2], cxdx[3]);
	printf("%d, %d, %d, %d\n\n", fgSumUpToIndex[0], fgSumUpToIndex[1], fgSumUpToIndex[2], fgSumUpToIndex[3]);

	while (true)
	{
		index += flagA;
		//if(histogram[index].x)//�����̓R�����g�O���Ă���������
		{
			for (int i = 0; i <= n; i++)
				fgSumUpToIndex[i] += histogram[index * k + i] * sign;
			h = 0.0f;
			for (int i = 0; i <= n; i++)
				h += cxdx[i] * fgSumUpToIndex[i];
			if ((h >= 0.5f) == flagA)
			{
				int result_center = index;
				index += flag2;
				return result_center;
			}
		}
		index += flag2;
		if (index >= 255)
		{
			//if (x == 453 && y == 262)
			if (false)
			{
				printf("(%d, %d): %f\n", x, y, h);
				printf("%f, %f, %f, %f\n", cxdx[0], cxdx[1], cxdx[2], cxdx[3]);
				printf("%d, %d, %d, %d\n", fgSumUpToIndex[0], fgSumUpToIndex[1], fgSumUpToIndex[2], fgSumUpToIndex[3]);
			}
			return 255;
		}
		else if (index <= -1)
		{
			printf("-1: %f\n", h);
			return 0;
			//printf("%f, %f, %f, %f\n", cxdx[0], cxdx[1], cxdx[2], cxdx[3]);
		}
	}

}

//�����������s
__device__ inline void
de_findMedianDebug(float*& cxdx2, int* histogramX, int *& fgSumUpToIndex2, int& index2, int n, int x, int y, const float4& cxdx, const int4* histogram, int4& fgSumUpToIndex, int& index, int& result1, int& result2)
{
	//g3
	float h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.z + cxdx.z * fgSumUpToIndex.w + cxdx.w * fgSumUpToIndex.x;
	const int flagA = h < 0.5f;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	printf(" - g3 -\n");
	printf("(%d, %d): %f\n", x, y, h);
	printf("%f, %f, %f, %f\n", cxdx.x, cxdx.y, cxdx.z, cxdx.w);
	printf("%d, %d, %d, %d\n\n", fgSumUpToIndex.y, fgSumUpToIndex.z, fgSumUpToIndex.w, fgSumUpToIndex.x);

	//multi
	float h2 = 0.0f;
	for (int i = 0; i <= n; i++) {
		h2 += cxdx2[i] * fgSumUpToIndex2[i];
	}
	const int flagA2 = h2 < 0.5f;
	const int flag22 = flagA2 - 1;
	const int sign2 = flagA2 * 2 - 1;
	const int k = n + 1;

	printf(" - multi -\n");
	printf("(%d, %d): %f\n", x, y, h2);
	printf("%f, %f, %f, %f\n", cxdx2[0], cxdx2[1], cxdx2[2], cxdx2[3]);
	printf("%d, %d, %d, %d\n\n", fgSumUpToIndex2[0], fgSumUpToIndex2[1], fgSumUpToIndex2[2], fgSumUpToIndex2[3]);

	bool doflag1 = true;
	bool doflag2 = true;
	int saveIndex1;
	int saveIndex2;

	while (doflag1 || doflag2)
	{
		//g3
		if (doflag1)
		{
			index += flagA;
			saveIndex1 = index;
			//if(histogram[index].x)
			{
				fgSumUpToIndex.x += histogram[index].x * sign;
				fgSumUpToIndex.y += histogram[index].y * sign;
				fgSumUpToIndex.z += histogram[index].z * sign;
				fgSumUpToIndex.w += histogram[index].w * sign;
				h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.z + cxdx.z * fgSumUpToIndex.w + cxdx.w * fgSumUpToIndex.x;
				if ((h >= 0.5f) == flagA)
				{
					int result_center = index;
					index += flag2;
					result1 = result_center;
					doflag1 = false;
					//return result_center;
				}
			}
			index += flag2;
		}


		//multi
		if (doflag2)
		{
			index2 += flagA2;
			saveIndex2 = index;
			//if(histogram[index].x)//�����̓R�����g�O���Ă���������
			{
				for (int i = 0; i <= n; i++)
					fgSumUpToIndex2[i] += histogramX[index2 * k + i] * sign2;
				h2 = 0.0f;
				for (int i = 0; i <= n; i++)
					h2 += cxdx2[i] * fgSumUpToIndex2[i];
				if ((h2 >= 0.5f) == flagA2)
				{
					int result_center = index2;
					index2 += flag22;
					result2 = result_center;
					doflag2 = false;
					//return result_center;
				}
			}
			index2 += flag22;
		}

		int4 tmpSum2 = { fgSumUpToIndex2[3], fgSumUpToIndex2[0], fgSumUpToIndex2[1], fgSumUpToIndex2[2] };
		float4 tmpCxdx = { cxdx2[0], cxdx2[1], cxdx2[2], cxdx2[3] };
		int4 tempHist1 = { histogram[saveIndex1].x, histogram[saveIndex1].y, histogram[saveIndex1].z, histogram[saveIndex1].w };
		int4 tempHist2 = { histogramX[saveIndex2 * k + 0], fgSumUpToIndex2[saveIndex2 * k + 1], fgSumUpToIndex2[saveIndex2 * k + 2], fgSumUpToIndex2[saveIndex2 * k + 3] };

		if (doflag1 && doflag2 && (h != h2))
		{
			printf("%d %d %d %d --- %d %d %d %d\n", fgSumUpToIndex.x, fgSumUpToIndex.y, fgSumUpToIndex.z, fgSumUpToIndex.w, fgSumUpToIndex2[3], fgSumUpToIndex2[0], fgSumUpToIndex2[1], fgSumUpToIndex2[2]);
			printf("%f %f %f %f --- %f %f %f %f\n", cxdx.x, cxdx.y, cxdx.z, cxdx.w, cxdx2[0], cxdx2[1], cxdx2[2], cxdx2[3]);
			//histogram
			printf("%d %d %d %d --- %d %d %d %d\n", histogram[saveIndex1].x, histogram[saveIndex1].y, histogram[saveIndex1].z, histogram[saveIndex1].w, histogramX[saveIndex2 * k + 0], fgSumUpToIndex2[saveIndex2 * k + 1], fgSumUpToIndex2[saveIndex2 * k + 2], fgSumUpToIndex2[saveIndex2 * k + 3]);

		}

		printf("g3:%d, %f\tmulti:%d, %f\n", index, h, index2, h2);



		if (index2 >= 255)
		{
			//if (x == 453 && y == 262)
			if (false)
			{
				printf("(%d, %d): %f\n", x, y, h2);
				printf("%f, %f, %f, %f\n", cxdx2[0], cxdx2[1], cxdx2[2], cxdx2[3]);
				printf("%d, %d, %d, %d\n", fgSumUpToIndex2[0], fgSumUpToIndex2[1], fgSumUpToIndex2[2], fgSumUpToIndex2[3]);
			}
			//return 255;
			result2 = 255;
			doflag2 = false;
		}
		else if (index2 <= -1)
		{
			printf("-1: %f\n", h2);
			result2 = 0;
			doflag2 = false;
			//return 0;
			//printf("%f, %f, %f, %f\n", cxdx[0], cxdx[1], cxdx[2], cxdx[3]);
		}
	}


}


__global__ void
de_filter2DDebug(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t* texG3, float4* CxDx, size_t pitchI1, size_t pitchF4)
{
	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + tid - radius;


	__shared__ int index;
	__shared__ int4 fgSumUpToIndex;//index�ȉ��a
	int f;
	int g[3];

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram4[i] = make_int4(0, 0, 0, 0);
		}
		fgSumUpToIndex = make_int4(0, 0, 0, 0);
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
			g[0] = tex2D<int>(texG3[0], xPos, yy);
			g[1] = tex2D<int>(texG3[1], xPos, yy);
			g[2] = tex2D<int>(texG3[2], xPos, yy);
			atomicAdd(&histogram4[f].x, 1);
			atomicAdd(&histogram4[f].y, g[0]);
			atomicAdd(&histogram4[f].z, g[1]);
			atomicAdd(&histogram4[f].w, g[2]);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g[0]);
				atomicAdd(&fgSumUpToIndex.z, g[1]);
				atomicAdd(&fgSumUpToIndex.w, g[2]);
			}

		}
	}
	//thread����
	__syncthreads();

	//1�s�ڂ̒����l�v�Z
	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		float4 cxdx = *((float4*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		//�q�X�g�O�����ɒǉ�
		f = tex2D<int>(texF, xPos, y + radius);
		g[0] = tex2D<int>(texG3[0], xPos, y + radius);
		g[1] = tex2D<int>(texG3[1], xPos, y + radius);
		g[2] = tex2D<int>(texG3[2], xPos, y + radius);
		atomicAdd(&histogram4[f].x, 1);
		atomicAdd(&histogram4[f].y, g[0]);
		atomicAdd(&histogram4[f].z, g[1]);
		atomicAdd(&histogram4[f].w, g[2]);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, g[0]);
			atomicAdd(&fgSumUpToIndex.z, g[1]);
			atomicAdd(&fgSumUpToIndex.w, g[2]);
		}
		//�q�X�g�O��������폜
		f = tex2D<int>(texF, xPos, y - radius - 1);
		g[0] = tex2D<int>(texG3[0], xPos, y - radius - 1);
		g[1] = tex2D<int>(texG3[1], xPos, y - radius - 1);
		g[2] = tex2D<int>(texG3[2], xPos, y - radius - 1);
		atomicSub(&histogram4[f].x, 1);
		atomicSub(&histogram4[f].y, g[0]);
		atomicSub(&histogram4[f].z, g[1]);
		atomicSub(&histogram4[f].w, g[2]);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, g[0]);
			atomicSub(&fgSumUpToIndex.z, g[1]);
			atomicSub(&fgSumUpToIndex.w, g[2]);
		}

		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			float4 cxdx = *((float4*)((char*)CxDx + y * pitchF4) + x);
			//�����l�v�Z
			//*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);

			if (x == TX)
			{
				//*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedianDebug(cxdx, histogram4, fgSumUpToIndex, index, x, y);
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
				printf("%d ", *((int*)((char*)result_center + y * pitchI1) + x));
			}
			else
			{
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
			}

		}
		//thread����
		__syncthreads();
	}

}


__global__ void
de_filter2DDebug(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t* texGX, float** CxDx, size_t pitchI1, size_t pitchF1, int n)
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
	int *fgXSumUpToIndex = &buffers[0];
	int *histogramX = &buffers[n + 1];
	//g1,g3�ł�fgXSumUpToIndex��f,g�̏��Ŏ����������Acxdx�Ə��Ԃ����킹�邽�߂ɁAg,...,g f�̏��ɂ���
	//histogram�����l
	//histogram��1�����ɕ���ł��āA�ebin�ɂ��āAg,...,g,f �̏��ɕ���ł���

	__shared__ int index;
	int f;
	int *g = new int[n];

	const int k = n + 1;
	float *cxdx;

	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		cxdx = new float[k];
		//�q�X�g�O����������
		for (int i = 0; i <= Imax * n; i++)
		{
			histogramX[i] = 0;
		}
		for (int i = 0; i <= n; i++)
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
				atomicAdd(&histogramX[f*k + i], g[i]);
			atomicAdd(&histogramX[f*k + n], 1);
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
		for (int i = 0; i < k; i++)
			cxdx[i] = *((float*)((char*)CxDx[i]) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);
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
			atomicAdd(&histogramX[f*k + i], g[i]);
		atomicAdd(&histogramX[f*k + n], 1);
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
			atomicSub(&histogramX[f*k + i], g[i]);
		atomicSub(&histogramX[f*k + n], 1);
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
			for (int i = 0; i < k; i++)
				cxdx[i] = *((float*)((char*)CxDx[i] + y * pitchF1) + x);
			//�����l�v�Z

//			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);
			//*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedianDebug(cxdx, histogramX, fgXSumUpToIndex, index, n, x, y);


			if (x == TX)
			{
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);
				//*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedianDebug(cxdx, histogramX, fgXSumUpToIndex, index, n, x, y);
				printf("%d ", *((int*)((char*)result_center + y * pitchI1) + x));
			}
			else
			{
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);
			}
			if (*((int*)((char*)result_center + y * pitchI1) + x) == 255)
			{
				//printf("f");
			}
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

//multi test�p
__global__ void
de_filter2DTest(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t* texGX, float** CxDx2, size_t pitchI1, size_t pitchF1, int n, size_t pitchF4, float4* CxDx1)
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
	int *fgXSumUpToIndex = &buffers[0];
	int *histogramX = &buffers[n + 1];
	//g1,g3�ł�fgXSumUpToIndex��f,g�̏��Ŏ����������Acxdx�Ə��Ԃ����킹�邽�߂ɁAg,...,g f�̏��ɂ���
	//histogram�����l
	//histogram��1�����ɕ���ł��āA�ebin�ɂ��āAg,...,g,f �̏��ɕ���ł���

	__shared__ int index;
	__shared__ int4 fgSumUpToIndex;//index�ȉ��a
	int f;
	int g[3];
	int4 *histogram4 = (int4*)&buffers[(Imax + 1) * (n + 1)];




	__shared__ int index2;
	int f2;
	int *g2 = new int[n];

	const int k = n + 1;
	float *cxdx2;




	//���S�X���b�h�̂ݎ��s
	if (tid == radius)
	{
		//g3
		//�q�X�g�O����������
		for (int i = 0; i < Imax; i++)
		{
			histogram4[i] = make_int4(0, 0, 0, 0);
		}
		fgSumUpToIndex = make_int4(0, 0, 0, 0);
		index = tex2D<int>(texF, x, 0);//current index

		//multi
		cxdx2 = new float[k];
		//�q�X�g�O����������
		for (int i = 0; i < Imax * k; i++)
		{
			histogramX[i] = 0;
		}
		for (int i = 0; i < k; i++)
		{
			fgXSumUpToIndex[i] = 0;
		}
		index2 = tex2D<int>(texF, x, 0);//current index
	}
	//thread����
	__syncthreads();

	//1�ڃq�X�g�O�����`��
	for (int yy = -radius; yy <= radius; yy++)
	{
		//g3
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		{
			f = tex2D<int>(texF, xPos, yy);
			g[0] = tex2D<int>(texGX[0], xPos, yy);
			g[1] = tex2D<int>(texGX[1], xPos, yy);
			g[2] = tex2D<int>(texGX[2], xPos, yy);
			atomicAdd(&histogram4[f].x, 1);
			atomicAdd(&histogram4[f].y, g[0]);
			atomicAdd(&histogram4[f].z, g[1]);
			atomicAdd(&histogram4[f].w, g[2]);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g[0]);
				atomicAdd(&fgSumUpToIndex.z, g[1]);
				atomicAdd(&fgSumUpToIndex.w, g[2]);
			}

		}

		//multi
		//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
		{
			f2 = tex2D<int>(texF, xPos, yy);
			for (int i = 0; i < n; i++)
				g2[i] = tex2D<int>(texGX[i], xPos, yy);
			for (int i = 0; i < n; i++)
				atomicAdd(&histogramX[f2*k + i], g2[i]);
			atomicAdd(&histogramX[f2*k + n], 1);
			if (f2 <= index2)
			{
				for (int i = 0; i < n; i++)
					atomicAdd(&fgXSumUpToIndex[i], g2[i]);
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
		//g3
		float4 cxdx = *((float4*)((char*)CxDx1) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);

		//multi
		for (int i = 0; i < k; i++)
			cxdx2[i] = *((float*)((char*)CxDx2[i]) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx2, histogramX, fgXSumUpToIndex, index2, n);
	}
	//thread����
	__syncthreads();

	//2�s�ڈȍ~�̏���
	for (int y = 1; y < height; y++)
	{
		//g3

		//�q�X�g�O�����ɒǉ�
		f = tex2D<int>(texF, xPos, y + radius);
		g[0] = tex2D<int>(texGX[0], xPos, y + radius);
		g[1] = tex2D<int>(texGX[1], xPos, y + radius);
		g[2] = tex2D<int>(texGX[2], xPos, y + radius);
		atomicAdd(&histogram4[f].x, 1);
		atomicAdd(&histogram4[f].y, g[0]);
		atomicAdd(&histogram4[f].z, g[1]);
		atomicAdd(&histogram4[f].w, g[2]);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, g[0]);
			atomicAdd(&fgSumUpToIndex.z, g[1]);
			atomicAdd(&fgSumUpToIndex.w, g[2]);
		}
		//�q�X�g�O��������폜
		f = tex2D<int>(texF, xPos, y - radius - 1);
		g[0] = tex2D<int>(texGX[0], xPos, y - radius - 1);
		g[1] = tex2D<int>(texGX[1], xPos, y - radius - 1);
		g[2] = tex2D<int>(texGX[2], xPos, y - radius - 1);
		atomicSub(&histogram4[f].x, 1);
		atomicSub(&histogram4[f].y, g[0]);
		atomicSub(&histogram4[f].z, g[1]);
		atomicSub(&histogram4[f].w, g[2]);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, g[0]);
			atomicSub(&fgSumUpToIndex.z, g[1]);
			atomicSub(&fgSumUpToIndex.w, g[2]);
		}

		//multi
		//�q�X�g�O�����ɒǉ�
		f2 = tex2D<int>(texF, xPos, y + radius);
		for (int i = 0; i < n; i++)
			g2[i] = tex2D<int>(texGX[i], xPos, y + radius);
		for (int i = 0; i < n; i++)
			atomicAdd(&histogramX[f2*k + i], g2[i]);
		atomicAdd(&histogramX[f2*k + n], 1);
		if (f2 <= index2)
		{
			for (int i = 0; i < n; i++)
				atomicAdd(&fgXSumUpToIndex[i], g2[i]);
			atomicAdd(&fgXSumUpToIndex[n], 1);
		}
		//�q�X�g�O��������폜
		f2 = tex2D<int>(texF, xPos, y - radius - 1);
		for (int i = 0; i < n; i++)
			g2[i] = tex2D<int>(texGX[i], xPos, y - radius - 1);
		for (int i = 0; i < n; i++)
			atomicSub(&histogramX[f2*k + i], g2[i]);
		atomicSub(&histogramX[f2*k + n], 1);
		if (f2 <= index2)
		{
			for (int i = 0; i < n; i++)
				atomicSub(&fgXSumUpToIndex[i], g2[i]);
			atomicSub(&fgXSumUpToIndex[n], 1);
		}

		//�q�X�g�O�����Ȃǂ̃`�F�b�N
		if (f != f2)
			printf("f != f2\n");
		if (g[0] != g2[0])
			printf("g[0] != g2[0]\n");
		if (g[1] != g2[1])
			printf("g[1] != g2[1]\n");
		if (g[2] != g2[2])
			printf("g[2] != g2[2]\n");

		if (histogram4[f].x != histogramX[f2*k + n] || histogram4[f].y != histogramX[f2*k + 0] || histogram4[f].z != histogramX[f2*k + 1] | histogram4[f].w != histogramX[f2*k + 2]) {
			//printf("(%d) %d %d %d %d HHHHHH (%d) %d %d %d %d\n", f, histogram4[f].x, histogram4[f].y, histogram4[f].z, histogram4[f].w, f2, histogramX[f2*k + n], histogramX[f2*k + 0], histogramX[f2*k + 1], histogramX[f2*k + 2]);
			//�q�X�g�O����
			printf("(%d, %d) g3 | multi\n", x, y);
			for (int i = 0; i < 256; i++)
			{
				printf("(%d) %d %d %d %d | %d %d %d %d\n", i, histogram4[i].x, histogram4[i].y, histogram4[i].z, histogram4[i].w, histogramX[i*k + n], histogramX[i*k + 0], histogramX[i*k + 1], histogramX[i*k + 2]);

			}

		}


		//thread����
		__syncthreads();
		//���S�X���b�h�̂ݎ��s
		if (tid == radius)
		{
			//g3
			float4 cxdx = *((float4*)((char*)CxDx1 + y * pitchF4) + x);
			int saveIndex = index;
			int4 saveSumupto = { fgSumUpToIndex.x,fgSumUpToIndex.y,fgSumUpToIndex.z,fgSumUpToIndex.w };
			//�����l�v�Z
			int result1 = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
			*((int*)((char*)result_center + y * pitchI1) + x) = result1;

			//multi
			for (int i = 0; i < k; i++)
				cxdx2[i] = *((float*)((char*)CxDx2[i] + y * pitchF1) + x);
			int saveIndex2 = index2;
			int4 saveSumupto2 = { fgXSumUpToIndex[0],fgXSumUpToIndex[1],fgXSumUpToIndex[2],fgXSumUpToIndex[3] };
			//�����l�v�Z
			int result2 = de_findMedian(cxdx2, histogramX, fgXSumUpToIndex, index2, n);
			*((int*)((char*)result_center + y * pitchI1) + x) = result2;

			/*
			if (cxdx.x != cxdx2[0] || cxdx.y != cxdx2[1] || cxdx.z != cxdx2[2] || cxdx.w != cxdx2[3])
			{
				printf("(%d, %d) ", x, y);
			}
			*/

			/*
			if (result1 != result2)
			{
				if (saveIndex == saveIndex2)
				{
					printf("############\n");
					//g3
					index = saveIndex;
					fgSumUpToIndex.x = saveSumupto.x;
					fgSumUpToIndex.y = saveSumupto.y;
					fgSumUpToIndex.z = saveSumupto.z;
					fgSumUpToIndex.w = saveSumupto.w;
					int result11, result21;
					//multi
					index2 = saveIndex2;
					fgXSumUpToIndex[0] = saveSumupto2.x;
					fgXSumUpToIndex[1] = saveSumupto2.y;
					fgXSumUpToIndex[2] = saveSumupto2.z;
					fgXSumUpToIndex[3] = saveSumupto2.w;


					//����
					de_findMedianDebug(cxdx2, histogramX, fgXSumUpToIndex, index2, n, x, y, cxdx, histogram4, fgSumUpToIndex, index, result11, result21);



				}
				//printf("%f, %f, %f, %f\n", cxdx.x, cxdx.y, cxdx.z, cxdx.w);
				//printf("%f, %f, %f, %f\n", cxdx2[0], cxdx2[1], cxdx2[2], cxdx2[3]);

				//int result3 = 0;// de_findMedianDebug(cxdx2, histogramX, fgXSumUpToIndex, index2, n, x, y);
				//printf("(%d,%d) = %d,%d | %d,%d | %d\n", x, y, result1, result2, saveIndex, saveIndex2, result3);

			}
			*/
		}
		//thread����
		__syncthreads();
	}


	delete g2;
	if (tid == radius)
	{
		delete cxdx2;
	}
	__syncthreads();
}


//multi test�p
void cu_filter2DTest(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, float4* cxdx, DeviceArray<float>* cxdx2)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

	//�c������ shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	int n = g->arrayLength;
	int m = f->arrayLength;
	for (int i = 0; i < m; i++)
	{
		std::cout << "<" << i << ">" << std::endl;
		de_filter2DTest << <gridSizeX, blockSize, (Imax + 1) * sizeof(int) * (n + 1) + Imax * sizeof(int) * 4, NULL >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF.host[i], texG.device, cxdx2->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>(), n, sizeInfo.pitch<float4>(), cxdx);
		//		Utility::showDevice(f->host[i], sizeInfo, "in", false, 255);
		/*
		if (i == 2)
		{
			de_filter2DDebug << <gridSizeX, blockSize, (Imax + 1) * sizeof(int) * (n + 1), NULL >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF.host[i], texG.device, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>(), n);

		}
		else
		{
			de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int) * (n + 1), NULL >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF.host[i], texG.device, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>(), n);
		}
		*/
		//cudaDeviceSynchronize();
//		Utility::showDevice(result_center->host[i], sizeInfo, "res", false, 255, true);
	}
	cudaDeviceSynchronize();
}



#endif