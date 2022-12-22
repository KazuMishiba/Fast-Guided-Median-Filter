#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "boxFilter.cuh"
#include "Utility.h"

#include "common.h"


//テンプレートがこの形式だと使えないのでこのファイルは読み込まない

//gX
template<int N>
__global__ void
de_gXsum_x(int width, int height, int radius, int** sumG, int** sumGG, cudaTextureObject_t* texGX, size_t pitchI1)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	int g[N];
	int _g[N];
	int sumg[N] = { 0 };
	int sumgg[(N + 1)*N / 2] = { 0 };

	//x=0
	for (int x = -radius; x <= radius; x++)
	{
		for (int i = 0; i < N; i++)
			g[i] = tex2D<int>(texGX[i], x, y);

		for (int i = 0; i < N; i++)
			sumg[i] += g[i];

		int n = 0;
		for (int i = 0; i < N; i++) {
			for (int j = i; j < N; j++) {
				sumgg[n] += g[i] * g[j];
				n++;
			}
		}
	}
	for (int i = 0; i < N; i++)
		*((int*)((char*)(sumG[i]) + y * pitchI1)) = sumg[i];
	for (int i = 0; i < (N + 1)*N / 2; i++)
		*((int*)((char*)sumGG[i] + y * pitchI1)) = sumgg[i];

	for (int x = 1; x < width; x++)
	{
		for (int i = 0; i < N; i++)
			g[i] = tex2D<int>(texGX[i], x + radius, y);
		for (int i = 0; i < N; i++)
			_g[i] = tex2D<int>(texGX[i], x - radius - 1, y);
		for (int i = 0; i < N; i++)
			sumg[i] += g[i] - _g[i];
		int n = 0;
		for (int i = 0; i < N; i++) {
			for (int j = i; j < N; j++) {
				sumgg[n] += g[i] * g[j] - _g[i] * _g[j];
				n++;
			}
		}
		for (int i = 0; i < N; i++)
			*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg[i];
		for (int i = 0; i < (N + 1)*N / 2; i++)
			*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg[i];
	}

}


template<int N>
__global__ void
de_gXsum_y(int width, int height, int radius, int** sumG, int** sumGG, cudaTextureObject_t* texSumG, cudaTextureObject_t* texSumGG, size_t pitchI1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	int sumg[N] = { 0 };
	int sumgg[(N + 1)*N / 2] = { 0 };

	//y = 0
	for (int y = -radius; y <= radius; y++)
	{
		for (int i = 0; i < N; i++)
			sumg[i] += tex2D<int>(texSumG[i], x, y);
		for (int i = 0; i < (N + 1)*N / 2; i++)
			sumgg[i] += tex2D<int>(texSumGG[i], x, y);

	}
	for (int i = 0; i < N; i++)
		*((int*)((char*)sumG[i]) + x) = sumg[i];
	for (int i = 0; i < (N + 1)*N / 2; i++)
		*((int*)((char*)sumGG[i]) + x) = sumgg[i];

	for (int y = 1; y < height; y++)
	{
		for (int i = 0; i < N; i++)
			sumg[i] += tex2D<int>(texSumG[i], x, y + radius) - tex2D<int>(texSumG[i], x, y - radius - 1);
		for (int i = 0; i < (N + 1)*N / 2; i++)
			sumgg[i] += tex2D<int>(texSumGG[i], x, y + radius) - tex2D<int>(texSumGG[i], x, y - radius - 1);

		for (int i = 0; i < N; i++)
			*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg[i];
		for (int i = 0; i < (N + 1)*N / 2; i++)
			*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg[i];
	}
}



//cxX
template<int N>
__global__ void
de_calculateCxXDx(int width, int height, int** GX, int** sumG, int** sumGG, float eps2, float pixel_sum_window_inv, float** cxdx, size_t pitchI1, size_t pitchF1)
{
	//cxdxの要素数はN+1 (cxがN、dxが1)
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float g_ave[N];
	for (int i = 0; i < N; i++)
		g_ave[i] = *((int*)((char*)sumG[i] + y * pitchI1) + x) * pixel_sum_window_inv;
	//const float g_ave1 = *((int*)((char*)sumG[0] + y * pitchI1) + x) * pixel_sum_window_inv;
	//const float g_ave2 = *((int*)((char*)sumG[1] + y * pitchI1) + x) * pixel_sum_window_inv;
	//const float g_ave3 = *((int*)((char*)sumG[2] + y * pitchI1) + x) * pixel_sum_window_inv;
	float A[(N + 1)*N / 2];
	int n = 0;
	for (int j = 0; j < N; j++) {
		A[n] = *((int*)((char*)sumGG[n] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave[j] * g_ave[j] + eps2;
		n++;
		for (int i = j + 1; i < N; i++) {
			A[n] = *((int*)((char*)sumGG[n] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave[i] * g_ave[j];
			n++;
		}
	}
	/*
	const float v11 = *((int*)((char*)sumGG[0] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave1 + eps2;
	const float v12 = *((int*)((char*)sumGG[1] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave2;
	const float v13 = *((int*)((char*)sumGG[2] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave3;
	const float v22 = *((int*)((char*)sumGG[3] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave2 * g_ave2 + eps2;
	const float v23 = *((int*)((char*)sumGG[4] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave2 * g_ave3;
	const float v33 = *((int*)((char*)sumGG[5] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave3 * g_ave3 + eps2;
	*/
	//共役勾配法
	//初期値0
	//メモリ確保
	float r[N];
	float p[N];
	float Ap[N];
	float cx[N];
	float rsold = 0.0f;
	float alpha, rsnew;

	for (int i = 0; i < N; i++)
	{
		r[i] = g_ave[i];
		p[i] = r[i];
		rsold += r[i] * r[i];
	}

	for (int iter = 0; iter < N; iter++)
	{
		alpha = 0.0f;
		//Ap = A * p
		for (int j = 0; j < N; j++)
		{
			int m = 0;//直す
			Ap[j] = A[m] * p[j];
			for (int i = 1; i < N; i++)
			{
				m = 0;//直す
				Ap[j] += A[m] * p[j];
			}
			//alpha = rsold / (p' * Ap);
			alpha += p[j] * Ap[j];
		}
		alpha = rsold / alpha;

		rsnew = 0.0f;
		for (int i = 0; i < N; i++)
		{
			//x = x + alpha * p;
			cx[i] += alpha * p[i];
			//r = r - alpha * Ap;
			r[i] -= alpha * Ap[i];
			//rsnew = r' * r;
			rsnew += r[i] * r[i];
		}
		if (rsnew < 0.0000000001)
		{
			break;
		}
		//p = r + (rsnew / rsold) * p;
		float no = rsnew / rsold;
		for (int i = 0; i < N; i++)
		{
			p[i] = r[i] + no * p[i];
		}
		//rsold = rsnew;
		rsold = rsnew;

	}

	float dx = pixel_sum_window_inv;
	for (int i = 0; i < N; i++)
	{
		cx[i] *= pixel_sum_window_inv;
		dx -= cx[i] * g_ave[i];
		*((float*)((char*)cxdx[i] + y * pitchF1) + x) = cx[i];
	}
	*((float*)((char*)cxdx[N] + y * pitchF1) + x) = dx;
}



//gX
template<int N>
void cu_calculateSumGX(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* GX, int radius, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG)
{
	int blockSize = BLOCK_SIZE_1D;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);

	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int>* texG = new TextureArray<int>(GX, filterMode, sizeInfo);
	//texGの内容からgsumをtempGに、ggsumをtempGGに格納
	de_gXsum_x<N> << <gridSizeY, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, tempG->device, tempGG->device, texG->device, sizeInfo.pitch<int>());
	//tempGをtexSumGに、tempGGをtexSumGGにバインド
	TextureArray<int>* texSumG = new TextureArray<int>(tempG, filterMode, sizeInfo);
	TextureArray<int>* texSumGG = new TextureArray<int>(tempGG, filterMode, sizeInfo);
	//sumG、sumGG計算
	de_gXsum_y<N> << < gridSizeX, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, sumG->device, sumGG->device, texSumG->device, texSumGG->device, sizeInfo.pitch<int>());

	delete texG;
	delete texSumG;
	delete texSumGG;
}



//gX
template<int N>
void cu_calculateCxXDxFromG(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* GX, int radius, int pixelNumInWindow, float eps2, DeviceArray<float>* cxdx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG)
{
	cu_calculateSumGX<N>(sizeInfo, stream, GX, radius, sumG, sumGG, tempG, tempGG);
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	//sumGの値からcx, dxを計算
	//de_calculateCxXDx<N> << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, GX->device, sumG->device, sumGG->device, eps2, pixel_sum_window_inv, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
}