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

//GX nはGのチャンネル数
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
		//if(histogram[index].x)//ここはコメント外してもいいかも
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





//縦方向スキャン版 sharedメモリ使用版
/*
１ブロックあたり１列処理する。
１ブロックに対して、ウィンドウ直径分のスレッドを起動する。
ヒストグラムはsharedメモリに持ち、ブロック内で共有する。
各スレッドの役割は、テクスチャメモリから対応する位置のf,gをサンプリングして、アトミック演算でヒストグラムに追加、削除することである。
ブロック中メインのスレッドが一つあり、そのスレッドは自分自身を含む各スレッドのヒストグラム構築を待って中央値を計算する
*/
__global__ void
de_filter2D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//処理対象の中心座標は blockIdx.x になる。
	/*
	* threadは0～radius*2 まで使用するとして、それ以上はreturnする
	* 中心threadのidxはradius
	*/

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + tid - radius;

	__shared__ int index;
	__shared__ int2 fgSumUpToIndex;//index以下和

	int f, g;

	//中心スレッドのみ実行
	if (tid == radius)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram2[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
		index = tex2D<int>(texF, x, 0);//current index
	}
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		float2 cxdx = *((float2*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram2, fgSumUpToIndex, index);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//ヒストグラムに追加
		f = tex2D<int>(texF, xPos, y + radius);
		g = tex2D<int>(texG, xPos, y + radius);
		atomicAdd(&histogram2[f].x, 1);
		atomicAdd(&histogram2[f].y, g);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, g);
		}
		//ヒストグラムから削除
		f = tex2D<int>(texF, xPos, y - radius - 1);
		g = tex2D<int>(texG, xPos, y - radius - 1);
		atomicSub(&histogram2[f].x, 1);
		atomicSub(&histogram2[f].y, g);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, g);
		}

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			float2 cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//中央値計算
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram2, fgSumUpToIndex, index);
		}
		//thread同期
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
	__shared__ int4 fgSumUpToIndex;//index以下和
	int f;
	int g[3];

	//中心スレッドのみ実行
	if (tid == radius)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram4[i] = make_int4(0, 0, 0, 0);
		}
		fgSumUpToIndex = make_int4(0, 0, 0, 0);
		index = tex2D<int>(texF, x, 0);//current index
	}
	//thread同期
	__syncthreads();
	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		float4 cxdx = *((float4*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//ヒストグラムに追加
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
		//ヒストグラムから削除
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

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			float4 cxdx = *((float4*)((char*)CxDx + y * pitchF4) + x);
			//中央値計算
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
		}
		//thread同期
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

	//shared memoryをhistogramXとfgXSumUpToIndexに分ける
	extern __shared__ int  buffers[];
	int *fgXSumUpToIndex = &buffers[0];
	int *histogramX = &buffers[n+1];
	//g1,g3ではfgXSumUpToIndexはf,gの順で実装したが、cxdxと順番を合わせるために、g,...,g fの順にする
	//histogramも同様
	//histogramは1次元に並んでいて、各binについて、g,...,g,f の順に並んでいる

	__shared__ int index;
	int f;
	int *g = new int[n];

	const int k = n + 1;
	float *cxdx;

	//中心スレッドのみ実行
	if (tid == radius)
	{
		cxdx = new float[k];
		//ヒストグラム初期化
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
	//thread同期
	__syncthreads();
	
	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();


	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		for (int i = 0; i < k; i++)
			cxdx[i] = *((float*)((char*)CxDx[i]) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);		
	}
	//thread同期
	__syncthreads();
	
	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//ヒストグラムに追加
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
		//ヒストグラムから削除
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

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			for (int i = 0; i < k; i++)
				cxdx[i] = *((float*)((char*)CxDx[i] + y * pitchF1) + x);
			//中央値計算
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);
		}
		//thread同期
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
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド
	UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texGにgをバインド

	//縦方向版 shared
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
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド

	//縦方向版 shared
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
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド

	//縦方向版 shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	int n = g->arrayLength;
	//shared memoryは、ヒストグラム＋uptoindex必要で、ヒストグラムはImax * (n+1)、uptoindexはn+1必要
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

	//縦方向版 shared
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

	//縦方向版 shared
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
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド

	//縦方向版 shared
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

	//縦方向版 shared
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
//1行1スレッド試験用　I1G1のみ
/*
//キャッシュ使用
void cu_filter2D_Cache(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texF, texG;
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド
	UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texGにgをバインド

	//縦方向版 shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter2D_Cache << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}

//shared memory使用
void cu_filter2D_Shared(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texF, texG;
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド
	UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texGにgをバインド

	//縦方向版 shared
	int blockSize = radius * 2 + 1;
	int gridSizeX = sizeInfo.width;
	de_filter2D << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}
*/


//////////////////////////////////////////////////
//3D

//これは任意の次元でも使えそう（ポインタが適切なら）
//g1
__global__ void
de_filter3D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t* texF, cudaTextureObject_t* texG, int numOfFrames, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//処理対象の中心座標は blockIdx.x になる。
	/*
	* threadは0～radius*2 まで使用するとして、それ以上はreturnする
	* 中心threadのidxはradius
	*/

	int x = blockIdx.x;
	if (x >= width)
		return;
	int tid = threadIdx.x;
	if (tid >= radius * 2 + 1)
		return;
	int xPos = x + tid - radius;

	__shared__ int index;
	__shared__ int2 fgSumUpToIndex;//index以下和

	int f, g;

	//中心スレッドのみ実行
	if (tid == radius)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram2[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
		index = tex2D<int>(texF[numOfFrames / 2], x, 0);//current index 本当は処理中心フレームindexを指定したい(これだと端のときに、処理中心ではなくなる)
	}
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int k = 0; k < numOfFrames; k++)
	{
		for (int yy = -radius; yy <= radius; yy++)
		{
			//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		float2 cxdx = *((float2*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram2, fgSumUpToIndex, index);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		for (int k = 0; k < numOfFrames; k++)
		{
			//ヒストグラムに追加
			f = tex2D<int>(texF[k], xPos, y + radius);
			g = tex2D<int>(texG[k], xPos, y + radius);
			atomicAdd(&histogram2[f].x, 1);
			atomicAdd(&histogram2[f].y, g);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex.x, 1);
				atomicAdd(&fgSumUpToIndex.y, g);
			}
			//ヒストグラムから削除
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

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			float2 cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//中央値計算
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram2, fgSumUpToIndex, index);
		}
		//thread同期
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
	__shared__ int4 fgSumUpToIndex;//index以下和

	int f;
	int g[3];

	//中心スレッドのみ実行
	if (tid == radius)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram4[i] = make_int4(0, 0, 0, 0);
		}
		fgSumUpToIndex = make_int4(0, 0, 0, 0);
		index = tex2D<int>(texF[numOfFrames / 2], x, 0);//current index 本当は処理中心フレームindexを指定したい(これだと端のときに、処理中心ではなくなる)
	}
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int k = 0, n = 0; k < numOfFrames; k++, n+=3)
	{
		for (int yy = -radius; yy <= radius; yy++)
		{
			//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		float4 cxdx = *((float4*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		for (int k = 0, n = 0; k < numOfFrames; k++, n+=3)
		{
			//ヒストグラムに追加
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
			//ヒストグラムから削除
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

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			float4 cxdx = *((float4*)((char*)CxDx + y * pitchF4) + x);
			//中央値計算
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
		}
		//thread同期
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

	//fについて各チャンネルのdevice memoryを配列に格納
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

	//fについて各チャンネルのdevice memoryを配列に格納
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
スレッドは最小の処理単位であり、例えば各行について行方向に連続的に処理をする場合、スレッド数は画像高さ分になる。
例えば画像高さが512だったとしよう。
このとき、ヒストグラム解像度が256だとすると、もしスレッドがすべて起動できたとすると、ヒストグラムの記録に512*256*2(f,g)*4byte(int型)のメモリが必要となる。

総スレッド数 = グリッド数 × ブロック数

なので、基本的には適当にグリッド数かブロック数を決めればよい。

スレッドはいくつ同時に起動するのか。


連続するスレッドが連続するメモリにアクセスすると効率が良い（CPUとは異なる発想）
つまり、各スレッドが行方向にアクセスするのは効率が悪いということ。
ということは、列または行単位の処理をする場合は、どちらでもいいなら列方向にすべきであることが分かる。
これにより隣あう列のスレッド（＝連続するスレッド）が連続するメモリにアクセスすることになる。
（コアレスアクセス）

ブロックサイズの決め方
・Occupancyなるべく１００％にする
・ブロックあたりのスレッド数はなるべく小さく
・横方向はコアレスアクセス。なるべく長くする。

ブロックサイズは128threadの倍数であると良い(少なくとも３２の倍数)。


ブロックはSMX(streaming multiprocessor extreme)上で実行される。
SMの中には1つ以上のブロックがあり、処理される。
K40を例にすると、SMは15個あり、１個当たり６４K個の32bitレジスタを持っている、つまり合計で3840KByteの容量のレジスタを持っている。
１つのスレッドが使えるレジスタ数はKelperなら６３個。

このことから、ヒストグラムをレジスタに持たせるのは無理であり、たぶんL1キャッシュとかに確保されているのだろう。
L1とsharedメモリは同じ場所？なので、わざわざsharedメモリを使うメリットはない。
と思いきや、自動変数だからと言って必ずしもL1にキャッシュされないかもしれないらしい。
sharedメモリは明示的にその場所に確保できるという点で、確実にアクセスされることが分かっているものはsharedメモリに置いたほうがいいかもしれない。



sharedメモリの使用。
１ブロックごとに48～64KBまで確保できる。
ヒストグラムは１列ごとに
256*2(f,g)*4byte(int型)=2KB
必要で、ブロックサイズが32だとその時点でオーバーする。
このサイズを見るに、ヒストグラムは確保してもL1にはキャッシュされないのか。



高速にするにはhost to deviceのメモリ転送をまとめて行う。
（といっても今回はそんなにないか？ 画像くらいか）



メジアン探す工程は、探さずに固定値代入するのと速度変わらなかった。つまりメジアン探すのは今のアルゴリズムではほとんど時間がかからない。
短縮できそうなのは、メモリアクセスやスレッド分割などくらいか。

テクスチャアクセスをfloatではなくintにした(+0.5fしないかつint型）が全く影響なし。
histogramをint2から独立した２つに分けたが全く速度変わらず。

何が支配的なんだ？
メジアン探さず、atomicAdd無くしたら2/3になった。これが限界だが、add無くすということはヒストグラム更新しないことになるので、そうはいかない。
なるべくatomic演算を減らす方法を考える。
⇒頑張ったが無理。

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


//意味なかった

//f,g分離版 占有率を高める
__global__ void
de_filter2D_shared3(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//処理対象の中心座標は blockDim.x * blockIdx.x + threadIdx.x になる。
	//
	/*
	* threadは0～radius*2 まで使用するとして、それ以上はreturnする
	* 中心threadのidxはradius
	*/

	int diameter = radius * 2 + 1;

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x >= width)
		return;
	//中心座標からの相対位置を表すtidはthreadIdx.yになる
	int tid = threadIdx.y;
	if (tid >= diameter)
		return;
	int xPos = x + tid - radius;

	__shared__ int2 histogram[8][256];//4は決め打ち(base 128 でr=15のとき)だがそのうち可変にする
	__shared__ int index[8];
	__shared__ int2 fgSumUpToIndex[8];//index以下和
	//[4]のどれを使用するのかは、threadIdx.xできまる
	int column = threadIdx.x;


	int f;
	int g;
	float2 cxdx;

	//中心スレッドのみ実行
	if (tid == radius)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram[column][i] = make_int2(0, 0);
		}
		fgSumUpToIndex[column] = make_int2(0, 0);
		index[column] = tex2D<int>(texF, float(x) + 0.5f, 0.5f);//current index
		cxdx = *((float2*)((char*)CxDx) + x);
	}
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[column], fgSumUpToIndex[column], index[column]);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//ヒストグラムに追加
		f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius) + 0.5f);
		g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius) + 0.5f);
		atomicAdd(&histogram[column][f].x, 1);
		atomicAdd(&histogram[column][f].y, g);
		if (f <= index[column])
		{
			atomicAdd(&fgSumUpToIndex[column].x, 1);
			atomicAdd(&fgSumUpToIndex[column].y, g);
		}
		//ヒストグラムから削除
		f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius - 1) + 0.5f);
		g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius - 1) + 0.5f);
		atomicSub(&histogram[column][f].x, 1);
		atomicSub(&histogram[column][f].y, g);
		if (f <= index[column])
		{
			atomicSub(&fgSumUpToIndex[column].x, 1);
			atomicSub(&fgSumUpToIndex[column].y, g);
		}

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//中央値計算
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[column], fgSumUpToIndex[column], index[column]);
		}
		//thread同期
		__syncthreads();
	}
}



//f,g分離版
void cu_filter2D_shared3(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureObject_t texF, texG;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	//texFにfをバインド
	Utility::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);
	//texGにgをバインド
	Utility::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);

	//縦方向版 shared
	/*
	占有率を高めることを考える。
	今の実装の場合、blockSizeがフィルタ直径になっている。
	blockSizeは32の倍数が好ましく、特に128の倍数が良い(理由はよくわからない)。
	よって、好ましい指定のblockSizeになるように調整する。
	*/
	int baseBlockSize = 128;
	const int diameter = radius * 2 + 1;
	//baseBlockSizeがdiameterよりも小さい場合は調整
	baseBlockSize *= ((diameter / baseBlockSize) + 1);
	//diameter以上の最小の2のべき乗の数
	int blockY = 1;
	while (true)
	{
		blockY *= 2;
		if (blockY >= diameter)
			break;
	}
	//baseBlockSizeにblockYがいくつ入るか
	int blockX = baseBlockSize / blockY;
	dim3 blockSize = dim3(blockX, blockY, 1);
	//printf("%d %d\n", blockX, blockY);
	//int gridSizeX = sizeInfo.width;// ceil(sizeInfo.width / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockX);
	de_filter2D_shared3 << <gridSizeX, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}




//過去実装


__global__ void
de_filter2D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//処理対象の中心座標は blockIdx.x になる。
	/*
	* threadは0～radius*2 まで使用するとして、それ以上はreturnする
	* 中心threadのidxはradius
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
	__shared__ int2 fgSumUpToIndex;//index以下和


	int f;
	int g;
	float2 cxdx;

	//中心スレッドのみ実行
	if (tid == radius)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
		index = tex2D<int>(texF, float(x) + 0.5f, 0.5f);//current index
		cxdx = *((float2*)((char*)CxDx) + x);
	}
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx.x, cxdx.y, histogram, fgSumUpToIndex, index);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//ヒストグラムに追加
		f = tex2D<int>(texF, float(xPos) + 0.5f, float(y + radius) + 0.5f);
		g = tex2D<int>(texG, float(xPos) + 0.5f, float(y + radius) + 0.5f);
		atomicAdd(&histogram[f].x, 1);
		atomicAdd(&histogram[f].y, g);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, g);
		}
		//ヒストグラムから削除
		f = tex2D<int>(texF, float(xPos) + 0.5f, float(y - radius - 1) + 0.5f);
		g = tex2D<int>(texG, float(xPos) + 0.5f, float(y - radius - 1) + 0.5f);
		atomicSub(&histogram[f].x, 1);
		atomicSub(&histogram[f].y, g);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, g);
		}

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//中央値計算
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram, fgSumUpToIndex, index);
		}
		//thread同期
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
				//超えたのでこのindexがmedian
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
				//超えたのでこのindexがmedian
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
				//超えたのでこのindexがmedian
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
//確認用 画素x,yのcx、dxを計算
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
				//超えたのでこのindexがmedian
				int result_center = index;
				index += flag2;
				return result_center;
			}
		}
		index += flag2;
	}
}

//ヒストグラムのサンプリングを間引く
/*
hの和が1であることが成立しなくなるのでhalf=0.5も成り立たず、255を超える箇所が出てしまうのでこのままでは使えない。
合計値も更新していこうか。
できたが、ヒストグラム更新が疎なので模様が目立つ。
*/
__global__ void
de_filter2D_histogramSampling(int width, int height, int samplingRate, float weightingFactor, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//処理対象の中心座標は blockIdx.x になる。
	/*
	* threadは0～radius*2 まで使用するとして、それ以上はreturnする
	* 中心threadのidxはradius
	*/

	//画素のサンプル範囲は±radiusだが、

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
	__shared__ int2 fgSumUpToIndex;//index以下和
	__shared__ int2 fgSumUpToAllIndex;//合計値
	__shared__ float2 cxdx;


	int f;
	int g;

	//中心スレッドのみ実行
	if (tid == centerPos)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
		fgSumUpToAllIndex = make_int2(0, 0);
		index = tex2D<int>(texF, float(x) + 0.5f, 0.5f);//current index
		cxdx = *((float2*)((char*)CxDx) + x);
	}
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy += samplingRate)//
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == centerPos)
	{
		float adjustedCx = cxdx.x * weightingFactor;
		float adjustedDx = cxdx.y * weightingFactor;
		float half = (adjustedCx * fgSumUpToAllIndex.y + adjustedDx * fgSumUpToAllIndex.x) * 0.5f;
		*((int*)((char*)result_center) + x) = de_findMedian(adjustedCx, adjustedDx, histogram, fgSumUpToIndex, index, half);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		if (y % samplingRate == 0)
		{
			//ヒストグラム更新
			//ヒストグラムに追加
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
			//ヒストグラムから削除
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
		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == centerPos)
		{
			cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			//中央値計算
			float adjustedCx = cxdx.x * weightingFactor;
			float adjustedDx = cxdx.y * weightingFactor;
			float half = (adjustedCx * fgSumUpToAllIndex.y + adjustedDx * fgSumUpToAllIndex.x) * 0.5f;
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(adjustedCx, adjustedDx, histogram, fgSumUpToIndex, index, half);
		}
		//thread同期
		__syncthreads();
	}
}


//上記問題解決のために、ヒストグラムを2つ用意して重み付きで用いる
__global__ void
de_filter2D_histogramSampling2(int width, int height, int samplingRate, float weightingFactor, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//処理対象の中心座標は blockIdx.x になる。
	/*
	* threadは0～radius*2 まで使用するとして、それ以上はreturnする
	* 中心threadのidxはradius
	*/

	//画素のサンプル範囲は±radiusだが、

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
	__shared__ int2 fgSumUpToIndex[2];//index以下和
	__shared__ int2 fgSumUpToAllIndex[2];//合計値
	__shared__ float2 cxdx;

	//座標的に大きいほうのヒストグラムインデックス（0 or 1）
	__shared__ int forwardId;

	int f;
	int g;

	//中心スレッドのみ実行
	if (tid == centerPos)
	{
		//ヒストグラム初期化
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
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy += samplingRate)//
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == centerPos)
	{
		float half = (cxdx.x * fgSumUpToAllIndex[0].y + cxdx.y * fgSumUpToAllIndex[0].x) * 0.5f;
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[0], fgSumUpToIndex[0], index, half);
		forwardId = 1;
	}
	//thread同期
	__syncthreads();

	//次のヒストグラム形成
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


	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		if (y % samplingRate == 0)
		{
			if (tid == centerPos)
			{
				//forwardを用いて中央値の計算
				//float half = (cxdx.x * fgSumUpToAllIndex[forwardId].y + cxdx.y * fgSumUpToAllIndex[forwardId].x) * 0.5f;
				//*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], fgSumUpToIndex[forwardId], index, half);

				float forwardWeight = 1.0f;
				float backwardWeight = 0.0f;

				float half = (cxdx.x * (fgSumUpToAllIndex[forwardId].y * forwardWeight + fgSumUpToAllIndex[!forwardId].y * backwardWeight) + cxdx.y * (fgSumUpToAllIndex[forwardId].x * forwardWeight + fgSumUpToAllIndex[!forwardId].x * backwardWeight)) * 0.5f;
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], histogram[!forwardId], fgSumUpToIndex[forwardId], fgSumUpToIndex[!forwardId], index, half, forwardId, forwardWeight, backwardWeight);

				//forwardの入れ替え
				forwardId = !forwardId;
			}

			//thread同期
			__syncthreads();

			//ヒストグラム更新 backwardのを2列相当分更新してforward, backwardを入れ替える
			//ヒストグラムに追加
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
			//ヒストグラムから削除
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

			//ヒストグラムに追加
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
			//ヒストグラムから削除
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
			//重み付き中央値の計算
			//中心スレッドのみ実行
			if (tid == centerPos)
			{
				cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);

				//重みの計算
				int l1 = y % samplingRate;//backwardからの距離
				//int l2 = samplingRate - l1;//forwardまでの距離
				float forwardWeight = l1 / (float)samplingRate;
				float backwardWeight = 1 - forwardWeight;

				//中央値計算
				float half = (cxdx.x * (fgSumUpToAllIndex[forwardId].y * forwardWeight + fgSumUpToAllIndex[!forwardId].y * backwardWeight) + cxdx.y * (fgSumUpToAllIndex[forwardId].x * forwardWeight + fgSumUpToAllIndex[!forwardId].x * backwardWeight)) * 0.5f;
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], histogram[!forwardId], fgSumUpToIndex[forwardId], fgSumUpToIndex[!forwardId], index, half, forwardId, forwardWeight, backwardWeight);
			}


		}


		//thread同期
		__syncthreads();
	}
}


//原因不明の線が出るのでx方向は間引かない
__global__ void
de_filter2D_histogramSampling3(int width, int height, int samplingRate, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//処理対象の中心座標は blockIdx.x になる。
	/*
	* threadは0～radius*2 まで使用するとして、それ以上はreturnする
	* 中心threadのidxはradius
	*/

	//画素のサンプル範囲は±radiusだが、

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
	__shared__ int2 fgSumUpToIndex[2];//index以下和
	__shared__ int2 fgSumUpToAllIndex[2];//合計値
	__shared__ float2 cxdx;

	//座標的に大きいほうのヒストグラムインデックス（0 or 1）
	__shared__ int forwardId;

	int f;
	int g;

	//中心スレッドのみ実行
	if (tid == centerPos)
	{
		//ヒストグラム初期化
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
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy += samplingRate)//
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == centerPos)
	{
		float half = (cxdx.x * fgSumUpToAllIndex[0].y + cxdx.y * fgSumUpToAllIndex[0].x) * 0.5f;
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[0], fgSumUpToIndex[0], index, half);
		forwardId = 1;
	}
	//thread同期
	__syncthreads();

	//次のヒストグラム形成
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


	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		if (y % samplingRate == 0)
		{
			if (tid == centerPos)
			{
				//forwardを用いて中央値の計算
				//float half = (cxdx.x * fgSumUpToAllIndex[forwardId].y + cxdx.y * fgSumUpToAllIndex[forwardId].x) * 0.5f;
				//*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], fgSumUpToIndex[forwardId], index, half);

				float forwardWeight = 1.0f;
				float backwardWeight = 0.0f;

				float half = (cxdx.x * (fgSumUpToAllIndex[forwardId].y * forwardWeight + fgSumUpToAllIndex[!forwardId].y * backwardWeight) + cxdx.y * (fgSumUpToAllIndex[forwardId].x * forwardWeight + fgSumUpToAllIndex[!forwardId].x * backwardWeight)) * 0.5f;
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], histogram[!forwardId], fgSumUpToIndex[forwardId], fgSumUpToIndex[!forwardId], index, half, forwardId, forwardWeight, backwardWeight);

				//forwardの入れ替え
				forwardId = !forwardId;
			}

			//thread同期
			__syncthreads();

			//ヒストグラム更新 backwardのを2列相当分更新してforward, backwardを入れ替える
			//ヒストグラムに追加
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
			//ヒストグラムから削除
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

			//ヒストグラムに追加
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
			//ヒストグラムから削除
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
			//重み付き中央値の計算
			//中心スレッドのみ実行
			if (tid == centerPos)
			{
				cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);

				//重みの計算
				int l1 = y % samplingRate;//backwardからの距離
				int l2 = samplingRate - l1;//forwardまでの距離
				float forwardWeight = l1 / (float)samplingRate;
				float backwardWeight = 1 - forwardWeight;

				//色重みしてみる？
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


				//中央値計算
				float half = (cxdx.x * (fgSumUpToAllIndex[forwardId].y * forwardWeight + fgSumUpToAllIndex[!forwardId].y * backwardWeight) + cxdx.y * (fgSumUpToAllIndex[forwardId].x * forwardWeight + fgSumUpToAllIndex[!forwardId].x * backwardWeight)) * 0.5f;
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx.x, cxdx.y, histogram[forwardId], histogram[!forwardId], fgSumUpToIndex[forwardId], fgSumUpToIndex[!forwardId], index, half, forwardId, forwardWeight, backwardWeight);
			}


		}


		//thread同期
		__syncthreads();
	}
}



//ヒストグラムサンプリング
void cu_filter2D_histogramSampling(SizeInfo& sizeInfo, cudaStream_t stream, int samplingRate, int radius, int Imax, int* result_center, int* f, int* g, float2* cxdx)
{
	cudaTextureObject_t texF, texG;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	Utility::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド
	Utility::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texGにgをバインド

	//半径をsamplingRateの倍数になるように変更する
	int adjustedRadius = radius - radius % samplingRate;
	int originalDiameter = (radius * 2 + 1);
	int sampledLength = (adjustedRadius / samplingRate) * 2 + 1;
	//サンプリング考慮
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

//転送時間を除けばfのみでも速度は1%しか速くならなかった
//fのみ（セルフガイド）
__global__ void
de_filter2D_selfGuide(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, float2* CxDx, size_t pitchI1, size_t pitchF2)
{
	//処理対象の中心座標は blockIdx.x になる。
	/*
	* threadは0～radius*2 まで使用するとして、それ以上はreturnする
	* 中心threadのidxはradius
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
	__shared__ int2 fgSumUpToIndex;//index以下和

	int f;

	//中心スレッドのみ実行
	if (tid == radius)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram[i] = make_int2(0, 0);
		}
		fgSumUpToIndex = make_int2(0, 0);
		index = tex2D<int>(texF, x, 0);//current index
	}
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();
	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		float2 cxdx = *((float2*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram, fgSumUpToIndex, index);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//ヒストグラムに追加
		f = tex2D<int>(texF, xPos, y + radius);
		atomicAdd(&histogram[f].x, 1);
		atomicAdd(&histogram[f].y, f);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex.x, 1);
			atomicAdd(&fgSumUpToIndex.y, f);
		}
		//ヒストグラムから削除
		f = tex2D<int>(texF, xPos, y - radius - 1);
		atomicSub(&histogram[f].x, 1);
		atomicSub(&histogram[f].y, f);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex.x, 1);
			atomicSub(&fgSumUpToIndex.y, f);
		}

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			float2 cxdx = *((float2*)((char*)CxDx + y * pitchF2) + x);
			*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogram, fgSumUpToIndex, index);
		}
		//thread同期
		__syncthreads();
	}
}



//fのみ(セルフガイド）
void cu_filter2D_selfGuide(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, float2* cxdx)
{
	cudaTextureObject_t texF;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	//texFにfをバインド
	Utility::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);

	//縦方向版 shared
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

//GX nはGのチャンネル数
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
		//if(histogram[index].x)//ここはコメント外してもいいかも
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

//両方同時実行
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
			//if(histogram[index].x)//ここはコメント外してもいいかも
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
	__shared__ int4 fgSumUpToIndex;//index以下和
	int f;
	int g[3];

	//中心スレッドのみ実行
	if (tid == radius)
	{
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram4[i] = make_int4(0, 0, 0, 0);
		}
		fgSumUpToIndex = make_int4(0, 0, 0, 0);
		index = tex2D<int>(texF, x, 0);//current index
	}
	//thread同期
	__syncthreads();
	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();

	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		float4 cxdx = *((float4*)((char*)CxDx) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//ヒストグラムに追加
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
		//ヒストグラムから削除
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

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			float4 cxdx = *((float4*)((char*)CxDx + y * pitchF4) + x);
			//中央値計算
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
		//thread同期
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

	//shared memoryをhistogramXとfgXSumUpToIndexに分ける
	extern __shared__ int  buffers[];
	int *fgXSumUpToIndex = &buffers[0];
	int *histogramX = &buffers[n + 1];
	//g1,g3ではfgXSumUpToIndexはf,gの順で実装したが、cxdxと順番を合わせるために、g,...,g fの順にする
	//histogramも同様
	//histogramは1次元に並んでいて、各binについて、g,...,g,f の順に並んでいる

	__shared__ int index;
	int f;
	int *g = new int[n];

	const int k = n + 1;
	float *cxdx;

	//中心スレッドのみ実行
	if (tid == radius)
	{
		cxdx = new float[k];
		//ヒストグラム初期化
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
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();


	//1行目の中央値計算
	//中心スレッドのみ実行
	if (tid == radius)
	{
		for (int i = 0; i < k; i++)
			cxdx[i] = *((float*)((char*)CxDx[i]) + x);
		*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index, n);
	}
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//ヒストグラムに追加
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
		//ヒストグラムから削除
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

		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			for (int i = 0; i < k; i++)
				cxdx[i] = *((float*)((char*)CxDx[i] + y * pitchF1) + x);
			//中央値計算

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
		//thread同期
		__syncthreads();
	}


	delete g;
	if (tid == radius)
	{
		delete cxdx;
	}
	__syncthreads();
}

//multi test用
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

	//shared memoryをhistogramXとfgXSumUpToIndexに分ける
	extern __shared__ int  buffers[];
	int *fgXSumUpToIndex = &buffers[0];
	int *histogramX = &buffers[n + 1];
	//g1,g3ではfgXSumUpToIndexはf,gの順で実装したが、cxdxと順番を合わせるために、g,...,g fの順にする
	//histogramも同様
	//histogramは1次元に並んでいて、各binについて、g,...,g,f の順に並んでいる

	__shared__ int index;
	__shared__ int4 fgSumUpToIndex;//index以下和
	int f;
	int g[3];
	int4 *histogram4 = (int4*)&buffers[(Imax + 1) * (n + 1)];




	__shared__ int index2;
	int f2;
	int *g2 = new int[n];

	const int k = n + 1;
	float *cxdx2;




	//中心スレッドのみ実行
	if (tid == radius)
	{
		//g3
		//ヒストグラム初期化
		for (int i = 0; i < Imax; i++)
		{
			histogram4[i] = make_int4(0, 0, 0, 0);
		}
		fgSumUpToIndex = make_int4(0, 0, 0, 0);
		index = tex2D<int>(texF, x, 0);//current index

		//multi
		cxdx2 = new float[k];
		//ヒストグラム初期化
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
	//thread同期
	__syncthreads();

	//1つ目ヒストグラム形成
	for (int yy = -radius; yy <= radius; yy++)
	{
		//g3
		//x方向のヒストグラム形成は各スレッドが担当する
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
		//x方向のヒストグラム形成は各スレッドが担当する
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
	//thread同期
	__syncthreads();


	//1行目の中央値計算
	//中心スレッドのみ実行
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
	//thread同期
	__syncthreads();

	//2行目以降の処理
	for (int y = 1; y < height; y++)
	{
		//g3

		//ヒストグラムに追加
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
		//ヒストグラムから削除
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
		//ヒストグラムに追加
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
		//ヒストグラムから削除
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

		//ヒストグラムなどのチェック
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
			//ヒストグラム
			printf("(%d, %d) g3 | multi\n", x, y);
			for (int i = 0; i < 256; i++)
			{
				printf("(%d) %d %d %d %d | %d %d %d %d\n", i, histogram4[i].x, histogram4[i].y, histogram4[i].z, histogram4[i].w, histogramX[i*k + n], histogramX[i*k + 0], histogramX[i*k + 1], histogramX[i*k + 2]);

			}

		}


		//thread同期
		__syncthreads();
		//中心スレッドのみ実行
		if (tid == radius)
		{
			//g3
			float4 cxdx = *((float4*)((char*)CxDx1 + y * pitchF4) + x);
			int saveIndex = index;
			int4 saveSumupto = { fgSumUpToIndex.x,fgSumUpToIndex.y,fgSumUpToIndex.z,fgSumUpToIndex.w };
			//中央値計算
			int result1 = de_findMedian(cxdx, histogram4, fgSumUpToIndex, index);
			*((int*)((char*)result_center + y * pitchI1) + x) = result1;

			//multi
			for (int i = 0; i < k; i++)
				cxdx2[i] = *((float*)((char*)CxDx2[i] + y * pitchF1) + x);
			int saveIndex2 = index2;
			int4 saveSumupto2 = { fgXSumUpToIndex[0],fgXSumUpToIndex[1],fgXSumUpToIndex[2],fgXSumUpToIndex[3] };
			//中央値計算
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


					//同時
					de_findMedianDebug(cxdx2, histogramX, fgXSumUpToIndex, index2, n, x, y, cxdx, histogram4, fgSumUpToIndex, index, result11, result21);



				}
				//printf("%f, %f, %f, %f\n", cxdx.x, cxdx.y, cxdx.z, cxdx.w);
				//printf("%f, %f, %f, %f\n", cxdx2[0], cxdx2[1], cxdx2[2], cxdx2[3]);

				//int result3 = 0;// de_findMedianDebug(cxdx2, histogramX, fgXSumUpToIndex, index2, n, x, y);
				//printf("(%d,%d) = %d,%d | %d,%d | %d\n", x, y, result1, result2, saveIndex, saveIndex2, result3);

			}
			*/
		}
		//thread同期
		__syncthreads();
	}


	delete g2;
	if (tid == radius)
	{
		delete cxdx2;
	}
	__syncthreads();
}


//multi test用
void cu_filter2DTest(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<int>* g, float4* cxdx, DeviceArray<float>* cxdx2)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int> texG = TextureArray<int>(g, filterMode, sizeInfo);
	TextureArray<int> texF = TextureArray<int>(f, filterMode, sizeInfo);

	//縦方向版 shared
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