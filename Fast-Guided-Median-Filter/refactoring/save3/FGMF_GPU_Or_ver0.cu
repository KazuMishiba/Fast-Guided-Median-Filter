#include "FGMF_GPU_Or_ver0.cuh"
#include<opencv2/opencv.hpp>
#include <vector>

namespace FGMF_GPU_Or_ver0
{
	__constant__ int channelNum_g_plusOne;


	/*
	* c,dにDeviceArrayを使うと１チャンネルの場合で15%程遅くなる。ガイド３チャンネルだとより顕著に遅くなるだろう。
	* 逆に、c,dはテクスチャメモリにフェッチしておけばよい気がする。
	* これにより少なくとも１，３チャンネルはテンプレート使ってかける気がする。
	* f,gもまとめてテクスチャメモリにフェッチすればよいのでは？
	* 
	* とりあえず論文投稿時の実装をリファクタリング
	* より速いテスト実装として以下
	*	1-1ならf,gをfloat2としてテクスチャメモリにフェッチ
	*	1-3または3-1ならf,gをfloat4としてテクスチャメモリにフェッチ
	*	3-3ならf,gそれぞれをfloat4としてテクスチャメモリにフェッチ
	*	入力3チャンネルの場合は、カーネル内で同時に３チャンネル分処理
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


	
	__device__ inline float
		de_calculateHcum(const float2& cxdx, int*& fgSumUpToIndex)
	{
		return cxdx.x * fgSumUpToIndex[0] + cxdx.y * fgSumUpToIndex[1];
	}
	__device__ inline float
		de_calculateHcum(const float4& cxdx, const int* fgSumUpToIndex)
	{
		return cxdx.x * fgSumUpToIndex[0] + cxdx.y * fgSumUpToIndex[1] + cxdx.z * fgSumUpToIndex[2] + cxdx.w * fgSumUpToIndex[3];
	}
	__device__ inline float
		de_calculateHcum(const float* cxdx, const int* fgSumUpToIndex)
	{
		float h = 0.0f;
		for (int i = 0; i < channelNum_g_plusOne; i++)
			h += cxdx[i] * fgSumUpToIndex[i];
		return h;
	}

	__device__ inline float
		de_update_fgSumUpToIndex(const float2& cxdx, int* fgSumUpToIndex, const int* histogram, int index, const int sign)
	{
		fgSumUpToIndex[0] += histogram[index * 2] * sign;
		fgSumUpToIndex[1] += histogram[index * 2 + 1] * sign;
		return de_calculateHcum(cxdx, fgSumUpToIndex);
	}
	__device__ inline float
		de_update_fgSumUpToIndex(const float4& cxdx, int* fgSumUpToIndex, const int* histogram, int index, const int sign)
	{
		fgSumUpToIndex[0] += histogram[index * 4] * sign;
		fgSumUpToIndex[1] += histogram[index * 4 + 1] * sign;
		fgSumUpToIndex[2] += histogram[index * 4 + 2] * sign;
		fgSumUpToIndex[3] += histogram[index * 4 + 3] * sign;
		return de_calculateHcum(cxdx, fgSumUpToIndex);
	}
	__device__ inline float
		de_update_fgSumUpToIndex(const float* cxdx, int* fgSumUpToIndex, const int* histogram, int index, const int sign)
	{
		for (int i = 0; i < channelNum_g_plusOne; i++)
			fgSumUpToIndex[i] += histogram[index * channelNum_g_plusOne + i] * sign;
		return de_calculateHcum(cxdx, fgSumUpToIndex);
	}

	
	template<typename CD_TYPE>
	__device__ inline int
		de_findMedian(const CD_TYPE& cxdx, const int* histogram, int* fgSumUpToIndex, int& index)
	{
		float h = de_calculateHcum(cxdx, fgSumUpToIndex);
		const int flagA = h < 0.5f;
		const int flag2 = flagA - 1;
		const int sign = flagA * 2 - 1;
		while (true)
		{
			index += flagA;
			//if(histogram[index].x)
			{
				de_update_fgSumUpToIndex(cxdx, fgSumUpToIndex, histogram, index, sign);
				h = de_calculateHcum(cxdx, fgSumUpToIndex);
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
	
	/*
	//G1
	__device__ inline int
		de_findMedian(const float2& cxdx, const int* histogram, int* fgSumUpToIndex, int& index)
	{
		float h = de_calculateHcum(cxdx, fgSumUpToIndex);
		const int flagA = h < 0.5f;
		const int flag2 = flagA - 1;
		const int sign = flagA * 2 - 1;
		while (true)
		{
			index += flagA;
			//if(histogram[index].x)
			{
				h = de_update_fgSumUpToIndex(cxdx, fgSumUpToIndex,histogram, index, sign);
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
		de_findMedian(const float4& cxdx, const int* histogram, int* fgSumUpToIndex, int& index)
	{
		float h = de_calculateHcum(cxdx, fgSumUpToIndex);
		const int flagA = h < 0.5f;
		const int flag2 = flagA - 1;
		const int sign = flagA * 2 - 1;
		while (true)
		{
			index += flagA;
			//if(histogram[index].x)
			{
				h = de_update_fgSumUpToIndex(cxdx, fgSumUpToIndex, histogram, index, sign);
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
		de_findMedian(float* cxdx, int* histogram, int* fgSumUpToIndex, int& index)
	{
		float h = 0.0f;
		for (int i = 0; i < channelNum_g_plusOne; i++) {
			h += cxdx[i] * fgSumUpToIndex[i];
		}
		const int flagA = h < 0.5f;
		const int flag2 = flagA - 1;
		const int sign = flagA * 2 - 1;
		//const int k = n + 1;

		while (true)
		{
			index += flagA;
			//if(histogram[index].x)//ここはコメント外してもいいかも
			{
				for (int i = 0; i < channelNum_g_plusOne; i++)
					fgSumUpToIndex[i] += histogram[index * channelNum_g_plusOne + i] * sign;
				h = 0.0f;
				for (int i = 0; i < channelNum_g_plusOne ; i++)
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
	*/


	__device__ inline void
		de_addPixelToWindow(cudaTextureObject_t texF, cudaTextureObject_t texG, int x, int y, int* histogram, int* fgSumUpToIndex, int index)
	{
		int g = tex2D<int>(texG, x, y);
		int f = tex2D<int>(texF, x, y);
		atomicAdd(&histogram[f * 2], g);
		atomicAdd(&histogram[f * 2 + 1], 1);
		if (f <= index)
		{
			atomicAdd(&fgSumUpToIndex[0], g);
			atomicAdd(&fgSumUpToIndex[1], 1);
		}
	}

	__device__ inline void
		de_removePixelFromWindow(cudaTextureObject_t texF, cudaTextureObject_t texG, int x, int y, int* histogram, int* fgSumUpToIndex, int index)
	{
		int g = tex2D<int>(texG, x, y);
		int f = tex2D<int>(texF, x, y);
		atomicSub(&histogram[f * 2], g);
		atomicSub(&histogram[f * 2 + 1], 1);
		if (f <= index)
		{
			atomicSub(&fgSumUpToIndex[0], g);
			atomicSub(&fgSumUpToIndex[1], 1);
		}
	}

	template<typename CXDX_TYPE>
	__global__ void
		de_filter2D(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, CXDX_TYPE* cxdx, size_t pitchI1, size_t pitchF_channelNum_g_plusOne)
	{
		//処理対象の中心座標は blockIdx.x になる。
		/*
		* threadは0～radius*2 まで使用するとして、それ以上はreturnする
		* 中心threadのidxはradius
		*/

		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		int s_handle = s_center + tid - radius;

		extern __shared__ int  sharedData[];
		//元実装はf,gの順だったが、g,fの順に変える
		int* W_gf_cum = &sharedData[0];
		//int* W_GF = &sdata[2];
		int* W_GF = &sharedData[channelNum_g_plusOne];
		__shared__ int W_k;


		//中心スレッドのみ実行
		if (tid == radius)
		{
			//ヒストグラム初期化, fgSumUpToIndex初期化
			//for (int i = 0; i < (fRange + 1) * 2; i++)
			for (int i = 0; i < (fRange + 1) * channelNum_g_plusOne; i++)
				sharedData[i] = 0;
			W_k = tex2D<int>(texF, s_center, 0);//current index
		}

		__syncthreads();
		//1つ目ヒストグラム形成
		//x方向のヒストグラム形成は各スレッドが担当する
		for (int t = -radius; t <= radius; t++)
			de_addPixelToWindow(texF, texG, s_handle, t, W_GF, W_gf_cum, W_k);

		__syncthreads();
		//1行目の中央値計算
		//中心スレッドのみ実行
		if (tid == radius)
		{
			CXDX_TYPE _cxdx = *((CXDX_TYPE*)((char*)cxdx) + s_center);
			*((int*)((char*)result_center) + s_center) = de_findMedian<CXDX_TYPE>(_cxdx, W_GF, W_gf_cum, W_k);
		}

		//2行目以降の処理
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			//ヒストグラムに追加
			de_addPixelToWindow(texF, texG, s_handle, t + radius, W_GF, W_gf_cum, W_k);
			//ヒストグラムから削除
			de_removePixelFromWindow(texF, texG, s_handle, t - radius - 1, W_GF, W_gf_cum, W_k);

			__syncthreads();
			//中心スレッドのみ実行
			if (tid == radius)
			{
				CXDX_TYPE _cxdx = *((CXDX_TYPE*)((char*)cxdx + t * pitchF_channelNum_g_plusOne) + s_center);
				//中央値計算
				*((int*)((char*)result_center + t * pitchI1) + s_center) = de_findMedian<CXDX_TYPE>(_cxdx, W_GF, W_gf_cum, W_k);
			}
		}
	}


	
/**/
	__global__ void
		de_filter2D(int width, int height, int radius, int fRange, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t texG, float2* cxdx, size_t pitchI1, size_t pitchF_channelNum_g_plusOne)
	{
		//処理対象の中心座標は blockIdx.x になる。
		/*
		* threadは0～radius*2 まで使用するとして、それ以上はreturnする
		* 中心threadのidxはradius
		*/

		int s_center = blockIdx.x;
		int tid = threadIdx.x;
		if (s_center >= width || tid >= radius * 2 + 1)
			return;

		int s_handle = s_center + tid - radius;

		extern __shared__ int  sharedData[];
		//元実装はf,gの順だったが、g,fの順に変える
		int* W_gf_cum = &sharedData[0];
		//int* W_GF = &sdata[2];
		int* W_GF = &sharedData[channelNum_g_plusOne];
		__shared__ int W_k;


		//中心スレッドのみ実行
		if (tid == radius)
		{
			//ヒストグラム初期化, fgSumUpToIndex初期化
			//for (int i = 0; i < (fRange + 1) * 2; i++)
			for (int i = 0; i < (fRange + 1) * channelNum_g_plusOne; i++)
				sharedData[i] = 0;
			W_k = tex2D<int>(texF, s_center, 0);//current index
		}

		__syncthreads();
		//1つ目ヒストグラム形成
		//x方向のヒストグラム形成は各スレッドが担当する
		for (int t = -radius; t <= radius; t++)
			de_addPixelToWindow(texF, texG, s_handle, t, W_GF, W_gf_cum, W_k);

		__syncthreads();
		//1行目の中央値計算
		//中心スレッドのみ実行
		if (tid == radius)
		{
			float2 _cxdx = *((float2*)((char*)cxdx) + s_center);
			*((int*)((char*)result_center) + s_center) = de_findMedian<float2>(_cxdx, W_GF, W_gf_cum, W_k);
		}

		//2行目以降の処理
		for (int t = 1; t < height; t++)
		{
			__syncthreads();
			//ヒストグラムに追加
			de_addPixelToWindow(texF, texG, s_handle, t + radius, W_GF, W_gf_cum, W_k);
			//ヒストグラムから削除
			de_removePixelFromWindow(texF, texG, s_handle, t - radius - 1, W_GF, W_gf_cum, W_k);

			__syncthreads();
			//中心スレッドのみ実行
			if (tid == radius)
			{
				float2 _cxdx = *((float2*)((char*)cxdx + t * pitchF_channelNum_g_plusOne) + s_center);
				//中央値計算
				*((int*)((char*)result_center + t * pitchI1) + s_center) = de_findMedian<float2>(_cxdx, W_GF, W_gf_cum, W_k);
			}
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

		extern __shared__ int  sharedData[];
		//元実装はf,gの順だったが、g,fの順に変える
		int* fgSumUpToIndex = &sharedData[0];
		int* histogram4 = &sharedData[4];

		__shared__ int index;
		int f;
		int g[3];

		//中心スレッドのみ実行
		if (tid == radius)
		{
			//ヒストグラム初期化
			for (int i = 0; i < (Imax + 1) * 4; i++)
			{
				sharedData[i] = 0;
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
				g[0] = tex2D<int>(texG3[0], xPos, yy);
				g[1] = tex2D<int>(texG3[1], xPos, yy);
				g[2] = tex2D<int>(texG3[2], xPos, yy);
				atomicAdd(&histogram4[f * 4], g[0]);
				atomicAdd(&histogram4[f * 4 + 1], g[1]);
				atomicAdd(&histogram4[f * 4 + 2], g[2]);
				atomicAdd(&histogram4[f * 4 + 3], 1);
				if (f <= index)
				{
					atomicAdd(&fgSumUpToIndex[0], g[0]);
					atomicAdd(&fgSumUpToIndex[1], g[1]);
					atomicAdd(&fgSumUpToIndex[2], g[2]);
					atomicAdd(&fgSumUpToIndex[3], 1);
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
			*((int*)((char*)result_center) + x) = de_findMedian<float4>(cxdx, histogram4, fgSumUpToIndex, index);
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
			atomicAdd(&histogram4[f * 4], g[0]);
			atomicAdd(&histogram4[f * 4 + 1], g[1]);
			atomicAdd(&histogram4[f * 4 + 2], g[2]);
			atomicAdd(&histogram4[f * 4 + 3], 1);
			if (f <= index)
			{
				atomicAdd(&fgSumUpToIndex[0], g[0]);
				atomicAdd(&fgSumUpToIndex[1], g[1]);
				atomicAdd(&fgSumUpToIndex[2], g[2]);
				atomicAdd(&fgSumUpToIndex[3], 1);
			}
			//ヒストグラムから削除
			f = tex2D<int>(texF, xPos, y - radius - 1);
			g[0] = tex2D<int>(texG3[0], xPos, y - radius - 1);
			g[1] = tex2D<int>(texG3[1], xPos, y - radius - 1);
			g[2] = tex2D<int>(texG3[2], xPos, y - radius - 1);
			atomicSub(&histogram4[f * 4], g[0]);
			atomicSub(&histogram4[f * 4 + 1], g[1]);
			atomicSub(&histogram4[f * 4 + 2], g[2]);
			atomicSub(&histogram4[f * 4 + 3], 1);
			if (f <= index)
			{
				atomicSub(&fgSumUpToIndex[0], g[0]);
				atomicSub(&fgSumUpToIndex[1], g[1]);
				atomicSub(&fgSumUpToIndex[2], g[2]);
				atomicSub(&fgSumUpToIndex[3], 1);
			}

			//thread同期
			__syncthreads();
			//中心スレッドのみ実行
			if (tid == radius)
			{
				float4 cxdx = *((float4*)((char*)CxDx + y * pitchF4) + x);
				//中央値計算
				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian<float4>(cxdx, histogram4, fgSumUpToIndex, index);
			}
			//thread同期
			__syncthreads();
		}

	}


	//gX
	__global__ void
		de_filter2D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t* texGX, float** CxDx, size_t pitchI1, size_t pitchF1)
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
		int* fgXSumUpToIndex = &buffers[0];
		int* histogramX = &buffers[channelNum_g_plusOne];
		//g1,g3ではfgXSumUpToIndexはf,gの順で実装したが、cxdxと順番を合わせるために、g,...,g fの順にする
		//histogramも同様
		//histogramは1次元に並んでいて、各binについて、g,...,g,f の順に並んでいる
		const int n = channelNum_g_plusOne - 1;

		__shared__ int index;
		int f;
		int* g = new int[n];

		float* cxdx;

		//中心スレッドのみ実行
		if (tid == radius)
		{
			cxdx = new float[channelNum_g_plusOne];
			//ヒストグラム初期化
			for (int i = 0; i < Imax * channelNum_g_plusOne; i++)
			{
				histogramX[i] = 0;
			}
			for (int i = 0; i < channelNum_g_plusOne; i++)
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
		//thread同期
		__syncthreads();


		//1行目の中央値計算
		//中心スレッドのみ実行
		if (tid == radius)
		{
			for (int i = 0; i < channelNum_g_plusOne; i++)
				cxdx[i] = *((float*)((char*)CxDx[i]) + x);
//			*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index);
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
				atomicAdd(&histogramX[f * channelNum_g_plusOne + i], g[i]);
			atomicAdd(&histogramX[f * channelNum_g_plusOne + n], 1);
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
				atomicSub(&histogramX[f * channelNum_g_plusOne + i], g[i]);
			atomicSub(&histogramX[f * channelNum_g_plusOne + n], 1);
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
				for (int i = 0; i < channelNum_g_plusOne; i++)
					cxdx[i] = *((float*)((char*)CxDx[i] + y * pitchF1) + x);
				//中央値計算
//				*((int*)((char*)result_center + y * pitchI1) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, index);
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
/*
	//I1G1 refactoring
	void cu_filter2D(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, int* g, DeviceArray<float>* cxdx)
	{
		cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
		cudaTextureObject_t texF, texG;
		UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド
		UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texGにgをバインド

	//縦方向版 shared
		int blockSize = radius * 2 + 1;
		int gridSizeX = sizeInfo.width;
		de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int)* (1 + 1), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());

		cudaDestroyTextureObject(texF);
		cudaDestroyTextureObject(texG);
	}
	*/
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
		int k = 2;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		//de_filter2D << <gridSizeX, blockSize, Imax * sizeof(int2), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());
		de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int)* (1 + 1), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float2>());

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
		int k = 4;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		//de_filter2D << <gridSizeX, blockSize, Imax * sizeof(int) * 4, stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG.device, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
		de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int)* (1 + 3), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG.device, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());

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
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		//shared memoryは、ヒストグラム＋uptoindex必要で、ヒストグラムはImax * (n+1)、uptoindexはn+1必要
		de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int)* (n + 1), stream >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG.device, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());

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
		int k = 2;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
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
		int k = 4;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
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
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int)* (n + 1), NULL >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center, texF, texG.device, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
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
		int k = n + 1;
		cudaMemcpyToSymbol(channelNum_g_plusOne, &k, sizeof(int));
		for (int i = 0; i < m; i++)
		{
			de_filter2D << <gridSizeX, blockSize, (Imax + 1) * sizeof(int)* (n + 1), NULL >> > (sizeInfo.width, sizeInfo.height, radius, Imax, result_center->host[i], texF.host[i], texG.device, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>());
		}
		//cudaDeviceSynchronize();
	}






	//////////////////////////////////////////////////
	//3D
	
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
	__device__ inline int
		de_findMedian(const float4& cxdx, const int4* histogram, int4& fgSumUpToIndex, int& index)
	{
		float h = cxdx.x * fgSumUpToIndex.y + cxdx.y * fgSumUpToIndex.z + cxdx.z * fgSumUpToIndex.w + cxdx.w * fgSumUpToIndex.x;
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

		extern __shared__ int2 histogram2[];

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

		extern __shared__ int4 histogram4[];
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
		for (int k = 0, n = 0; k < numOfFrames; k++, n += 3)
		{
			for (int yy = -radius; yy <= radius; yy++)
			{
				//x方向のヒストグラム形成は各スレッドが担当する
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
			for (int k = 0, n = 0; k < numOfFrames; k++, n += 3)
			{
				//ヒストグラムに追加
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
					atomicAdd(&fgSumUpToIndex.x, 1);
					atomicAdd(&fgSumUpToIndex.y, g[0]);
					atomicAdd(&fgSumUpToIndex.z, g[1]);
					atomicAdd(&fgSumUpToIndex.w, g[2]);
				}
				//ヒストグラムから削除
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

}