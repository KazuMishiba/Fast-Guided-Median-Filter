#include "CalculateC_D_ver0.cuh"

namespace FGMF_GPU_Or_ver0
{

//境界拡張版
__global__ void
de_gsum_x(int width, int height, int radius, int2* sumG, cudaTextureObject_t texG, size_t pitchI2)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	int g;
	int _g;
	int sumg = 0;
	int sumgg = 0;

	//x=0
	for (int x = -radius; x <= radius; x++)
	{
		g = tex2D<int>(texG, x, y);
		sumg += g;
		sumgg += g * g;
	}
	*((int2*)((char*)sumG + y * pitchI2)) = make_int2(sumg, sumgg);

	for (int x = 1; x < width; x++)
	{
		g = tex2D<int>(texG, x + radius, y);
		_g = tex2D<int>(texG, x - radius - 1, y);
		sumg += g - _g;
		sumgg += g * g - _g * _g;
		*((int2*)((char*)sumG + y * pitchI2) + x) = make_int2(sumg, sumgg);
	}
}
__global__ void
de_gsum_y(int width, int height, int radius, int2* sumG, cudaTextureObject_t texSumG, size_t pitchI2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	int2 tmp, _tmp;
	int sumg = 0;
	int sumgg = 0;

	//y = 0
	for (int y = -radius; y <= radius; y++)
	{
		tmp = tex2D<int2>(texSumG, x, y);
		sumg += tmp.x;
		sumgg += tmp.y;
	}
	*((int2*)((char*)sumG) + x) = make_int2(sumg, sumgg);

	for (int y = 1; y < height; y++)
	{
		tmp = tex2D<int2>(texSumG, x, y + radius);
		_tmp = tex2D<int2>(texSumG, x, y - radius - 1);
		sumg += tmp.x - _tmp.x;
		sumgg += tmp.y - _tmp.y;
		*((int2*)((char*)sumG + y * pitchI2) + x) = make_int2(sumg, sumgg);
	}
}

//g3
__global__ void
de_g3sum_x(int width, int height, int radius, int** sumG, int** sumGG, cudaTextureObject_t* texG3, size_t pitchI1)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	int g[3];
	int _g[3];
	int sumg[3] = { 0, 0, 0 };
	int sumgg[6] = { 0,0,0,0,0,0 };//11,12,13,22,23,33

	//x=0
	for (int x = -radius; x <= radius; x++)
	{
		for (int i = 0; i < 3; i++)
			g[i] = tex2D<int>(texG3[i], x, y);

		for (int i = 0; i < 3; i++)
			sumg[i] += g[i];

		int n = 0;
		for (int i = 0; i < 3; i++) {
			for (int j = i; j < 3; j++) {
				sumgg[n] += g[i] * g[j];
				n++;
			}
		}
	}
	for (int i = 0; i < 3; i++)
		*((int*)((char*)(sumG[i]) + y * pitchI1)) = sumg[i];
	for (int i = 0; i < 6; i++)
		*((int*)((char*)sumGG[i] + y * pitchI1)) = sumgg[i];

	for (int x = 1; x < width; x++)
	{
		for (int i = 0; i < 3; i++)
			g[i] = tex2D<int>(texG3[i], x + radius, y);
		for (int i = 0; i < 3; i++)
			_g[i] = tex2D<int>(texG3[i], x - radius - 1, y);
		for (int i = 0; i < 3; i++)
			sumg[i] += g[i] - _g[i];
		int n = 0;
		for (int i = 0; i < 3; i++) {
			for (int j = i; j < 3; j++) {
				sumgg[n] += g[i] * g[j] - _g[i] * _g[j];
				n++;
			}
		}
		for (int i = 0; i < 3; i++)
			*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg[i];
		for (int i = 0; i < 6; i++)
			*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg[i];
	}
	
}


__global__ void
de_g3sum_y(int width, int height, int radius, int** sumG, int** sumGG, cudaTextureObject_t* texSumG, cudaTextureObject_t* texSumGG, size_t pitchI1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;
	
	int sumg[3] = { 0, 0, 0 };
	int sumgg[6] = { 0, 0, 0, 0, 0, 0 };

	//y = 0
	for (int y = -radius; y <= radius; y++)
	{
		for (int i = 0; i < 3; i++)
			sumg[i] += tex2D<int>(texSumG[i], x, y);
		for (int i = 0; i < 6; i++)
			sumgg[i] += tex2D<int>(texSumGG[i], x, y);

	}
	for (int i = 0; i < 3; i++)
		*((int*)((char*)sumG[i]) + x) = sumg[i];
	for (int i = 0; i < 6; i++)
		*((int*)((char*)sumGG[i]) + x) = sumgg[i];

	for (int y = 1; y < height; y++)
	{
		for (int i = 0; i < 3; i++)
			sumg[i] += tex2D<int>(texSumG[i], x, y + radius) - tex2D<int>(texSumG[i], x, y - radius - 1);
		for (int i = 0; i < 6; i++)
			sumgg[i] += tex2D<int>(texSumGG[i], x, y + radius) - tex2D<int>(texSumGG[i], x, y - radius - 1);

		for (int i = 0; i < 3; i++)
			*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg[i];
		for (int i = 0; i < 6; i++)
			*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg[i];
	}	
}


//gX
__global__ void
de_gXsum_x(int width, int height, int radius, int** sumG, int** sumGG, cudaTextureObject_t* texGX, size_t pitchI1, int n)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	const int m = (n + 1)*n / 2;
	int *g = new int[n];
	int *_g = new int[n];
	int *sumg = new int[n];
	for (int i = 0; i < n; i++)
		sumg[i] = 0;
	int *sumgg = new int[m];
	for (int i = 0; i < m; i++)
		sumgg[i] = 0;


	//x=0
	for (int x = -radius; x <= radius; x++)
	{
		for (int i = 0; i < n; i++)
			g[i] = tex2D<int>(texGX[i], x, y);

		for (int i = 0; i < n; i++)
			sumg[i] += g[i];

		int k = 0;
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				sumgg[k] += g[i] * g[j];
				k++;
			}
		}
	}
	for (int i = 0; i < n; i++)
		*((int*)((char*)(sumG[i]) + y * pitchI1)) = sumg[i];
	for (int i = 0; i < m; i++)
		*((int*)((char*)sumGG[i] + y * pitchI1)) = sumgg[i];

	for (int x = 1; x < width; x++)
	{
		for (int i = 0; i < n; i++)
			g[i] = tex2D<int>(texGX[i], x + radius, y);
		for (int i = 0; i < n; i++)
			_g[i] = tex2D<int>(texGX[i], x - radius - 1, y);
		for (int i = 0; i < n; i++)
			sumg[i] += g[i] - _g[i];
		int k = 0;
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				sumgg[k] += g[i] * g[j] - _g[i] * _g[j];
				k++;
			}
		}
		for (int i = 0; i < n; i++)
			*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg[i];
		for (int i = 0; i < m; i++)
			*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg[i];
	}

	delete g;
	delete _g;
	delete sumg;
	delete sumgg;

}

__global__ void
de_gXsum_y(int width, int height, int radius, int** sumG, int** sumGG, cudaTextureObject_t* texSumG, cudaTextureObject_t* texSumGG, size_t pitchI1, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	const int m = (n + 1)*n / 2;
	int *sumg = new int[n];
	for (int i = 0; i < n; i++)
		sumg[i] = 0;
	int *sumgg = new int[m];
	for (int i = 0; i < m; i++)
		sumgg[i] = 0;


	//y = 0
	for (int y = -radius; y <= radius; y++)
	{
		for (int i = 0; i < n; i++)
			sumg[i] += tex2D<int>(texSumG[i], x, y);
		for (int i = 0; i < m; i++)
			sumgg[i] += tex2D<int>(texSumGG[i], x, y);

	}
	for (int i = 0; i < n; i++)
		*((int*)((char*)sumG[i]) + x) = sumg[i];
	for (int i = 0; i < m; i++)
		*((int*)((char*)sumGG[i]) + x) = sumgg[i];

	for (int y = 1; y < height; y++)
	{
		for (int i = 0; i < n; i++)
			sumg[i] += tex2D<int>(texSumG[i], x, y + radius) - tex2D<int>(texSumG[i], x, y - radius - 1);
		for (int i = 0; i < m; i++)
			sumgg[i] += tex2D<int>(texSumGG[i], x, y + radius) - tex2D<int>(texSumGG[i], x, y - radius - 1);

		for (int i = 0; i < n; i++)
			*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg[i];
		for (int i = 0; i < m; i++)
			*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg[i];
	}

	delete sumg;
	delete sumgg;
}


//refactoring用cx,dx float2ではなくDeviceArray<float>* cxdx
__global__ void
de_calculateCxDx(int width, int height, int* G, int2* sumG, float eps2, float pixel_sum_window_inv, float** cxdx, size_t pitchI1, size_t pitchI2, size_t pitchF1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int2 tmp = *((int2*)((char*)sumG + y * pitchI2) + x);
	int g = *((int*)((char*)G + y * pitchI1) + x);
	float g_ave = ((float)tmp.x) * pixel_sum_window_inv;
	float gg_ave = ((float)tmp.y) * pixel_sum_window_inv;
	float vx = gg_ave - g_ave * g_ave + eps2;
	float tmp2 = ((float)g) - g_ave;
	float cx2 = tmp2 * pixel_sum_window_inv / vx;
	//*((float2*)((char*)cxdx + y * pitchF2) + x) = make_float2(cx2, pixel_sum_window_inv - g_ave * cx2);
	*((float*)((char*)cxdx[0] + y * pitchF1) + x) = cx2;
	*((float*)((char*)cxdx[1] + y * pitchF1) + x) = pixel_sum_window_inv - g_ave * cx2;
}


//cx,dx float2
__global__ void
de_calculateCxDx(int width, int height, int* G, int2* sumG, float eps2, float pixel_sum_window_inv, float2* cxdx, size_t pitchI1, size_t pitchI2, size_t pitchF2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int2 tmp = *((int2*)((char*)sumG + y * pitchI2) + x);
	int g = *((int*)((char*)G + y * pitchI1) + x);
	float g_ave = ((float)tmp.x) * pixel_sum_window_inv;
	float gg_ave = ((float)tmp.y) * pixel_sum_window_inv;
	float vx = gg_ave - g_ave * g_ave + eps2;
	float tmp2 = ((float)g) - g_ave;
	float cx2 = tmp2 * pixel_sum_window_inv / vx;
	*((float2*)((char*)cxdx + y * pitchF2) + x) = make_float2(cx2, pixel_sum_window_inv - g_ave * cx2);
}
//cx3
__global__ void
de_calculateCx3Dx(int width, int height, int** G3, int** sumG, int** sumGG, float eps2, float pixel_sum_window_inv, float4* cxdx, size_t pitchI1, size_t pitchF4)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	const float g_ave1 = *((int*)((char*)sumG[0] + y * pitchI1) + x) * pixel_sum_window_inv;
	const float g_ave2 = *((int*)((char*)sumG[1] + y * pitchI1) + x) * pixel_sum_window_inv;
	const float g_ave3 = *((int*)((char*)sumG[2] + y * pitchI1) + x) * pixel_sum_window_inv;
	const float v11 = *((int*)((char*)sumGG[0] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave1 + eps2;
	const float v12 = *((int*)((char*)sumGG[1] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave2;
	const float v13 = *((int*)((char*)sumGG[2] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave3;
	const float v22 = *((int*)((char*)sumGG[3] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave2 * g_ave2 + eps2;
	const float v23 = *((int*)((char*)sumGG[4] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave2 * g_ave3;
	const float v33 = *((int*)((char*)sumGG[5] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave3 * g_ave3 + eps2;
	const float delta =
		v11 * v22 * v33 +
		v12 * v23 * v13 * 2 -
		v13 * v13 * v22 -
		v12 * v12 * v33 -
		v11 * v23 * v23;
	if (abs(delta) > 0.000001f)
	{
		const float deltaInv = 1.0f / delta;
		const float vinv11 = (v22 * v33 - v23 * v23);
		const float vinv12 = (v13 * v23 - v12 * v33);
		const float vinv13 = (v12 * v23 - v13 * v22);
		const float vinv22 = (v11 * v33 - v13 * v13);
		const float vinv23 = (v13 * v12 - v11 * v23);
		const float vinv33 = (v11 * v22 - v12 * v12);
		const float tmp1 = *((int*)((char*)G3[0] + y * pitchI1) + x) - g_ave1;
		const float tmp2 = *((int*)((char*)G3[1] + y * pitchI1) + x) - g_ave2;
		const float tmp3 = *((int*)((char*)G3[2] + y * pitchI1) + x) - g_ave3;
		const float mult = pixel_sum_window_inv * deltaInv;
		const float cx1 = (tmp1 * vinv11 + tmp2 * vinv12 + tmp3 * vinv13) * mult;
		const float cx2 = (tmp1 * vinv12 + tmp2 * vinv22 + tmp3 * vinv23) * mult;
		const float cx3 = (tmp1 * vinv13 + tmp2 * vinv23 + tmp3 * vinv33) * mult;
		const float dx = pixel_sum_window_inv - g_ave1 * cx1 - g_ave2 * cx2 - g_ave3 * cx3;

		*((float4*)((char*)cxdx + y * pitchF4) + x) = make_float4(cx1, cx2, cx3, dx);
	}
	else
	{
		//逆行列が存在しないので、cx=0, dx=画素数の逆数とする
		*((float4*)((char*)cxdx + y * pitchF4) + x) = make_float4(0.0f, 0.0f, 0.0f, pixel_sum_window_inv);
		//printf("%f\n", delta);
	}
	/*
	if (x == 50 && y == 50)
	{
		printf("g3\n");
		printf("%f %f %f\n", g_ave1, g_ave2, g_ave3);

		printf("%f %f %f %f %f %f\n", v11, v12, v13, v22, v23, v33);
		printf("%f %f %f %f\n", cx1/pixel_sum_window_inv, cx2/ pixel_sum_window_inv, cx3/ pixel_sum_window_inv, dx);
		printf("%f\n", pixel_sum_window_inv);

	}
	*/
}
//cxX
__global__ void
de_calculateCxXDx(int width, int height, int** GX, int** sumG, int** sumGG, float eps2, float pixel_sum_window_inv, float** cxdx, size_t pitchI1, size_t pitchF1, int n)
{
	//cxdxの要素数はN+1 (cxがN、dxが1)
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;
	
#if 1


	const int m = (n + 1)*n / 2;

	float *g_ave = new float[n];
	for (int i = 0; i < n; i++)
		g_ave[i] = *((int*)((char*)sumG[i] + y * pitchI1) + x) * pixel_sum_window_inv;
	float *A = new float[m];
	int k = 0;
	for (int j = 0; j < n; j++) {
		A[k] = *((int*)((char*)sumGG[k] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave[j] * g_ave[j] + eps2;
		k++;
		for (int i = j + 1; i < n; i++) {
			A[k] = *((int*)((char*)sumGG[k] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave[i] * g_ave[j];
			k++;
		}
	}



	//共役勾配法
	//初期値0
	//メモリ確保
	float *cx = new float[n];
	float *r = new float[n];
	float *p = new float[n];
	float *Ap = new float[n];
	float rsold = 0.0f;
	float alpha, rsnew;

	bool flag = true;

	for (int i = 0; i < n; i++)
	{
		r[i] = *((int*)((char*)GX[i] + y * pitchI1) + x) - g_ave[i];
		p[i] = r[i];
		rsold += r[i] * r[i];
		cx[i] = 0.0f;

		flag = flag & (r[i] == 0.0f);
	}



	if (!flag)
	{
		for (int iter = 0; iter < n; iter++)
		{
			alpha = 0.0f;
			//Ap = A * p
			int t = 0;
			for (int j = 0; j < n; j++)
			{
				//対角の演算結果で初期化
				//i == j
				Ap[j] = A[t] * p[j];

				//ミラー成分
				int m = j;
				int d = n - 1;
				for (int i = 0; i < j; i++)
				{
					Ap[j] += A[m] * p[i];
					m += d;
					d--;
				}
				//
				for (int i = j + 1; i < n; i++)
				{
					t++;
					Ap[j] += A[t] * p[i];
				}
				t++;

				//alpha = rsold / (p' * Ap);
				alpha += p[j] * Ap[j];
			}

			alpha = rsold / alpha;

			rsnew = 0.0f;
			for (int i = 0; i < n; i++)
			{
				//x = x + alpha * p;
				cx[i] += alpha * p[i];
				//r = r - alpha * Ap;
				r[i] -= alpha * Ap[i];
				//rsnew = r' * r;
				rsnew += r[i] * r[i];
			}
			if (rsnew < 0.000000000000001)
			{
				//printf("b:");
				break;
			}
			//p = r + (rsnew / rsold) * p;
			float no = rsnew / rsold;
			for (int i = 0; i < n; i++)
			{
				p[i] = r[i] + no * p[i];
			}
			//rsold = rsnew;
			rsold = rsnew;

		}

		float dx = pixel_sum_window_inv;
		for (int i = 0; i < n; i++)
		{
			cx[i] *= pixel_sum_window_inv;
			dx -= cx[i] * g_ave[i];
			*((float*)((char*)cxdx[i] + y * pitchF1) + x) = cx[i];

		}
		*((float*)((char*)cxdx[n] + y * pitchF1) + x) = dx;
	}
	else
	{
		//逆行列が存在しない場合
		//cx全て０、dxは画素数の逆数（結果としては平均値フィルタカーネルと同じ）
		for (int i = 0; i < n; i++)
		{
			*((float*)((char*)cxdx[i] + y * pitchF1) + x) = 0.0f;

		}
		*((float*)((char*)cxdx[n] + y * pitchF1) + x) = pixel_sum_window_inv;
	}

	//if (isnan(cx[0]))
	{
		/*
		printf("\n");

		printf("g:\n");
		for (int i = 0; i < n; i++)
			printf("%f ", *((int*)((char*)GX[i] + y * pitchI1) + x));
		printf("\n");
		*/
		/*
		printf("gave:\n");
		for (int i = 0; i < n; i++)
			printf("%f ", g_ave[i]);
		printf("\n");

		printf("g-gave:\n");
		for (int i = 0; i < n; i++)
			printf("%f ", *((int*)((char*)GX[i] + y * pitchI1) + x) - g_ave[i]);
		printf("\n");

		printf("cx:\n");
		for (int i = 0; i < n; i++)
			printf("%f ", cx[i]);
		printf("\n");
		*/
	}

	delete g_ave;
	delete cx;
	delete A;
	delete r;
	delete p;
	delete Ap;

#endif




#if 0
	const float g_ave1 = *((int*)((char*)sumG[0] + y * pitchI1) + x) * pixel_sum_window_inv;
	const float g_ave2 = *((int*)((char*)sumG[1] + y * pitchI1) + x) * pixel_sum_window_inv;
	const float g_ave3 = *((int*)((char*)sumG[2] + y * pitchI1) + x) * pixel_sum_window_inv;
	const float v11 = *((int*)((char*)sumGG[0] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave1 + eps2;
	const float v12 = *((int*)((char*)sumGG[1] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave2;
	const float v13 = *((int*)((char*)sumGG[2] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave1 * g_ave3;
	const float v22 = *((int*)((char*)sumGG[3] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave2 * g_ave2 + eps2;
	const float v23 = *((int*)((char*)sumGG[4] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave2 * g_ave3;
	const float v33 = *((int*)((char*)sumGG[5] + y * pitchI1) + x) * pixel_sum_window_inv - g_ave3 * g_ave3 + eps2;
	const float delta =
		v11 * v22 * v33 +
		v12 * v23 * v13 * 2 -
		v13 * v13 * v22 -
		v12 * v12 * v33 -
		v11 * v23 * v23;
	const float deltaInv = 1.0f / delta;
	const float vinv11 = (v22 * v33 - v23 * v23);
	const float vinv12 = (v13 * v23 - v12 * v33);
	const float vinv13 = (v12 * v23 - v13 * v22);
	const float vinv22 = (v11 * v33 - v13 * v13);
	const float vinv23 = (v13 * v12 - v11 * v23);
	const float vinv33 = (v11 * v22 - v12 * v12);
	const float tmp1 = *((int*)((char*)GX[0] + y * pitchI1) + x) - g_ave1;
	const float tmp2 = *((int*)((char*)GX[1] + y * pitchI1) + x) - g_ave2;
	const float tmp3 = *((int*)((char*)GX[2] + y * pitchI1) + x) - g_ave3;
	const float mult = pixel_sum_window_inv * deltaInv;
	const float cx1 = (tmp1 * vinv11 + tmp2 * vinv12 + tmp3 * vinv13) * mult;
	const float cx2 = (tmp1 * vinv12 + tmp2 * vinv22 + tmp3 * vinv23) * mult;
	const float cx3 = (tmp1 * vinv13 + tmp2 * vinv23 + tmp3 * vinv33) * mult;
	const float dx = pixel_sum_window_inv - g_ave1 * cx1 - g_ave2 * cx2 - g_ave3 * cx3;


	*((float*)((char*)cxdx[0] + y * pitchF1) + x) = cx1;
	*((float*)((char*)cxdx[1] + y * pitchF1) + x) = cx2;
	*((float*)((char*)cxdx[2] + y * pitchF1) + x) = cx3;
	*((float*)((char*)cxdx[3] + y * pitchF1) + x) = dx;

#endif





}

/*
共役勾配法実装メモ
MATLABコード
function x = conjgrad(A, b, x)
	r = b - A * x;
	p = r;
	rsold = r' * r;

	for i = 1:length(b)
		Ap = A * p;
		alpha = rsold / (p' * Ap);
		x = x + alpha * p;
		r = r - alpha * Ap;
		rsnew = r' * r;
		if sqrt(rsnew) < 1e-10
			  break
		end
		p = r + (rsnew / rsold) * p;
		rsold = rsnew;
	end
end

%A = a * a' + c Iのときの動作
function x = conjgrad2(a, b, c, x)
	r = b - a * (a' * x) - c * x;
	p = r;
	rsold = r' * r;

	for i = 1:length(b)
		Ap = a * (a' * p) + c * p;
		alpha = rsold / (p' * Ap);
		x = x + alpha * p;
		r = r - alpha * Ap;
		rsnew = r' * r;
		if sqrt(rsnew) < 1e-10
			  break
		end
		p = r + (rsnew / rsold) * p;
		rsold = rsnew;
	end
end
これはウィンドウ半径内の画素を記録しないといけないので今回使えない。

*/


//sumG(sumg, sumgg, pixel_num, g) を計算
void cu_calculateSumG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* sumG, int2* temp)
{
	int blockSize = BLOCK_SIZE_1D;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);

	cudaTextureObject_t texG;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	//texGにGをバインド
	UtilityForCUDA::setLinearArrayToTexture(G, texG, sizeInfo, filterMode);
	//texGの内容からsumGなど計算しtempに格納
	de_gsum_x << < gridSizeY, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, temp, texG, sizeInfo.pitch<int2>());
	//tempをtexGにバインドし、縦方向のsumGなど計算しsumGに格納
	UtilityForCUDA::setLinearArrayToTexture(temp, texG, sizeInfo, filterMode);
	//texGの内容からsumGなど計算しsumGに格納
	de_gsum_y << < gridSizeX, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, sumG, texG, sizeInfo.pitch<int2>());

	cudaDestroyTextureObject(texG);
}
//g3
void cu_calculateSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G3, int radius, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG)
{
	int blockSize = BLOCK_SIZE_1D;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);

	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int>* texG = new TextureArray<int>(G3, filterMode, sizeInfo);
	//texGの内容からgsumをtempGに、ggsumをtempGGに格納
	de_g3sum_x << < gridSizeY, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, tempG->device, tempGG->device, texG->device, sizeInfo.pitch<int>());
	//tempGをtexSumGに、tempGGをtexSumGGにバインド
	TextureArray<int>* texSumG = new TextureArray<int>(tempG, filterMode, sizeInfo);
	TextureArray<int>* texSumGG = new TextureArray<int>(tempGG, filterMode, sizeInfo);
	//sumG、sumGG計算
	de_g3sum_y << < gridSizeX, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, sumG->device, sumGG->device, texSumG->device, texSumGG->device, sizeInfo.pitch<int>());
	
	delete texG;
	delete texSumG;
	delete texSumGG;
}
//gX
void cu_calculateSumGX(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* GX, int radius, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG, int n)
{
	int blockSize = BLOCK_SIZE_1D;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);

	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	TextureArray<int>* texG = new TextureArray<int>(GX, filterMode, sizeInfo);
	//texGの内容からgsumをtempGに、ggsumをtempGGに格納
	de_gXsum_x << < gridSizeY, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, tempG->device, tempGG->device, texG->device, sizeInfo.pitch<int>(), n);
	//tempGをtexSumGに、tempGGをtexSumGGにバインド
	TextureArray<int>* texSumG = new TextureArray<int>(tempG, filterMode, sizeInfo);
	TextureArray<int>* texSumGG = new TextureArray<int>(tempGG, filterMode, sizeInfo);
	//sumG、sumGG計算
	de_gXsum_y << < gridSizeX, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, sumG->device, sumGG->device, texSumG->device, texSumGG->device, sizeInfo.pitch<int>(), n);

	delete texG;
	delete texSumG;
	delete texSumGG;
}



//refactoring用　cx,dxをfloat2ではなくDeviceArray<float>* cxdxで定義
void cu_calculateCxDxFromG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, DeviceArray<float>* cxdx, int2* sumG, int2* temp)
{
	cu_calculateSumG(sizeInfo, stream, G, radius, sumG, temp);
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow; ((radius * 2 + 1) * (radius * 2 + 1));
	//sumGの値からcx, dxを計算
	de_calculateCxDx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, sumG, eps2, pixel_sum_window_inv, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float>());
}



//sumgも計算 2D用 cx,dxをfloat2で定義
void cu_calculateCxDxFromG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, float2* cxdx, int2* sumG, int2* temp)
{
	cu_calculateSumG(sizeInfo, stream, G, radius, sumG, temp);
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow; ((radius * 2 + 1) * (radius * 2 + 1));
	//sumGの値からcx, dxを計算
	de_calculateCxDx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, sumG, eps2, pixel_sum_window_inv, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float2>());
}
//g3
void cu_calculateCx3DxFromG(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G3, int radius, int pixelNumInWindow, float eps2, float4* cxdx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG)
{
	cu_calculateSumG3(sizeInfo, stream, G3, radius, sumG, sumGG, tempG, tempGG);
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	//sumGの値からcx, dxを計算
	de_calculateCx3Dx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G3->device, sumG->device, sumGG->device, eps2, pixel_sum_window_inv, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
}

//gX
void cu_calculateCxXDxFromG(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* GX, int radius, int pixelNumInWindow, float eps2, DeviceArray<float>* cxdx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG)
{
	int n = GX->arrayLength;
	cu_calculateSumGX(sizeInfo, stream, GX, radius, sumG, sumGG, tempG, tempGG, n);
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	//sumGの値からcx, dxを計算
	//線型方程式を解いているこの部分がほとんどの時間を消費している
	de_calculateCxXDx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, GX->device, sumG->device, sumGG->device, eps2, pixel_sum_window_inv, cxdx->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>(), n);

	//Utility::showDevice(cxdx->host[0], sizeInfo, "cxdx", true, 100000.0f);
}

//////////////////////////////////////////////////////////
// 3D以上用


//追加と削除
__global__ void
de_updateSumG(int width, int height, int2* addSumG, int2* remSumG, int2* sumG, size_t pitchI2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int2 addsumg = *((int2*)((char*)addSumG + y * pitchI2) + x);
	int2 remsumg = *((int2*)((char*)remSumG + y * pitchI2) + x);
	int2 sumg = *((int2*)((char*)sumG + y * pitchI2) + x);
	*((int2*)((char*)sumG + y * pitchI2) + x) = make_int2(
		sumg.x + addsumg.x - remsumg.x,
		sumg.y + addsumg.y - remsumg.y
	);
}
__global__ void
de_updateSumG3(int width, int height, int** addSumG, int** addSumGG, int** remSumG, int** remSumGG, int** sumG, int** sumGG, size_t pitchI1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	for (int i = 0; i < 3; i++)
	{
		int addsumg = *((int*)((char*)addSumG[i] + y * pitchI1) + x);
		int remsumg = *((int*)((char*)remSumG[i] + y * pitchI1) + x);
		int sumg = *((int*)((char*)sumG[i] + y * pitchI1) + x);
		*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg + addsumg - remsumg;
	}
	for (int i = 0; i < 6; i++)
	{
		int addsumgg = *((int*)((char*)addSumGG[i] + y * pitchI1) + x);
		int remsumgg = *((int*)((char*)remSumGG[i] + y * pitchI1) + x);
		int sumgg = *((int*)((char*)sumGG[i] + y * pitchI1) + x);
		*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg + addsumgg - remsumgg;
	}
}
//追加のみ
__global__ void
de_addSumG(int width, int height, int2* addSumG, int2* sumG, size_t pitchI2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int2 addsumg = *((int2*)((char*)addSumG + y * pitchI2) + x);
	int2 sumg = *((int2*)((char*)sumG + y * pitchI2) + x);
	*((int2*)((char*)sumG + y * pitchI2) + x) = make_int2(
		sumg.x + addsumg.x,
		sumg.y + addsumg.y
	);
}
__global__ void
de_addSumG3(int width, int height, int** addSumG, int** addSumGG, int** sumG, int** sumGG, size_t pitchI1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	for (int i = 0; i < 3; i++)
	{
		int addsumg = *((int*)((char*)addSumG[i] + y * pitchI1) + x);
		int sumg = *((int*)((char*)sumG[i] + y * pitchI1) + x);
		*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg + addsumg;
	}
	for (int i = 0; i < 6; i++)
	{
		int addsumgg = *((int*)((char*)addSumGG[i] + y * pitchI1) + x);
		int sumgg = *((int*)((char*)sumGG[i] + y * pitchI1) + x);
		*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg + addsumgg;
	}
}
//削除のみ
__global__ void
de_remSumG(int width, int height, int2* remSumG, int2* sumG, size_t pitchI2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int2 remsumg = *((int2*)((char*)remSumG + y * pitchI2) + x);
	int2 sumg = *((int2*)((char*)sumG + y * pitchI2) + x);
	*((int2*)((char*)sumG + y * pitchI2) + x) = make_int2(
		sumg.x - remsumg.x,
		sumg.y - remsumg.y
	);
}
__global__ void
de_remSumG3(int width, int height, int** remSumG, int** remSumGG, int** sumG, int** sumGG, size_t pitchI1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	for (int i = 0; i < 3; i++)
	{
		int remsumg = *((int*)((char*)remSumG[i] + y * pitchI1) + x);
		int sumg = *((int*)((char*)sumG[i] + y * pitchI1) + x);
		*((int*)((char*)sumG[i] + y * pitchI1) + x) = sumg - remsumg;
	}
	for (int i = 0; i < 6; i++)
	{
		int remsumgg = *((int*)((char*)remSumGG[i] + y * pitchI1) + x);
		int sumgg = *((int*)((char*)sumGG[i] + y * pitchI1) + x);
		*((int*)((char*)sumGG[i] + y * pitchI1) + x) = sumgg - remsumgg;
	}
}

//update
void cu_updateSumG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* addSumG, int2* remSumG, int2* sumG, int2* temp)
{
	cu_calculateSumG(sizeInfo, stream, G, radius, addSumG, temp);
	de_updateSumG << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, addSumG, remSumG, sumG, sizeInfo.pitch<int2>());
}
//g3
void cu_updateSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, DeviceArray<int>* addSumG, DeviceArray<int>* addSumGG, DeviceArray<int>* remSumG, DeviceArray<int>* remSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG)
{
	cu_calculateSumG3(sizeInfo, stream, G, radius, addSumG, addSumGG, tempG, tempGG);
	de_updateSumG3 << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, addSumG->device, addSumGG->device, remSumG->device, remSumGG->device, sumG->device, sumGG->device, sizeInfo.pitch<int>());
}

//add
void cu_addSumG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* addSumG, int2* sumG, int2* temp)
{
	cu_calculateSumG(sizeInfo, stream, G, radius, addSumG, temp);
	de_addSumG << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, addSumG, sumG, sizeInfo.pitch<int2>());
}
//g3
void cu_addSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, DeviceArray<int>* addSumG, DeviceArray<int>* addSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG)
{
	cu_calculateSumG3(sizeInfo, stream, G, radius, addSumG, addSumGG, tempG, tempGG);
	de_addSumG3 << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, addSumG->device, addSumGG->device, sumG->device, sumGG->device, sizeInfo.pitch<int>());

}

//rem
void cu_remSumG(SizeInfo& sizeInfo, cudaStream_t stream, int2* remSumG, int2* sumG)
{
	de_remSumG << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, remSumG, sumG, sizeInfo.pitch<int2>());
}
//g3
void cu_remSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* remSumG, DeviceArray<int>* remSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG)
{
	de_remSumG3 << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, remSumG->device, remSumGG->device, sumG->device, sumGG->device, sizeInfo.pitch<int>());
}

//cxdx
void cu_calculateCxDx(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, float2* cxdx, int2* sumG)
{
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	//sumGの値からcx, dxを計算
	de_calculateCxDx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, sumG, eps2, pixel_sum_window_inv, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float2>());
}
//g3
void cu_calculateCx3Dx(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, int pixelNumInWindow, float eps2, float4* cxdx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG)
{
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	//sumGの値からcx, dxを計算
	de_calculateCx3Dx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G->device, sumG->device, sumGG->device, eps2, pixel_sum_window_inv, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
}



/*
//sumgも計算 3D用 追加と削除を行う版　addSumGに追加するsumgを計算し、今のsumGに加えるとともにremSumGを削除する addSumGは確保された空のものを渡し、remSumGは中身のある削除するやつを渡す
void cu_updateCxDx(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, float eps2, float* cx, float* dx, int2* addSumG, int2* remSumG, int2* sumG, int2* temp)
{
	cu_calculateSumG(sizeInfo, stream, G, radius, addSumG, temp);
	//addSumGの追加とremSumGの削除
	de_updateSumG << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, addSumG, remSumG, sumG, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>());
	//sumGの値からcx, dxを計算
	de_calculateCxDx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, sumG, eps2, cx, dx, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float>());
}
//追加のみ
void cu_updateCxDx_add(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, float eps2, float* cx, float* dx, int2* addSumG, int2* sumG, int2* temp)
{
	cu_calculateSumG(sizeInfo, stream, G, radius, addSumG, temp);
	//addSumGの追加
	de_addSumG << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, addSumG, sumG, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>());
	//sumGの値からcx, dxを計算
	de_calculateCxDx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, sumG, eps2, cx, dx, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float>());
}
//削除のみ
void cu_updateCxDx_rem(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, float eps2, float* cx, float* dx, int2* remSumG, int2* sumG, int2* temp)
{
	//remSumGの削除
	de_remSumG << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, remSumG, sumG, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>());
	//sumGの値からcx, dxを計算
	de_calculateCxDx << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, sumG, eps2, cx, dx, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float>());
}
*/

#if 0
//境界拡張しない版
//境界拡張しない版
__global__ void
de_gsum_x(int width, int height, int radius, int4* sumG, cudaTextureObject_t texG, size_t pitchI4)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;


	int g;
	int _g;
	int sumg = 0.0f;
	int sumgg = 0.0f;
	int pixNum = radius + 1;

	//x=0
	for (int x = 0; x <= radius; x++)
	{
		g = tex2D<int>(texG, float(x) + 0.5f, float(y) + 0.5f);
		sumg += g;
		sumgg += g * g;
	}
	*((int4*)((char*)sumG + y * pitchI4)) = make_int4(sumg, sumgg, pixNum, g);

	int x = 1;
	//x=1~radius
	for (; x <= radius; x++)
	{
		g = tex2D<int>(texG, float(x + radius) + 0.5f, float(y) + 0.5f);
		sumg += g;
		sumgg += g * g;
		pixNum++;
		*((int4*)((char*)sumG + y * pitchI4) + x) = make_int4(sumg, sumgg, pixNum, g);
	}
	//x+radiusがwidth-1になるまで加減算する
	//x = width - 1 - radius
	int bound = width - 1 - radius;
	for (; x <= bound; x++)
	{
		g = tex2D<int>(texG, float(x + radius) + 0.5f, float(y) + 0.5f);
		_g = tex2D<int>(texG, float(x - radius - 1) + 0.5f, float(y) + 0.5f);
		sumg += g - _g;
		sumgg += g * g - _g * _g;
		*((int4*)((char*)sumG + y * pitchI4) + x) = make_int4(sumg, sumgg, pixNum, g);
	}
	//x= ~width-1
	for (; x < width; x++)
	{
		_g = tex2D<int>(texG, float(x - radius - 1) + 0.5f, float(y) + 0.5f);
		sumg -= _g;
		sumgg -= _g * _g;
		pixNum--;
		*((int4*)((char*)sumG + y * pitchI4) + x) = make_int4(sumg, sumgg, pixNum, g);
	}
}

__global__ void
de_gsum_y(int width, int height, int radius, int4* sumG, cudaTextureObject_t texSumG, size_t pitchI4)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	int4 tmp, _tmp;
	int sumg = 0.0f;
	int sumgg = 0.0f;
	int pixNum = 0;

	//y = 0
	for (int y = 0; y <= radius; y++)
	{
		tmp = tex2D<int4>(texSumG, float(x) + 0.5f, float(y) + 0.5f);
		sumg += tmp.x;
		sumgg += tmp.y;
		pixNum += tmp.z;
	}
	*((int4*)((char*)sumG) + x) = make_int4(sumg, sumgg, pixNum, sumg / pixNum);

	int y = 1;
	//y=1~radius
	for (; y <= radius; y++)
	{
		tmp = tex2D<int4>(texSumG, float(x) + 0.5f, float(y + radius) + 0.5f);
		sumg += tmp.x;
		sumgg += tmp.y;
		pixNum += tmp.z;
		*((int4*)((char*)sumG + y * pitchI4) + x) = make_int4(sumg, sumgg, pixNum, sumg / pixNum);
	}
	int bound = height - 1 - radius;
	for (; y < bound; y++)
	{
		tmp = tex2D<int4>(texSumG, float(x) + 0.5f, float(y + radius) + 0.5f);
		_tmp = tex2D<int4>(texSumG, float(x) + 0.5f, float(y - radius - 1) + 0.5f);
		sumg += tmp.x - _tmp.x;
		sumgg += tmp.y - _tmp.y;
		*((int4*)((char*)sumG + y * pitchI4) + x) = make_int4(sumg, sumgg, pixNum, sumg / pixNum);
	}
	for (; y < height; y++)
	{
		_tmp = tex2D<int4>(texSumG, float(x) + 0.5f, float(y - radius - 1) + 0.5f);
		sumg -= _tmp.x;
		sumgg -= _tmp.y;
		pixNum -= tmp.z;
		*((int4*)((char*)sumG + y * pitchI4) + x) = make_int4(sumg, sumgg, pixNum, sumg / pixNum);
	}
}


__global__ void
de_calculateCxDx(int width, int height, int* G, int4* sumG, float eps2, float2* cxdx, size_t pitchI1, size_t pitchI4, size_t pitchF2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int4 tmp = *((int4*)((char*)sumG + y * pitchI4) + x);
	int g = *((int*)((char*)G + y * pitchI1) + x);
	float pixel_sum_window_inv = 1.0f / (float)tmp.z;
	float g_ave = ((float)tmp.x) * pixel_sum_window_inv;
	float gg_ave = ((float)tmp.y) * pixel_sum_window_inv;
	float vx = gg_ave - g_ave * g_ave + eps2;
	float tmp2 = ((float)g) - g_ave;
	float cx2 = tmp2 * pixel_sum_window_inv / vx;
	*((float2*)((char*)cxdx + y * pitchF2) + x) = make_float2(cx2, pixel_sum_window_inv - g_ave * cx2);
}


#endif

}