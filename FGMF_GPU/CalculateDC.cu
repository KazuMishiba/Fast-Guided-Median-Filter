#include "CalculateDC.cuh"

namespace FGMF_GPU_Or
{

///////////////////////////////////////////
//Calculate sum in the window of guide image G

//For Grayscale
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

//For color
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


//for multichannel
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


//For Grayscale
void cu_calculateSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* sumG, int2* temp)
{
	int blockSize = sizeInfo.blockSize_.x;
	int gridSizeY = ceil(sizeInfo.height_ / (float)sizeInfo.blockSize_.y);
	int gridSizeX = ceil(sizeInfo.width_ / (float)sizeInfo.blockSize_.x);

	cudaTextureObject_t texG;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	Helper::UtilityForCUDA::setLinearArrayToTexture(G, texG, sizeInfo, filterMode);
	de_gsum_x << < gridSizeY, blockSize, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, temp, texG, sizeInfo.pitch<int2>());
	Helper::UtilityForCUDA::setLinearArrayToTexture(temp, texG, sizeInfo, filterMode);
	de_gsum_y << < gridSizeX, blockSize, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, sumG, texG, sizeInfo.pitch<int2>());

	cudaDestroyTextureObject(texG);
}
//For color
void cu_calculateSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G3, int radius, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG, Helper::DeviceArray<int>* tempG, Helper::DeviceArray<int>* tempGG)
{
	int blockSize = sizeInfo.blockSize_.x;
	int gridSizeY = ceil(sizeInfo.height_ / (float)sizeInfo.blockSize_.y);
	int gridSizeX = ceil(sizeInfo.width_ / (float)sizeInfo.blockSize_.x);

	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	Helper::TextureArray<int>* texG = new Helper::TextureArray<int>(G3, filterMode, sizeInfo);
	de_g3sum_x << < gridSizeY, blockSize, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, tempG->device, tempGG->device, texG->device, sizeInfo.pitch<int>());
	Helper::TextureArray<int>* texSumG = new Helper::TextureArray<int>(tempG, filterMode, sizeInfo);
	Helper::TextureArray<int>* texSumGG = new Helper::TextureArray<int>(tempGG, filterMode, sizeInfo);
	de_g3sum_y << < gridSizeX, blockSize, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, sumG->device, sumGG->device, texSumG->device, texSumGG->device, sizeInfo.pitch<int>());

	delete texG;
	delete texSumG;
	delete texSumGG;
}
//for multichannel
void cu_calculateSumGX(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* GX, int radius, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG, Helper::DeviceArray<int>* tempG, Helper::DeviceArray<int>* tempGG, int n)
{
	int blockSize = sizeInfo.blockSize_.x;
	int gridSizeY = ceil(sizeInfo.height_ / (float)sizeInfo.blockSize_.y);
	int gridSizeX = ceil(sizeInfo.width_ / (float)sizeInfo.blockSize_.x);

	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	Helper::TextureArray<int>* texG = new Helper::TextureArray<int>(GX, filterMode, sizeInfo);
	de_gXsum_x << < gridSizeY, blockSize, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, tempG->device, tempGG->device, texG->device, sizeInfo.pitch<int>(), n);
	Helper::TextureArray<int>* texSumG = new Helper::TextureArray<int>(tempG, filterMode, sizeInfo);
	Helper::TextureArray<int>* texSumGG = new Helper::TextureArray<int>(tempGG, filterMode, sizeInfo);
	de_gXsum_y << < gridSizeX, blockSize, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, radius, sumG->device, sumGG->device, texSumG->device, texSumGG->device, sizeInfo.pitch<int>(), n);

	delete texG;
	delete texSumG;
	delete texSumGG;
}




///////////////////////////////////////////
// Calculate d and c

//For Grayscale
__global__ void
de_calculateDC(int width, int height, int* G, int2* sumG, float eps2, float pixel_sum_window_inv, float2* dc, size_t pitchI1, size_t pitchI2, size_t pitchF2)
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
	float cx = tmp2 * pixel_sum_window_inv / vx;
	*((float2*)((char*)dc + y * pitchF2) + x) = make_float2(pixel_sum_window_inv - g_ave * cx, cx);
}
//For color
__global__ void
de_calculateDC3(int width, int height, int** G3, int** sumG, int** sumGG, float eps2, float pixel_sum_window_inv, float4* dc, size_t pitchI1, size_t pitchF4)
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
	if (abs(delta) > 1e-6)
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

		*((float4*)((char*)dc + y * pitchF4) + x) = make_float4(dx, cx1, cx2, cx3);
	}
	else
	{
		//Non-invertible
		*((float4*)((char*)dc + y * pitchF4) + x) = make_float4(pixel_sum_window_inv, 0.0f, 0.0f, 0.0f);
	}
}



//for multichannel
__global__ void
de_calculateDCx(int width, int height, int** GX, int** sumG, int** sumGG, float eps2, float pixel_sum_window_inv, float** dc, size_t pitchI1, size_t pitchF1, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

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

	//conjugate gradient method
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
		
		flag = flag && (abs(r[i]) <= 1e-10);
		
	}

	if (!flag)
	{
		for (int iter = 0; iter < n; iter++)
		{
			alpha = 0.0f;
			int t = 0;
			for (int j = 0; j < n; j++)
			{
				Ap[j] = A[t] * p[j];
				int m = j;
				int d = n - 1;
				for (int i = 0; i < j; i++)
				{
					Ap[j] += A[m] * p[i];
					m += d;
					d--;
				}
				for (int i = j + 1; i < n; i++)
				{
					t++;
					Ap[j] += A[t] * p[i];
				}
				t++;
				alpha += p[j] * Ap[j];
			}

			alpha = rsold / alpha;
			rsnew = 0.0f;
			for (int i = 0; i < n; i++)
			{
				cx[i] += alpha * p[i];
				r[i] -= alpha * Ap[i];
				rsnew += r[i] * r[i];
			}
			if (rsnew < 1e-15)
			{
				break;
			}
			float no = rsnew / rsold;
			for (int i = 0; i < n; i++)
			{
				p[i] = r[i] + no * p[i];
			}
			rsold = rsnew;
			
		}
		
		float dx = pixel_sum_window_inv;
		for (int i = 0; i < n; i++)
		{
			cx[i] *= pixel_sum_window_inv;
			dx -= cx[i] * g_ave[i];
			*((float*)((char*)dc[i+1] + y * pitchF1) + x) = cx[i];
		}
		*((float*)((char*)dc[0] + y * pitchF1) + x) = dx;
		
	}
	else
	{
		//Non-invertible
		for (int i = 0; i < n; i++)
		{
			*((float*)((char*)dc[i+1] + y * pitchF1) + x) = 0.0f;
		}
		*((float*)((char*)dc[0] + y * pitchF1) + x) = pixel_sum_window_inv;
		
	}

	delete g_ave;
	delete cx;
	delete A;
	delete r;
	delete p;
	delete Ap;
}




//For Grayscale
void cu_calculateDC(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G, int radius, int pixelNumInWindow, float eps2, float2* dc)
{
	int2* sumG, * temp;
	Helper::UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo);
	Helper::UtilityForCUDA::allocateDeviceMemory(temp, sizeInfo);

	cu_calculateSumG(sizeInfo, stream, G->host[0], radius, sumG, temp);
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	de_calculateDC << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, G->host[0], sumG, eps2, pixel_sum_window_inv, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float2>());

	cudaFree(sumG);
	cudaFree(temp);
}

//For color
void cu_calculateDC(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G, int radius, int pixelNumInWindow, float eps2, float4* dc)
{
	Helper::DeviceArray<int>* sumG = new Helper::DeviceArray<int>(3, sizeInfo);
	Helper::DeviceArray<int>* tempG = new Helper::DeviceArray<int>(3, sizeInfo);
	Helper::DeviceArray<int>* sumGG = new Helper::DeviceArray<int>(6, sizeInfo);
	Helper::DeviceArray<int>* tempGG = new Helper::DeviceArray<int>(6, sizeInfo);

	cu_calculateSumG3(sizeInfo, stream, G, radius, sumG, sumGG, tempG, tempGG);
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	de_calculateDC3 << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, G->device, sumG->device, sumGG->device, eps2, pixel_sum_window_inv, dc, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());

	delete sumG;
	delete tempG;
	delete sumGG;
	delete tempGG;
}

//for multichannel
void cu_calculateDCx(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G, int radius, int pixelNumInWindow, float eps2, Helper::DeviceArray<float>* dc)
{
	int n = G->arrayLength;
	Helper::DeviceArray<int>* sumG = new Helper::DeviceArray<int>(n, sizeInfo);
	Helper::DeviceArray<int>* tempG = new Helper::DeviceArray<int>(n, sizeInfo);
	Helper::DeviceArray<int>* sumGG = new Helper::DeviceArray<int>((n + 1) * n / 2, sizeInfo);
	Helper::DeviceArray<int>* tempGG = new Helper::DeviceArray<int>((n + 1) * n / 2, sizeInfo);

	cu_calculateSumGX(sizeInfo, stream, G, radius, sumG, sumGG, tempG, tempGG, n);

	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	de_calculateDCx << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, G->device, sumG->device, sumGG->device, eps2, pixel_sum_window_inv, dc->device, sizeInfo.pitch<int>(), sizeInfo.pitch<float>(), n);
	delete sumG;
	delete tempG;
	delete sumGG;
	delete tempGG;
}

//////////////////////////////////////////////////////////
// For multidimensional data

//For Grayscale
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
//For color
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
//For Grayscale
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
//For color
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

//For Grayscale
void cu_addSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int2* addSumG, int2* sumG)
{
	de_addSumG << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, addSumG, sumG, sizeInfo.pitch<int2>());
}
//For color
void cu_addSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* addSumG, Helper::DeviceArray<int>* addSumGG, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG)
{
	de_addSumG3 << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, addSumG->device, addSumGG->device, sumG->device, sumGG->device, sizeInfo.pitch<int>());
}

//For Grayscale
void cu_remSumG(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int2* remSumG, int2* sumG)
{
	de_remSumG << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, remSumG, sumG, sizeInfo.pitch<int2>());
}
//For color
void cu_remSumG3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* remSumG, Helper::DeviceArray<int>* remSumGG, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG)
{
	de_remSumG3 << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, remSumG->device, remSumGG->device, sumG->device, sumGG->device, sizeInfo.pitch<int>());
}

//For Grayscale
void cu_calculateDC(Helper::SizeInfo& sizeInfo, cudaStream_t stream, int* G, int pixelNumInWindow, float eps2, float2* cxdx, int2* sumG)
{
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	de_calculateDC << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, G, sumG, eps2, pixel_sum_window_inv, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float2>());
}
//For color
void cu_calculateDC3(Helper::SizeInfo& sizeInfo, cudaStream_t stream, Helper::DeviceArray<int>* G, int pixelNumInWindow, float eps2, float4* cxdx, Helper::DeviceArray<int>* sumG, Helper::DeviceArray<int>* sumGG)
{
	float pixel_sum_window_inv = 1.0f / pixelNumInWindow;
	de_calculateDC3 << <sizeInfo.gridSize_, sizeInfo.blockSize_, 0, stream >> > (sizeInfo.width_, sizeInfo.height_, G->device, sumG->device, sumGG->device, eps2, pixel_sum_window_inv, cxdx, sizeInfo.pitch<int>(), sizeInfo.pitch<float4>());
}

}