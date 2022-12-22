#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "boxFilter.cuh"
#include "Utility.h"


__global__ void
de_meanfilter_x_tex(int width, int height, int radius, float* dst, cudaTextureObject_t tex, size_t pitch)
{
	float scale = 1.0f / (float)((radius << 1) + 1);
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	float t = 0.0f;

	for (int x = -radius; x <= radius; x++)
	{
		t += tex2D<float>(tex, float(x) + 0.5f, float(y) + 0.5f);
	}

	*((float*)((char*)dst + y * pitch)) = t * scale;

	for (int x = 1; x < width; x++)
	{
		t += tex2D<float>(tex, float(x + radius) + 0.5f, float(y) + 0.5f);
		t -= tex2D<float>(tex, float(x - radius - 1) + 0.5f, float(y) + 0.5f);
		*((float*)((char*)dst + y * pitch) + x) = t * scale;
	}
}

__global__ void
de_meanfilter_y_tex(int width, int height, int radius, float* dst, cudaTextureObject_t tex, size_t pitch)
{
	float scale = 1.0f / (float)((radius << 1) + 1);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	float t = 0.0f;

	for (int y = -radius; y <= radius; y++)
	{
		t += tex2D<float>(tex, float(x) + 0.5f, float(y) + 0.5f);
	}

	*((float*)((char*)dst) + x) = t * scale;

	for (int y = 1; y < height; y++)
	{
		t += tex2D<float>(tex, float(x) + 0.5f, float(y + radius) + 0.5f);
		t -= tex2D<float>(tex, float(x) + 0.5f, float(y - radius - 1) + 0.5f);
		*((float*)((char*)dst + y * pitch) + x) = t * scale;
	}
}


__global__ void
de_boxfilter_x_tex(int width, int height, int radius, float* dst, cudaTextureObject_t tex, size_t pitch)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	float t = 0.0f;

	for (int x = -radius; x <= radius; x++)
	{
		t += tex2D<float>(tex, float(x) + 0.5f, float(y) + 0.5f);
	}

	*((float*)((char*)dst + y * pitch)) = t;

	for (int x = 1; x < width; x++)
	{
		t += tex2D<float>(tex, float(x + radius) + 0.5f, float(y) + 0.5f);
		t -= tex2D<float>(tex, float(x - radius - 1) + 0.5f, float(y) + 0.5f);
		*((float*)((char*)dst + y * pitch) + x) = t;
	}
}

__global__ void
de_boxfilter_y_tex(int width, int height, int radius, float* dst, cudaTextureObject_t tex, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	float t = 0.0f;

	for (int y = -radius; y <= radius; y++)
	{
		t += tex2D<float>(tex, float(x) + 0.5f, float(y) + 0.5f);
	}

	*((float*)((char*)dst) + x) = t;

	for (int y = 1; y < height; y++)
	{
		t += tex2D<float>(tex, float(x) + 0.5f, float(y + radius) + 0.5f);
		t -= tex2D<float>(tex, float(x) + 0.5f, float(y - radius - 1) + 0.5f);
		*((float*)((char*)dst + y * pitch) + x) = t;
	}
}

//Box Filtering（総和） srcTexは入力をテクスチャとして登録して使うが、フィルタリング中に書き換わることに注意
void cu_boxFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float* srcTex, float* dst, float* tmp, int radius, cudaStream_t stream, cudaTextureObject_t tex, size_t pitch)
{
	de_boxfilter_x_tex << < blockSize, gridSizeY, 0, stream >> > (width, height, radius, tmp, tex, pitch);
	cudaMemcpy2DAsync(srcTex, pitch, tmp, pitch, width * sizeof(float), height, cudaMemcpyDefault, stream);
	de_boxfilter_y_tex << < blockSize, gridSizeX, 0, stream >> > (width, height, radius, dst, tex, pitch);
}


//Mean Filtering (平均値) srcTexは入力をテクスチャとして登録して使うが、フィルタリング中に書き換わることに注意
void cu_meanFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float* srcTex, float* dst, float* tmp, int radius, cudaStream_t stream, cudaTextureObject_t tex, size_t pitch)
{
	de_meanfilter_x_tex << < blockSize, gridSizeY, 0, stream >> > (width, height, radius, tmp, tex, pitch);
	cudaMemcpy2DAsync(srcTex, pitch, tmp, pitch, width * sizeof(float), height, cudaMemcpyDefault, stream);
	de_meanfilter_y_tex << < blockSize, gridSizeX, 0, stream >> > (width, height, radius, dst, tex, pitch);
}

//Mean Filtering (平均値) 上記の書き換わり回避版(関数内で入力をコピーする)
void cu_meanFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float* src, float* dst, float* tmp1, float*tmp2, int radius, cudaStream_t stream, size_t pitch, SizeInfo sizeInfo)
{
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
	cudaTextureObject_t tex;
	UtilityForCUDA::copyDeviceMemory(src, tmp1, sizeInfo, stream);
	UtilityForCUDA::setLinearArrayToTexture(tmp1, tex, sizeInfo, filterMode);
	de_meanfilter_x_tex << < blockSize, gridSizeY, 0, stream >> > (width, height, radius, tmp2, tex, pitch);
	cudaMemcpy2DAsync(tmp1, pitch, tmp2, pitch, width * sizeof(float), height, cudaMemcpyDefault, stream);
	de_meanfilter_y_tex << < blockSize, gridSizeX, 0, stream >> > (width, height, radius, dst, tex, pitch);
	cudaDestroyTextureObject(tex);
}




///////////////////////////////////
//カラー(float4)
__global__ void
de_meanfilter_x_tex(int width, int height, int radius, float4* dst, cudaTextureObject_t tex, size_t pitch)
{
	float scale = 1.0f / (float)((radius << 1) + 1);
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	float4 t = make_float4(0.0f,0.0f,0.0f,0.0f);

	for (int x = -radius; x <= radius; x++)
	{
		float4 s = tex2D<float4>(tex, float(x) + 0.5f, float(y) + 0.5f);
		t.x += s.x;
		t.y += s.y;
		t.z += s.z;
	}

	*((float4*)((char*)dst + y * pitch)) = make_float4(t.x * scale, t.y * scale, t.z * scale, 0.0f);

	for (int x = 1; x < width; x++)
	{
		float4 s1 = tex2D<float4>(tex, float(x + radius) + 0.5f, float(y) + 0.5f);
		float4 s2 = tex2D<float4>(tex, float(x - radius - 1) + 0.5f, float(y) + 0.5f);
		t.x += s1.x;
		t.y += s1.y;
		t.z += s1.z;
		t.x -= s2.x;
		t.y -= s2.y;
		t.z -= s2.z;

		*((float4*)((char*)dst + y * pitch) + x) = make_float4(t.x * scale, t.y * scale, t.z * scale, 0.0f);
	}
}

__global__ void
de_meanfilter_y_tex(int width, int height, int radius, float4* dst, cudaTextureObject_t tex, size_t pitch)
{
	float scale = 1.0f / (float)((radius << 1) + 1);
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	float4 t = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for (int y = -radius; y <= radius; y++)
	{
		float4 s = tex2D<float4>(tex, float(x) + 0.5f, float(y) + 0.5f);
		t.x += s.x;
		t.y += s.y;
		t.z += s.z;
	}

	*((float4*)((char*)dst) + x) = make_float4(t.x * scale, t.y * scale, t.z * scale, 0.0f);

	for (int y = 1; y < height; y++)
	{
		float4 s1 = tex2D<float4>(tex, float(x) + 0.5f, float(y + radius) + 0.5f);
		float4 s2 = tex2D<float4>(tex, float(x) + 0.5f, float(y - radius - 1) + 0.5f);
		t.x += s1.x;
		t.y += s1.y;
		t.z += s1.z;
		t.x -= s2.x;
		t.y -= s2.y;
		t.z -= s2.z;

		*((float4*)((char*)dst + y * pitch) + x) = make_float4(t.x * scale, t.y * scale, t.z * scale, 0.0f);
	}
}


__global__ void
de_boxfilter_x_tex(int width, int height, int radius, float4* dst, cudaTextureObject_t tex, size_t pitch)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	float4 t = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for (int x = -radius; x <= radius; x++)
	{
		float4 s = tex2D<float4>(tex, float(x) + 0.5f, float(y) + 0.5f);
		t.x += s.x;
		t.y += s.y;
		t.z += s.z;
	}

	*((float4*)((char*)dst + y * pitch)) = make_float4(t.x, t.y, t.z, 0.0f);

	for (int x = 1; x < width; x++)
	{
		float4 s1 = tex2D<float4>(tex, float(x + radius) + 0.5f, float(y) + 0.5f);
		float4 s2 = tex2D<float4>(tex, float(x - radius - 1) + 0.5f, float(y) + 0.5f);
		t.x += s1.x;
		t.y += s1.y;
		t.z += s1.z;
		t.x -= s2.x;
		t.y -= s2.y;
		t.z -= s2.z;

		*((float4*)((char*)dst + y * pitch) + x) = make_float4(t.x, t.y, t.z, 0.0f);
	}
}

__global__ void
de_boxfilter_y_tex(int width, int height, int radius, float4* dst, cudaTextureObject_t tex, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	float4 t = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	for (int y = -radius; y <= radius; y++)
	{
		float4 s = tex2D<float4>(tex, float(x) + 0.5f, float(y) + 0.5f);
		t.x += s.x;
		t.y += s.y;
		t.z += s.z;
	}

	*((float4*)((char*)dst) + x) = make_float4(t.x, t.y, t.z, 0.0f);

	for (int y = 1; y < height; y++)
	{
		float4 s1 = tex2D<float4>(tex, float(x) + 0.5f, float(y + radius) + 0.5f);
		float4 s2 = tex2D<float4>(tex, float(x) + 0.5f, float(y - radius - 1) + 0.5f);
		t.x += s1.x;
		t.y += s1.y;
		t.z += s1.z;
		t.x -= s2.x;
		t.y -= s2.y;
		t.z -= s2.z;

		*((float4*)((char*)dst + y * pitch) + x) = make_float4(t.x, t.y, t.z, 0.0f);
	}
}

//Box Filtering（総和） srcTexは入力をテクスチャとして登録して使うが、フィルタリング中に書き換わることに注意
void cu_boxFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float4* srcTex, float4* dst, float4* tmp, int radius, cudaStream_t stream, cudaTextureObject_t tex, size_t pitchF4)
{
	de_boxfilter_x_tex << < blockSize, gridSizeY, 0, stream >> > (width, height, radius, tmp, tex, pitchF4);
	cudaMemcpy2DAsync(srcTex, pitchF4, tmp, pitchF4, width * sizeof(float4), height, cudaMemcpyDefault, stream);
	de_boxfilter_y_tex << < blockSize, gridSizeX, 0, stream >> > (width, height, radius, dst, tex, pitchF4);
}


//Mean Filtering (平均値) srcTexは入力をテクスチャとして登録して使うが、フィルタリング中に書き換わることに注意
void cu_meanFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float4* srcTex, float4* dst, float4* tmp, int radius, cudaStream_t stream, cudaTextureObject_t tex, size_t pitchF4)
{
	de_meanfilter_x_tex << < blockSize, gridSizeY, 0, stream >> > (width, height, radius, tmp, tex, pitchF4);
	cudaMemcpy2DAsync(srcTex, pitchF4, tmp, pitchF4, width * sizeof(float), height, cudaMemcpyDefault, stream);
	de_meanfilter_y_tex << < blockSize, gridSizeX, 0, stream >> > (width, height, radius, dst, tex, pitchF4);
}




//////////////////////////////////
//遅い実装
__global__ void
de_mean2d(int width, int height, float* src, float* dst, int r, size_t pitchF1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int left = x - r > 0 ? x - r : 0;
	int right = x + r < width ? x + r : width - 1;
	int top = y - r > 0 ? y - r : 0;
	int bottom = y + r < height ? y + r : height - 1;

	int num = 0;
	float sum = 0.0f;
	for (int yy = top; yy <= bottom; yy++)
	{
		for (int xx = left; xx <= right; xx++)
		{
			sum += *((float*)((char*)src + yy * pitchF1) + xx);
			num++;
		}
	}
	*((float*)((char*)dst + y * pitchF1) + x) = sum / (float)num;
}

void cu_mean2d(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* src, float* dst, int r, size_t pitchF1)
{
	de_mean2d << <gridSize, blockSize, 0, stream >> > (width, height, src, dst, r, pitchF1);
}
