#include "GuidedFilter.cuh"



__global__ void
de_fgsum_x(int width, int height, int radius, int2* sumFG, cudaTextureObject_t texF, cudaTextureObject_t texG, size_t pitchI2)
{
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height)
		return;

	int g;
	int _g;
	int f;
	int _f;
	int2 sumfg = make_int2(0,0);//fg, f

	//x=0
	for (int x = -radius; x <= radius; x++)
	{
		f = tex2D<int>(texF, x, y);
		g = tex2D<int>(texG, x, y);
		sumfg.x += f*g;
		sumfg.y += f;
	}
	*((int2*)((char*)sumFG + y * pitchI2)) = sumfg;

	for (int x = 1; x < width; x++)
	{
		f = tex2D<int>(texF, x + radius, y);
		_f = tex2D<int>(texF, x - radius - 1, y);
		g = tex2D<int>(texG, x + radius, y);
		_g = tex2D<int>(texG, x - radius - 1, y);
		sumfg.x += f * g - _f * _g;
		sumfg.y += f - _f;
		*((int2*)((char*)sumFG + y * pitchI2) + x) = sumfg;
	}
}
__global__ void
de_fgsum_y(int width, int height, int radius, int2* sumFG, cudaTextureObject_t texSumFG, size_t pitchI2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= width)
		return;

	int2 tmp, _tmp;
	int2 sumfg = make_int2(0,0);

	//y = 0
	for (int y = -radius; y <= radius; y++)
	{
		tmp = tex2D<int2>(texSumFG, x, y);
		sumfg.x += tmp.x;
		sumfg.y += tmp.y;
	}
	*((int2*)((char*)sumFG) + x) = sumfg;

	for (int y = 1; y < height; y++)
	{
		tmp = tex2D<int2>(texSumFG, x, y + radius);
		_tmp = tex2D<int2>(texSumFG, x, y - radius - 1);
		sumfg.x += tmp.x - _tmp.x;
		sumfg.y += tmp.y - _tmp.y;
		*((int2*)((char*)sumFG + y * pitchI2) + x) = sumfg;
	}
}

__global__ void
de_filterNaive(int width, int height, int* result, float2* cxdx, int2* sumFG, size_t pitchI1, size_t pitchI2, size_t pitchF2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float2 CxDx = *((float2*)((char*)cxdx + y * pitchF2) + x);
	int2 sumfg = *((int2*)((char*)sumFG + y * pitchI2) + x);

	*((int*)((char*)result + y * pitchI1) + x) = CxDx.x * sumfg.x + CxDx.y * sumfg.y;
}

void cu_calculateFG(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int* f, int* g, int2* sumFG, int2* temp)
{
	int blockSize = BLOCK_SIZE_1D;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint;
	cudaTextureObject_t texF, texG;
	UtilityForCUDA::setLinearArrayToTexture(f, texF, sizeInfo, filterMode);//texFにfをバインド
	UtilityForCUDA::setLinearArrayToTexture(g, texG, sizeInfo, filterMode);//texGにgをバインド

	de_fgsum_x << < gridSizeY, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, temp, texF, texG, sizeInfo.pitch<int2>());

	UtilityForCUDA::setLinearArrayToTexture(temp, texG, sizeInfo, filterMode);
	de_fgsum_y << < gridSizeX, blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, radius, sumFG, texG, sizeInfo.pitch<int2>());

	cudaDestroyTextureObject(texF);
	cudaDestroyTextureObject(texG);
}

//I1G1
void cu_filterSimplified(SizeInfo& sizeInfo, cudaStream_t stream, int radius, float eps2, int* result, int* f, int* g, float2* cxdx, int2* sumFG, int2* sumG, int2* temp)
{
	int pixelNumInWindow = (radius * 2 + 1) * (radius * 2 + 1);
	//cxdx
	cu_calculateCxDxFromG(sizeInfo, stream, g, radius, pixelNumInWindow, eps2, cxdx, sumG, temp);
	//fg
	cu_calculateFG(sizeInfo,  stream,radius, f, g, sumFG, temp);
	//result
	de_filterNaive <<< sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, result, cxdx, sumFG, sizeInfo.pitch<int>(), sizeInfo.pitch<int2>(), sizeInfo.pitch<float2>());

}