#include "Utility.h"




//GPUメモリの確認
void cu_memoryInfo()
{
	size_t mf, ma;
	cudaMemGetInfo(&mf, &ma);
	cudaDeviceSynchronize();
	std::cout << "[GPU Memory] free: " << mf << " total: " << ma << std::endl << std::endl;
}

SizeInfo::SizeInfo()
{
}
SizeInfo::SizeInfo(int width, int height)
{
	dim3 blockSize = dim3(32, 32, 1);
	this->setInfo(width, height, 256, blockSize);
}
SizeInfo::SizeInfo(int width, int height, int range, dim3 blockSize)
{
	this->setInfo(width, height, range, blockSize);
}
void SizeInfo::setInfo(int width, int height, int range, dim3 blockSize)
{
	this->width = width;
	this->height = height;
	this->range = range;
	this->blockSize = dim3(blockSize);

	int gridSizeX = ceil(width / (float)this->blockSize.x);
	int gridSizeY = ceil(height / (float)this->blockSize.y);
	int gridSizeZ = 1;
	this->gridSize = dim3(gridSizeX, gridSizeY, gridSizeZ);

	//デバイスチェック
	//cudaDeviceProp deviceProp = cu_CudaDeviceInit();
	//ピッチの取得
	//デバイスにリニアメモリをcudaMallocPitchで確保(float4)
	float4* dst_f4;
	gpuErrchk(cudaMallocPitch(&dst_f4, &this->pitchF4, this->width * sizeof(float4), this->height));
	cudaFree(dst_f4);
	//デバイスにリニアメモリをcudaMallocPitchで確保(float1)
	float* dst_f1;
	gpuErrchk(cudaMallocPitch(&dst_f1, &this->pitchF1, this->width * sizeof(float), this->height));
	cudaFree(dst_f1);
	//float2
	float2* dst_f2;
	gpuErrchk(cudaMallocPitch(&dst_f2, &this->pitchF2, this->width * sizeof(float2), this->height));
	cudaFree(dst_f2);
	//unsigned char*
	unsigned char* dst_uc1;
	gpuErrchk(cudaMallocPitch(&dst_uc1, &this->pitchUC1, this->width * sizeof(unsigned char), this->height));
	cudaFree(dst_uc1);
	//uchar3*
	uchar3* dst_uc3;
	gpuErrchk(cudaMallocPitch(&dst_uc3, &this->pitchUC3, this->width * sizeof(uchar3), this->height));
	cudaFree(dst_uc3);
	//int*
	int* dst_i1;
	gpuErrchk(cudaMallocPitch(&dst_i1, &this->pitchI1, this->width * sizeof(int), this->height));
	cudaFree(dst_i1);
	//int2*
	int2* dst_i2;
	gpuErrchk(cudaMallocPitch(&dst_i2, &this->pitchI2, this->width * sizeof(int2), this->height));
	cudaFree(dst_i2);
	//int4*
	int4* dst_i4;
	gpuErrchk(cudaMallocPitch(&dst_i4, &this->pitchI4, this->width * sizeof(int4), this->height));
	cudaFree(dst_i4);
}

