#pragma once
#include "Utility.h"
//void cu_meanFiltering(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* src, float* dst, float* temp, int radius, size_t pitch);

void cu_boxFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float* srcTex, float* dst, float* tmp, int radius, cudaStream_t stream, cudaTextureObject_t tex, size_t pitch);
void cu_meanFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float* srcTex, float* dst, float* tmp, int radius, cudaStream_t stream, cudaTextureObject_t tex, size_t pitch);
void cu_mean2d(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* src, float* dst, int r, size_t pitchF1);
void cu_boxFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float4* srcTex, float4* dst, float4* tmp, int radius, cudaStream_t stream, cudaTextureObject_t tex, size_t pitchF4);
void cu_meanFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float4* srcTex, float4* dst, float4* tmp, int radius, cudaStream_t stream, cudaTextureObject_t tex, size_t pitchF4);

void cu_meanFiltering(int blockSize, int gridSizeX, int gridSizeY, int width, int height, float* src, float* dst, float* tmp1, float*tmp2, int radius, cudaStream_t stream, size_t pitch, SizeInfo sizeInfo);
