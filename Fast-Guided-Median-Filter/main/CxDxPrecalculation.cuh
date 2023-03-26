#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "boxFilter.cuh"
#include "Utility.h"

#include "common.h"

//2D
void cu_calculateCxDxFromG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, float2* cxdx, int2* sumG, int2* temp);
void cu_calculateCx3DxFromG(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G3, int radius, int pixelNumInWindow, float eps2, float4* cxdx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);
void cu_calculateCxXDxFromG(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* GX, int radius, int pixelNumInWindow, float eps2, DeviceArray<float>* cxdx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);

void cu_calculateSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G3, int radius, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);
void cu_updateSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, DeviceArray<int>* addSumG, DeviceArray<int>* addSumGG, DeviceArray<int>* remSumG, DeviceArray<int>* remSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);
void cu_addSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, DeviceArray<int>* addSumG, DeviceArray<int>* addSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);
void cu_remSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* remSumG, DeviceArray<int>* remSumGG, DeviceArray<int>* sumG, DeviceArray<int>* sumGG);
void cu_calculateCx3Dx(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G, int radius, int pixelNumInWindow, float eps2, float4* cxdx, DeviceArray<int>* sumG, DeviceArray<int>* sumGG);

//‘½ŽŸŒ³—p
void cu_calculateSumG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* sumG, int2* temp);
void cu_updateSumG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* addSumG, int2* remSumG, int2* sumG, int2* temp);
void cu_addSumG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int2* addSumG, int2* sumG, int2* temp);
void cu_remSumG(SizeInfo& sizeInfo, cudaStream_t stream, int2* remSumG, int2* sumG);
void cu_calculateCxDx(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, float2* cxdx, int2* sumG);




void cu_calculateSumG3(SizeInfo& sizeInfo, cudaStream_t stream, DeviceArray<int>* G3, int radius, DeviceArray<int>* sumG, DeviceArray<int>* sumGG, DeviceArray<int>* tempG, DeviceArray<int>* tempGG);



//refactoring—p
void cu_calculateCxDxFromG(SizeInfo& sizeInfo, cudaStream_t stream, int* G, int radius, int pixelNumInWindow, float eps2, DeviceArray<float>* cxdx, int2* sumG, int2* temp);

