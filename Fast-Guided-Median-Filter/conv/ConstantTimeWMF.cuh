#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "boxFilter.cuh"

#include <stdio.h>
//#include "boxFilter.cuh"
#include "Utility.h"
#include "Utility.cuh"
#include<opencv2/opencv.hpp>
#include <vector>

void cu_ConstantTimeWMF(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, float* g, float eps2);
void cu_ConstantTimeWMF(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, float* g, float eps2);
void cu_ConstantTimeWMF(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, DeviceArray<float>* g, float eps2);
void cu_ConstantTimeWMF(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<float>* g, float eps2);