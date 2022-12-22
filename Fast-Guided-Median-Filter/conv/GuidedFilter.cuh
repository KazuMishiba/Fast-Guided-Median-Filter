#pragma once


#include<opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utility.h"
#include "CxDxPrecalculation.cuh"


void cu_filterSimplified(SizeInfo& sizeInfo, cudaStream_t stream, int radius, float eps2, int* result, int* f, int* g, float2* cxdx, int2* sumFG, int2* sumG, int2* temp);

