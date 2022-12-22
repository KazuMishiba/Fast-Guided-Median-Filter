#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utility.h"
#include <stdio.h>


void cu_print_xy(int* src, int x, int y, SizeInfo& sizeInfo);
void cu_print_xy(float* src, int x, int y, SizeInfo& sizeInfo);


void cu_initializeWithValue(int* dst, int val, SizeInfo& sizeInfo, cudaStream_t stream = NULL);
void cu_initializeWithValue(float* dst, float val, SizeInfo& sizeInfo, cudaStream_t stream = NULL);
