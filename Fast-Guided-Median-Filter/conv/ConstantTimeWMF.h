#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include<opencv2/opencv.hpp>
#include <cmath>

#include "Utility.h"
#include "ConstantTimeWMF.cuh"

class ConstantTimeWMF
{
public:

	template<typename ITYPE>
	static void filter2DGPU(ITYPE*& I, float*& G, ITYPE*& result, int radius, float eps2, int Imax, SizeInfo& sizeInfo);
	//I1G3, I3G3 template GTYPE = int or DeviceArray<int>
	template<typename ITYPE>
	static void filter2DGPU(ITYPE*& I, DeviceArray<float>*& G, ITYPE*& result, int radius, float eps2, int Imax, SizeInfo& sizeInfo);

};


//////////////////////////////////////////////////////////////////////
//2D
//I1G1, I3G1 ITYPE = int or DeviceArray<int>
template<typename ITYPE>
static void ConstantTimeWMF::filter2DGPU(ITYPE*& I, float*& G, ITYPE*& result, int radius, float eps2, int Imax, SizeInfo& sizeInfo)
{
	cu_ConstantTimeWMF(sizeInfo, NULL, radius, Imax, result, I, G, eps2);
}


//I1G3, I3G3 ITYPE = int or DeviceArray<int>
template<typename ITYPE>
static void ConstantTimeWMF::filter2DGPU(ITYPE*& I, DeviceArray<float>*& G, ITYPE*& result, int radius, float eps2, int Imax, SizeInfo& sizeInfo)
{
	cu_ConstantTimeWMF(sizeInfo, NULL, radius, Imax, result, I, G, eps2);
}

