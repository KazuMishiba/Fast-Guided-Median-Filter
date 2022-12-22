#pragma once
#include<opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <immintrin.h>
#include <windows.h>
#include "Utility.h"
#include "l1solver.h"
#include "FGMF.h"
#include "FGMF_type1.h"
#include "FGMF_type2.h"
#include "FGMF_type3.h"
#include "GuidedFilter.h"
//#include "GMF_GPU.h"
#include "Experimenter.h"
#include <chrono>
#include "DataContainer.h"
#include "common.h"

class Method
{
public:
	Container_Image* image;


	Method();
	void test();
	void test2();
	void multiTest();

	void speedTest();
	void speedTest2();

};

