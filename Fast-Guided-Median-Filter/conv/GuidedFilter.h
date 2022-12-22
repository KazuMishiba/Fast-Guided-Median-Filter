#pragma once
#include "CxDxPrecalculation.cuh"
#include "GuidedFilter.cuh"

class GuidedFilter
{
public:
	//è]óà
	static void filterNaive(int*& I, int*& G, int*& result, int radius, float eps2, SizeInfo& sizeInfo);

	//simplified
	static void filterSimplified(int*& I, int*& G, int*& result, int radius, float eps2, SizeInfo& sizeInfo);

};

