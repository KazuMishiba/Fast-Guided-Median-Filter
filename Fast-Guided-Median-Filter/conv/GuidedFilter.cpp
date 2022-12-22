#include "GuidedFilter.h"

void GuidedFilter::filterNaive(int *& I, int *& G, int *& result, int radius, float eps2, SizeInfo & sizeInfo)
{

}

void GuidedFilter::filterSimplified(int *& I, int *& G, int *& result, int radius, float eps2, SizeInfo & sizeInfo)
{
	float2* cxdx;
	int2* sumG, *temp, * sumFG;
	UtilityForCUDA::allocateDeviceMemory(cxdx, sizeInfo);
	UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo);
	UtilityForCUDA::allocateDeviceMemory(temp, sizeInfo);
	UtilityForCUDA::allocateDeviceMemory(sumFG, sizeInfo);


	cu_filterSimplified(sizeInfo, NULL, radius, eps2, result, I, G, cxdx, sumFG, sumG, temp);


	//ÉÅÉÇÉääJï˙
	cudaFree(cxdx);
	cudaFree(sumG);
	cudaFree(temp);
	cudaFree(sumFG);

}
