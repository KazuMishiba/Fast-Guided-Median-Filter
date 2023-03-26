#include "FGMF_GPU_Or_ver0.h"

namespace FGMF_GPU_Or_ver0
{
	/*
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			cu_memoryInfo();
			if (abort) exit(code);
		}
	}*/

	void sizeinfoTest()
	{

		int width, height;
		size_t pitchF4, pitchF1, pitchF2, pitchUC1, pitchUC3, pitchI1, pitchI2, pitchI4;

		width = 98;
		height = 201;



		float4* dst_f4;
		gpuErrchk(cudaMallocPitch(&dst_f4, &pitchF4, width * sizeof(float4), height));
		cudaFree(dst_f4);
		//デバイスにリニアメモリをcudaMallocPitchで確保(float1)
		float* dst_f1;
		gpuErrchk(cudaMallocPitch(&dst_f1, &pitchF1, width * sizeof(float), height));
		cudaFree(dst_f1);
		//float2
		float2* dst_f2;
		gpuErrchk(cudaMallocPitch(&dst_f2, &pitchF2, width * sizeof(float2), height));
		cudaFree(dst_f2);
		//unsigned char*
		unsigned char* dst_uc1;
		gpuErrchk(cudaMallocPitch(&dst_uc1, &pitchUC1, width * sizeof(unsigned char), height));
		cudaFree(dst_uc1);
		//uchar3*
		uchar3* dst_uc3;
		gpuErrchk(cudaMallocPitch(&dst_uc3, &pitchUC3, width * sizeof(uchar3), height));
		cudaFree(dst_uc3);
		//int*
		int* dst_i1;
		gpuErrchk(cudaMallocPitch(&dst_i1, &pitchI1, width * sizeof(int), height));
		cudaFree(dst_i1);
		//int2*
		int2* dst_i2;
		gpuErrchk(cudaMallocPitch(&dst_i2, &pitchI2, width * sizeof(int2), height));
		cudaFree(dst_i2);
		//int4*
		int4* dst_i4;
		gpuErrchk(cudaMallocPitch(&dst_i4, &pitchI4, width * sizeof(int4), height));
		cudaFree(dst_i4);


		std::cout << "F4:" << pitchF4 << std::endl;




	}
}
