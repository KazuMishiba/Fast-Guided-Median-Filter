#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <curand_kernel.h>
#include <assert.h>
#include<opencv2/opencv.hpp>

namespace Helper
{
	//Cuda memory array
	template<class TYPE> class DeviceArray;

	////////////////////////////////////////////////////////////////////////////////
	//GPU memory check
	inline void cu_memoryInfo()
	{
		size_t mf, ma;
		cudaMemGetInfo(&mf, &ma);
		cudaDeviceSynchronize();
		std::cout << "[GPU Memory] free: " << mf << " total: " << ma << std::endl << std::endl;
	}
	//For error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			cu_memoryInfo();
			if (abort) exit(code);
		}
	}


	//Size information for CUDA
	class SizeInfo {
	public:
		int width_, height_, range_;
		dim3 blockSize_, gridSize_;
		SizeInfo(int width, int height, int range = 256, dim3 blockSize = dim3(8, 8, 1));

		template<typename TYPE>	inline size_t pitch();
		template<typename TYPE>	inline void setPitch(size_t pitch);

	private:
		size_t pitchF1, pitchI1, pitchF2, pitchI2, pitchI3, pitchI4, pitchF4, pitchUC1, pitchUC3;
	};

	inline SizeInfo::SizeInfo(int width, int height, int range , dim3 blockSize) :
		width_(width), height_(height), range_(range), blockSize_(blockSize), pitchF1(0), pitchI1(0), pitchF2(0), pitchI2(0), pitchI3(0), pitchI4(0), pitchF4(0), pitchUC1(0), pitchUC3(0)
	{
		int gridSizeX = ceil(width / (float)this->blockSize_.x);
		int gridSizeY = ceil(height / (float)this->blockSize_.y);
		int gridSizeZ = 1;
		this->gridSize_ = dim3(gridSizeX, gridSizeY, gridSizeZ);
	}

	template<> inline size_t SizeInfo::pitch<int>() { return this->pitchI1; }
	template<> inline size_t SizeInfo::pitch<int2>() { return this->pitchI2; }
	template<> inline size_t SizeInfo::pitch<int3>() { return this->pitchI3; }
	template<> inline size_t SizeInfo::pitch<int4>() { return this->pitchI4; }
	template<> inline size_t SizeInfo::pitch<float>() { return this->pitchF1; }
	template<> inline size_t SizeInfo::pitch<float2>() { return this->pitchF2; }
	template<> inline size_t SizeInfo::pitch<float4>() { return this->pitchF4; }
	template<> inline size_t SizeInfo::pitch<unsigned char>() { return this->pitchUC1; }
	template<> inline size_t SizeInfo::pitch<uchar3>() { return this->pitchUC3; }

	template<> inline void SizeInfo::setPitch<int>(size_t pitch) { this->pitchI1 = pitch; }
	template<> inline void SizeInfo::setPitch<int2>(size_t pitch) { this->pitchI2 = pitch; }
	template<> inline void SizeInfo::setPitch<int3>(size_t pitch) { this->pitchI3 = pitch; }
	template<> inline void SizeInfo::setPitch<int4>(size_t pitch) { this->pitchI4 = pitch; }
	template<> inline void SizeInfo::setPitch<float>(size_t pitch) { this->pitchF1 = pitch; }
	template<> inline void SizeInfo::setPitch<float2>(size_t pitch) { this->pitchF2 = pitch; }
	template<> inline void SizeInfo::setPitch<float4>(size_t pitch) { this->pitchF4 = pitch; }
	template<> inline void SizeInfo::setPitch<unsigned char>(size_t pitch) { this->pitchUC1 = pitch; }
	template<> inline void SizeInfo::setPitch<uchar3>(size_t pitch) { this->pitchUC3 = pitch; }


	////////////////////////////////////////////////////////////////////////////////
	//Utility
	class UtilityForCUDA
	{
	public:
		//Memory allocation
		template<typename TYPE>	static void allocateDeviceMemory(TYPE*& dst, SizeInfo& info);
		template<typename TYPE>	static void allocateDeviceMemory(TYPE*& dst, cv::Mat& initialValueMat, SizeInfo& info, cudaStream_t stream = NULL);
		template<typename TYPE>	static void allocateDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream = NULL);
		template<typename TYPE>	static void initializeDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream = NULL);
		//Memory copy
		template<typename TYPE>	static void copyDeviceMemory(TYPE*& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream = NULL);
		template<typename TYPE>	static void copyDeviceMemory(DeviceArray<TYPE>*& src, DeviceArray<TYPE>*& dst, int arrayLength, SizeInfo& info, cudaStream_t stream = NULL);
		//Upload cv::Mat to device memory
		template<typename TYPE>	static void uploadMatToDevice(cv::Mat& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream = NULL);
		//Download array from device memory to host memory as cv::Mat
		template<typename TYPE>	static void downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info, cv::Mat& dst);
		template<typename TYPE>	static cv::Mat downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info);
		template<typename TYPE>	static cv::Mat downloadLinearArrayAsMat(DeviceArray<TYPE>*& src, SizeInfo& info);
		template<typename TYPE> std::vector<cv::Mat> downloadLinearArrayAsMatVector(DeviceArray<TYPE>*& src, SizeInfo& info);
		//Set texture
		template<typename TYPE>	static void setLinearArrayToTexture(TYPE* src, cudaTextureObject_t& texObj, SizeInfo& info, cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModeLinear);

	private:
		template<typename TYPE>	inline static cudaChannelFormatDesc getCudaChannelFormatDesc();
		template<typename TYPE>	inline static int getCvType();
		template<typename TYPE>	inline static int getCvType(int channelNum);
	};



	////////////////////////////////////////////////////////////////////////////////
	//Memory allocation
	template<typename TYPE> void UtilityForCUDA::allocateDeviceMemory(TYPE*& dst, SizeInfo& info)
	{
		size_t pitch;
		gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width_ * sizeof(TYPE), info.height_));
		info.setPitch<TYPE>(pitch);
	}
	//Allocate memory and initialize with initialValueMat
	template<typename TYPE> void UtilityForCUDA::allocateDeviceMemory(TYPE*& dst, cv::Mat& initialValueMat, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch;
		gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width_ * sizeof(TYPE), info.height_));
		info.setPitch<TYPE>(pitch);
		gpuErrchk(cudaMemcpy2DAsync(dst, pitch, initialValueMat.data, initialValueMat.step, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault, stream));
	}
	//Allocate memory and initialize with 0
	template<typename TYPE> void UtilityForCUDA::allocateDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch;
		gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width_ * sizeof(TYPE), info.height_));
		info.setPitch<TYPE>(pitch);
		gpuErrchk(cudaMemset2DAsync(dst, pitch, 0, info.width_ * sizeof(TYPE), info.height_, stream));
	}
	//Initialize with 0
	template<typename TYPE> void UtilityForCUDA::initializeDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream)
	{
		gpuErrchk(cudaMemset2DAsync(dst, info.pitch<TYPE>(), 0, info.width_ * sizeof(TYPE), info.height_, stream));
	}

	//Copy device to device
	template<typename TYPE>	void UtilityForCUDA::copyDeviceMemory(TYPE*& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch = info.pitch<TYPE>();
		gpuErrchk(cudaMemcpy2DAsync(dst, pitch, src, pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault, stream));
	}
	template<typename TYPE>	void UtilityForCUDA::copyDeviceMemory(DeviceArray<TYPE>*& src, DeviceArray<TYPE>*& dst, int arrayLength, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch = info.pitch<TYPE>();
		for (int i = 0; i < arrayLength; i++)
		{
			gpuErrchk(cudaMemcpy2DAsync(dst->host[i], pitch, src->host[i], pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault, stream));
		}
	}
	//Upload cv::Mat to device memory
	template<typename TYPE>	void UtilityForCUDA::uploadMatToDevice(cv::Mat& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch = info.pitch<TYPE>();
		gpuErrchk(cudaMemcpy2DAsync(dst, pitch, src.data, src.step, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault, stream));
	}
	//Download array from device memory to host memory as cv::Mat
	template<typename TYPE>	void UtilityForCUDA::downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info, cv::Mat& dst)
	{
		size_t pitch = info.pitch<TYPE>();
		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpy2D(dst.data, dst.step, src, pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault));
	}
	template<typename TYPE>	cv::Mat UtilityForCUDA::downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info)
	{
		size_t pitch = info.pitch<TYPE>();
		cv::Mat dst(info.height_, info.width_, UtilityForCUDA::getCvType<TYPE>());
		cudaDeviceSynchronize();
		gpuErrchk(cudaMemcpy2D(dst.data, dst.step, src, pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault));
		return dst;
	}
	//Download DeviceArray from device memory to host memory as cv::Mat
	template<typename TYPE> cv::Mat UtilityForCUDA::downloadLinearArrayAsMat(DeviceArray<TYPE>*& src, SizeInfo& info)
	{
		size_t pitch = info.pitch<TYPE>();
		cv::Mat dst;
		std::vector<cv::Mat> planes;
		cudaDeviceSynchronize();
		for (int i = 0; i < src->arrayLength; i++)
		{
			cv::Mat tmp(info.height_, info.width_, UtilityForCUDA::getCvType<TYPE>());
			gpuErrchk(cudaMemcpy2D(tmp.data, tmp.step, src->host[i], pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault));
			planes.push_back(tmp);
		}
		cv::merge(planes, dst);
		return dst;
	}
	//Download DeviceArray from device memory to host memory as std::vector<cv::Mat>
	template<typename TYPE> std::vector<cv::Mat> UtilityForCUDA::downloadLinearArrayAsMatVector(DeviceArray<TYPE>*& src, SizeInfo& info)
	{
		size_t pitch = info.pitch<TYPE>();
		std::vector<cv::Mat> planes;
		cudaDeviceSynchronize();
		for (int i = 0; i < src->arrayLength; i++)
		{
			cv::Mat tmp(info.height_, info.width_, UtilityForCUDA::getCvType<TYPE>());
			gpuErrchk(cudaMemcpy2D(tmp.data, tmp.step, src->host[i], pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault));
			planes.push_back(tmp);
		}
		return planes;
	}





	////////////////////////////////////////////////////////////////////////////////
	//for cv
	template<> inline int UtilityForCUDA::getCvType<int>() { return CV_32SC1; }
	template<> inline int UtilityForCUDA::getCvType<int2>() { return CV_32SC2; }
	template<> inline int UtilityForCUDA::getCvType<int4>() { return CV_32SC4; }
	template<> inline int UtilityForCUDA::getCvType<float>() { return CV_32FC1; }
	template<> inline int UtilityForCUDA::getCvType<float2>() { return CV_32FC2; }
	template<> inline int UtilityForCUDA::getCvType<float4>() { return CV_32FC4; }
	template<> inline int UtilityForCUDA::getCvType<int>(int channelNum) { return CV_32SC(channelNum); }
	template<> inline int UtilityForCUDA::getCvType<float>(int channelNum) { return CV_32FC(channelNum); }

	////////////////////////////////////////////////////////////////////////////////
	//for texture
	template<> inline cudaChannelFormatDesc UtilityForCUDA::getCudaChannelFormatDesc<int>() { return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned); }
	template<> inline cudaChannelFormatDesc UtilityForCUDA::getCudaChannelFormatDesc<int2>() { return cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindSigned); }
	template<> inline cudaChannelFormatDesc UtilityForCUDA::getCudaChannelFormatDesc<int4>() { return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned); }
	template<> inline cudaChannelFormatDesc UtilityForCUDA::getCudaChannelFormatDesc<float>() { return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); }
	template<> inline cudaChannelFormatDesc UtilityForCUDA::getCudaChannelFormatDesc<float2>() { return cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat); }
	template<> inline cudaChannelFormatDesc UtilityForCUDA::getCudaChannelFormatDesc<float4>() { return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); }

	////////////////////////////////////////////////////////////////////////////////
	//Texture setting
	template<typename TYPE>	void UtilityForCUDA::setLinearArrayToTexture(TYPE* src, cudaTextureObject_t& texObj, SizeInfo& info, cudaTextureFilterMode filterMode)
	{
		cudaChannelFormatDesc channelDesc = UtilityForCUDA::getCudaChannelFormatDesc<TYPE>();
		size_t pitch = info.pitch<TYPE>();

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypePitch2D;
		resDesc.res.pitch2D.devPtr = src;
		resDesc.res.pitch2D.desc = channelDesc;
		resDesc.res.pitch2D.width = info.width_;
		resDesc.res.pitch2D.height = info.height_;
		resDesc.res.pitch2D.pitchInBytes = pitch;

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;
		texDesc.normalizedCoords = 0;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.filterMode = filterMode;

		gpuErrchk(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
	}


	////////////////////////////////////////////////////////////////////////////////
	//Cuda memory array
	template<class TYPE>
	class DeviceArray
	{
	public:
		DeviceArray();
		DeviceArray(int arrayLength, SizeInfo& info, bool initializeWithZero = false);
		DeviceArray(std::vector<cv::Mat>& mats, SizeInfo& info);
		DeviceArray(cv::Mat& mats, SizeInfo& info);
		~DeviceArray();

		void initializeWithZero(SizeInfo& info);

		TYPE** host;
		TYPE** device;
		int arrayLength;
	};
	template<class TYPE>DeviceArray<TYPE>::DeviceArray() {}
	template<class TYPE>
	DeviceArray<TYPE>::DeviceArray(int arrayLength, SizeInfo& info, bool initializeWithZero)
	{
		this->arrayLength = arrayLength;
		this->host = new TYPE * [arrayLength];
		gpuErrchk(cudaMalloc((void**)&this->device, sizeof(TYPE*) * arrayLength));
		for (int i = 0; i < arrayLength; i++)
		{
			if (initializeWithZero)
				UtilityForCUDA::allocateDeviceMemoryWithZero(this->host[i], info);
			else
				UtilityForCUDA::allocateDeviceMemory(this->host[i], info);
		}
		gpuErrchk(cudaMemcpy(this->device, this->host, sizeof(TYPE*) * arrayLength, cudaMemcpyDefault));
	}
	template<class TYPE>
	DeviceArray<TYPE>::DeviceArray(std::vector<cv::Mat>& mats, SizeInfo& info)
	{
		this->arrayLength = mats.size();
		this->host = new TYPE * [arrayLength];
		gpuErrchk(cudaMalloc((void**)&this->device, sizeof(TYPE*) * arrayLength));
		for (int i = 0; i < arrayLength; i++)
		{
			UtilityForCUDA::allocateDeviceMemory(this->host[i], mats[i], info);
		}
		gpuErrchk(cudaMemcpy(this->device, this->host, sizeof(TYPE*) * arrayLength, cudaMemcpyDefault));
	}
	template<class TYPE>
	DeviceArray<TYPE>::DeviceArray(cv::Mat& mat, SizeInfo& info)
	{
		std::vector<cv::Mat> mats;
		cv::split(mat, mats);
		this->arrayLength = mats.size();
		this->host = new TYPE * [arrayLength];
		gpuErrchk(cudaMalloc((void**)&this->device, sizeof(TYPE*) * arrayLength));
		for (int i = 0; i < arrayLength; i++)
		{
			UtilityForCUDA::allocateDeviceMemory(this->host[i], mats[i], info);
		}
		gpuErrchk(cudaMemcpy(this->device, this->host, sizeof(TYPE*) * arrayLength, cudaMemcpyDefault));
	}
	template<class TYPE>
	DeviceArray<TYPE>::~DeviceArray()
	{
		for (int i = 0; i < this->arrayLength; i++)
		{
			cudaFree(this->host[i]);
		}
		cudaFree(this->device);
		delete this->host;
	}
	//Initialize with 0
	template<class TYPE>
	inline void DeviceArray<TYPE>::initializeWithZero(SizeInfo& info)
	{
		for (int i = 0; i < arrayLength; i++)
			UtilityForCUDA::initializeDeviceMemoryWithZero(this->host[i], info);
	}

	////////////////////////////////////////////////////////////////////////////////
	//Array of texture
	template<class TYPE>
	class TextureArray
	{
	public:
		TextureArray() {};
		TextureArray(DeviceArray<TYPE>* deviceArray, cudaTextureFilterMode filterMode, SizeInfo info);
		TextureArray(std::vector<TYPE*>& deviceArray, cudaTextureFilterMode filterMode, SizeInfo info);
		TextureArray(std::vector<DeviceArray<TYPE>*>& deviceArray, cudaTextureFilterMode filterMode, SizeInfo info);
		~TextureArray();

		cudaTextureObject_t* host;
		cudaTextureObject_t* device;
		int arrayLength;
	};
	//Set each channel to texture individually
	template<class TYPE>
	TextureArray<TYPE>::TextureArray(DeviceArray<TYPE>* deviceArray, cudaTextureFilterMode filterMode, SizeInfo info)
	{
		this->arrayLength = deviceArray->arrayLength;
		this->host = new cudaTextureObject_t[arrayLength];
		gpuErrchk(cudaMalloc((void**)&this->device, sizeof(cudaTextureObject_t) * arrayLength));
		for (int i = 0; i < arrayLength; i++)
			UtilityForCUDA::setLinearArrayToTexture(deviceArray->host[i], this->host[i], info, filterMode);
		gpuErrchk(cudaMemcpy(this->device, this->host, sizeof(cudaTextureObject_t) * arrayLength, cudaMemcpyDefault));
	}
	//Set multiple single-channel data into individual textures
	template<class TYPE>
	TextureArray<TYPE>::TextureArray(std::vector<TYPE*>& deviceArray, cudaTextureFilterMode filterMode, SizeInfo info)
	{
		this->arrayLength = deviceArray.size();
		this->host = new cudaTextureObject_t[arrayLength];
		gpuErrchk(cudaMalloc((void**)&this->device, sizeof(cudaTextureObject_t) * arrayLength));
		for (int i = 0; i < arrayLength; i++)
			UtilityForCUDA::setLinearArrayToTexture(deviceArray[i], this->host[i], info, filterMode);
		gpuErrchk(cudaMemcpy(this->device, this->host, sizeof(cudaTextureObject_t) * arrayLength, cudaMemcpyDefault));
	}
	//Set multiple data with multiple channels to texture each channel separately
	template<class TYPE>
	TextureArray<TYPE>::TextureArray(std::vector<DeviceArray<TYPE>*>& deviceArray, cudaTextureFilterMode filterMode, SizeInfo info)
	{
		this->arrayLength = deviceArray.size() * deviceArray[0]->arrayLength;
		this->host = new cudaTextureObject_t[arrayLength];
		gpuErrchk(cudaMalloc((void**)&this->device, sizeof(cudaTextureObject_t) * arrayLength));
		int k = 0;
		for (int i = 0; i < deviceArray.size(); i++)
			for (int j = 0; j < deviceArray[i]->arrayLength; j++, k++)
				UtilityForCUDA::setLinearArrayToTexture(deviceArray[i]->host[j], this->host[k], info, filterMode);
		gpuErrchk(cudaMemcpy(this->device, this->host, sizeof(cudaTextureObject_t) * arrayLength, cudaMemcpyDefault));

	}

	template<class TYPE>
	TextureArray<TYPE>::~TextureArray()
	{
		for (int i = 0; i < this->arrayLength; i++)
		{
			cudaDestroyTextureObject(this->host[i]);
		}
		cudaFree(this->device);
		delete this->host;
	}


}