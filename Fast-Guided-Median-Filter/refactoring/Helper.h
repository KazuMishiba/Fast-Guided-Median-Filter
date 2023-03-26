#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "Utility.cuh"

#include <stdio.h>

#include <iostream>
#include <curand_kernel.h>
#include <assert.h>
#include<opencv2/opencv.hpp>

namespace Helper
{

	template<class TYPE> class DeviceArray;



	////////////////////////////////////////////////////////////////////////////////
	//GPUメモリの確認
	inline void cu_memoryInfo()
	{
		size_t mf, ma;
		cudaMemGetInfo(&mf, &ma);
		cudaDeviceSynchronize();
		std::cout << "[GPU Memory] free: " << mf << " total: " << ma << std::endl << std::endl;
	}

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

	class SizeInfo {
	public:
		int width_, height_, range_;
		dim3 blockSize_, gridSize_;
		SizeInfo();
		//SizeInfo(int width, int height);
		//SizeInfo(int width, int height, int range);
		SizeInfo(int width, int height, int range = 256, dim3 blockSize = dim3(32, 32, 1));
		void setInfo(int width, int height, int range, dim3 blockSize);

		template<typename TYPE>	inline size_t pitch();
		template<typename TYPE>	inline void setPitch(size_t pitch);

	private:
		size_t pitchF1, pitchI1, pitchF2, pitchI2, pitchI3, pitchI4, pitchF4, pitchUC1, pitchUC3;
	};

	inline SizeInfo::SizeInfo()
	{
	}
	inline SizeInfo::SizeInfo(int width, int height, int range , dim3 blockSize) :
		width_(width), height_(height), range_(range), blockSize_(blockSize)
	{
		int gridSizeX = ceil(width / (float)this->blockSize_.x);
		int gridSizeY = ceil(height / (float)this->blockSize_.y);
		int gridSizeZ = 1;
		this->gridSize_ = dim3(gridSizeX, gridSizeY, gridSizeZ);
	}




	//未代入時にエラー吐きたい
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
		//メモリ割り当て
		template<typename TYPE>	static void allocateDeviceMemory(TYPE*& dst, SizeInfo& info);
		template<typename TYPE>	static void allocateDeviceMemory(TYPE*& dst, cv::Mat& initialValueMat, SizeInfo& info, cudaStream_t stream = NULL);
		template<typename TYPE>	static void allocateDeviceMemory(TYPE*& dst, TYPE initialValue, SizeInfo& info, cudaStream_t stream = NULL);
		template<typename TYPE>	static void allocateDeviceMemory(TYPE*& dst, TYPE*& src, int num, cudaStream_t stream = NULL);
		template<typename TYPE>	static void allocateDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream = NULL);
		template<typename TYPE>	static void initializeDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream = NULL);

		/*
		static void showDevice(float*& src, SizeInfo& info, std::string title, bool useAbs = false, float scalingFactor = 1.0f);
		static void showDevice(int*& src, SizeInfo& info, std::string title, bool useAbs = false, int scalingFactor = 1);
		static void showDevice(int4*& src, int channel, SizeInfo& info, std::string title, bool useAbs = false, int scalingFactor = 1);*/
		//画像表示　マルチチャンネルの場合は全て表示
		template<typename TYPE>	static void showDevice(TYPE*& src, SizeInfo& info, std::string title, bool useAbs = false, TYPE scalingFactor = 1.0f, bool showPixelInfo = false);
		template<typename TYPE_SRC, typename TYPE_SCALING> static void showDevice(DeviceArray<TYPE_SRC>*& src, SizeInfo& info, std::string title, bool useAbs = false, TYPE_SCALING scalingFactor = 1.0f, bool asImage = false, bool showPixelInfo = false);
		//マルチチャンネルの指定のチャンネル表示
		template<typename TYPE_SRC, typename TYPE_SCALING> static void showDevice(DeviceArray<TYPE_SRC>*& src, int channel, SizeInfo& info, std::string title, bool useAbs = false, TYPE_SCALING scalingFactor = 1.0f, bool showPixelInfo = false);

		//copy
		template<typename TYPE>	static void copyDeviceMemory(TYPE*& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream = NULL);
		template<typename TYPE>	static void copyDeviceMemory(DeviceArray<TYPE>*& src, DeviceArray<TYPE>*& dst, int arrayLength, SizeInfo& info, cudaStream_t stream = NULL);
		//mat upload
		template<typename TYPE>	static void uploadMatToDevice(cv::Mat& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream = NULL);
		//download array as mat
		template<typename TYPE>	static void downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info, cv::Mat& dst);
		template<typename TYPE>	static cv::Mat downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info);
		template<typename TYPE>	static cv::Mat downloadLinearArrayAsMat(DeviceArray<TYPE>*& src, SizeInfo& info);

		//texture
		template<typename TYPE>	static void setLinearArrayToTexture(TYPE* src, cudaTextureObject_t& texObj, SizeInfo& info, cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModeLinear);

	private:
		template<typename TYPE>	inline static cudaChannelFormatDesc getCudaChannelFormatDesc();
		template<typename TYPE>	inline static int getCvType();
		template<typename TYPE>	inline static int getCvType(int channelNum);
		/*
		template<typename TYPE>	static void showDeviceForSingleChannel(TYPE*& src, SizeInfo& info, std::string title, bool useAbs, TYPE scalingFactor);
		template<typename TYPE_SRC, typename TYPE_SCALING>	static void showDeviceForMultiChannels(TYPE_SRC*& src, int channel, SizeInfo& info, std::string title, bool useAbs, TYPE_SCALING scalingFactor);
		*/
	};



	////////////////////////////////////////////////////////////////////////////////
	//メモリ確保
	template<typename TYPE> void UtilityForCUDA::allocateDeviceMemory(TYPE*& dst, SizeInfo& info)
	{
		size_t pitch;
		gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width_ * sizeof(TYPE), info.height_));
		info.setPitch<TYPE>(pitch);
	}
	//メモリを確保しinitialValueMatで初期化
	template<typename TYPE> void UtilityForCUDA::allocateDeviceMemory(TYPE*& dst, cv::Mat& initialValueMat, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch;
		gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width_ * sizeof(TYPE), info.height_));
		info.setPitch<TYPE>(pitch);
		cudaMemcpy2DAsync(dst, pitch, initialValueMat.data, initialValueMat.step, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault, stream);
	}
	//メモリを確保しinitialValueで初期化(float or int)
	template<typename TYPE> void UtilityForCUDA::allocateDeviceMemory(TYPE*& dst, TYPE initialValue, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch;
		gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width_ * sizeof(TYPE), info.height_));
		info.setPitch<TYPE>(pitch);
		cu_initializeWithValue(dst, initialValue, info, stream);
	}

	//１D配列用確保
	template<typename TYPE> void UtilityForCUDA::allocateDeviceMemory(TYPE*& dst, TYPE*& src, int num, cudaStream_t stream)
	{
		cudaMalloc((void**)&dst, sizeof(TYPE) * num);
		cudaMemcpyAsync(dst, src, sizeof(TYPE) * num, cudaMemcpyDefault, stream);
	}
	//0で初期化した確保
	template<typename TYPE> void UtilityForCUDA::allocateDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch;
		gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width_ * sizeof(TYPE), info.height_));
		info.setPitch<TYPE>(pitch);
		cudaMemset2DAsync(dst, pitch, 0, info.width_ * sizeof(TYPE), info.height_, stream);
	}
	//0で初期化
	template<typename TYPE> void UtilityForCUDA::initializeDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream)
	{
		cudaMemset2DAsync(dst, info.pitch<TYPE>(), 0, info.width_ * sizeof(TYPE), info.height_, stream);
	}

	//copy
	template<typename TYPE>	void UtilityForCUDA::copyDeviceMemory(TYPE*& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch = info.pitch<TYPE>();
		cudaMemcpy2DAsync(dst, pitch, src, pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault, stream);
	}
	template<typename TYPE>	void UtilityForCUDA::copyDeviceMemory(DeviceArray<TYPE>*& src, DeviceArray<TYPE>*& dst, int arrayLength, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch = info.pitch<TYPE>();
		for (int i = 0; i < arrayLength; i++)
		{
			cudaMemcpy2DAsync(dst->host[i], pitch, src->host[i], pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault, stream);
		}
	}
	//mat upload
	template<typename TYPE>	void UtilityForCUDA::uploadMatToDevice(cv::Mat& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream)
	{
		size_t pitch = info.pitch<TYPE>();
		cudaMemcpy2DAsync(dst, pitch, src.data, src.step, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault, stream);
	}

	//download array as mat
	template<typename TYPE>	void UtilityForCUDA::downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info, cv::Mat& dst)
	{
		size_t pitch = info.pitch<TYPE>();
		cudaDeviceSynchronize();//同期
		cudaMemcpy2D(dst.data, dst.step, src, pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault);
	}
	template<typename TYPE>	cv::Mat UtilityForCUDA::downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info)
	{
		size_t pitch = info.pitch<TYPE>();
		cv::Mat dst(info.height_, info.width_, UtilityForCUDA::getCvType<TYPE>());
		cudaDeviceSynchronize();//同期
		cudaMemcpy2D(dst.data, dst.step, src, pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault);
		return dst;
	}
	//DeviceArray用
	template<typename TYPE> cv::Mat UtilityForCUDA::downloadLinearArrayAsMat(DeviceArray<TYPE>*& src, SizeInfo& info)
	{
		size_t pitch = info.pitch<TYPE>();
		cv::Mat dst;
		std::vector<cv::Mat> planes;
		cudaDeviceSynchronize();//同期
		for (int i = 0; i < src->arrayLength; i++)
		{
			cv::Mat tmp(info.height_, info.width_, UtilityForCUDA::getCvType<TYPE>());
			cudaMemcpy2D(tmp.data, tmp.step, src->host[i], pitch, info.width_ * sizeof(TYPE), info.height_, cudaMemcpyDefault);
			planes.push_back(tmp);
		}
		cv::merge(planes, dst);
		return dst;
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
	// texture bind
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
	//cuda memory array
	template<class TYPE>
	class DeviceArray
	{
	public:
		DeviceArray();
		DeviceArray(int arrayLength, SizeInfo& info, bool initializeWithZero = false);
		DeviceArray(std::vector<cv::Mat>& mats, SizeInfo& info);
		DeviceArray(cv::Mat& mats, SizeInfo& info);
		~DeviceArray();

		TYPE** host;
		TYPE** device;
		int arrayLength;
	};
	template<class TYPE>DeviceArray<TYPE>::DeviceArray() {}
	template<class TYPE>
	DeviceArray<TYPE>::DeviceArray(int arrayLength, SizeInfo& info, bool initializeWithZero)
	{
		this->arrayLength = arrayLength;
		//*array確保
		this->host = new TYPE * [arrayLength];
		cudaMalloc((void**)&this->device, sizeof(TYPE*) * arrayLength);
		//**確保
		for (int i = 0; i < arrayLength; i++)
		{
			if (initializeWithZero)
				UtilityForCUDA::allocateDeviceMemoryWithZero(this->host[i], info);
			else
				UtilityForCUDA::allocateDeviceMemory(this->host[i], info);
		}
		cudaMemcpy(this->device, this->host, sizeof(TYPE*) * arrayLength, cudaMemcpyDefault);
	}
	template<class TYPE>
	DeviceArray<TYPE>::DeviceArray(std::vector<cv::Mat>& mats, SizeInfo& info)
	{
		this->arrayLength = mats.size();
		//*array確保
		this->host = new TYPE * [arrayLength];
		cudaMalloc((void**)&this->device, sizeof(TYPE*) * arrayLength);
		//**確保
		for (int i = 0; i < arrayLength; i++)
		{
			UtilityForCUDA::allocateDeviceMemory(this->host[i], mats[i], info);
		}
		cudaMemcpy(this->device, this->host, sizeof(TYPE*) * arrayLength, cudaMemcpyDefault);
	}
	template<class TYPE>
	DeviceArray<TYPE>::DeviceArray(cv::Mat& mat, SizeInfo& info)
	{
		this->arrayLength = mat.channels();
		//*array確保
		this->host = new TYPE * [arrayLength];
		cudaMalloc((void**)&this->device, sizeof(TYPE*) * arrayLength);
		if (arrayLength == 1)
		{
			UtilityForCUDA::allocateDeviceMemory(this->host[0], mat, info);
		}
		else
		{
			std::vector<cv::Mat> mats;
			cv::split(mat, mats);
			//**確保
			for (int i = 0; i < arrayLength; i++)
			{
				UtilityForCUDA::allocateDeviceMemory(this->host[i], mats[i], info);
			}
		}
		cudaMemcpy(this->device, this->host, sizeof(TYPE*) * arrayLength, cudaMemcpyDefault);
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

	////////////////////////////////////////////////////////////////////////////////
	//array of texture
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
	//DeviceArray<TYPE>* (1フレーム複数チャンネル情報)をテクスチャにバインド
	template<class TYPE>
	TextureArray<TYPE>::TextureArray(DeviceArray<TYPE>* deviceArray, cudaTextureFilterMode filterMode, SizeInfo info)
	{
		this->arrayLength = deviceArray->arrayLength;
		this->host = new cudaTextureObject_t[arrayLength];
		cudaMalloc((void**)&this->device, sizeof(cudaTextureObject_t) * arrayLength);
		//texにdeviceArrayのhost?をバインド
		for (int i = 0; i < arrayLength; i++)
			UtilityForCUDA::setLinearArrayToTexture(deviceArray->host[i], this->host[i], info, filterMode);
		cudaMemcpy(this->device, this->host, sizeof(cudaTextureObject_t) * arrayLength, cudaMemcpyDefault);
	}
	//std::vector<TYPE*> (複数フレーム1チャンネル情報)をテクスチャにバインド
	template<class TYPE>
	TextureArray<TYPE>::TextureArray(std::vector<TYPE*>& deviceArray, cudaTextureFilterMode filterMode, SizeInfo info)
	{
		this->arrayLength = deviceArray.size();
		this->host = new cudaTextureObject_t[arrayLength];
		cudaMalloc((void**)&this->device, sizeof(cudaTextureObject_t) * arrayLength);
		//texにdeviceArrayのhost?をバインド
		for (int i = 0; i < arrayLength; i++)
			UtilityForCUDA::setLinearArrayToTexture(deviceArray[i], this->host[i], info, filterMode);
		cudaMemcpy(this->device, this->host, sizeof(cudaTextureObject_t) * arrayLength, cudaMemcpyDefault);
	}
	//std::vector<DeviceArray<TYPE>*> (複数フレーム複数チャンネル情報)をテクスチャにバインド
	template<class TYPE>
	TextureArray<TYPE>::TextureArray(std::vector<DeviceArray<TYPE>*>& deviceArray, cudaTextureFilterMode filterMode, SizeInfo info)
	{
		/*
		Utility::showDevice(deviceArray[0]->host[0], info, "G", false, 256);
		Utility::showDevice(deviceArray[0]->host[1], info, "G", false, 256);
		Utility::showDevice(deviceArray[0]->host[2], info, "G", false, 256);
		*/
		//フレーム順に各チャンネルを1列に並べる
		this->arrayLength = deviceArray.size() * deviceArray[0]->arrayLength;
		this->host = new cudaTextureObject_t[arrayLength];
		cudaMalloc((void**)&this->device, sizeof(cudaTextureObject_t) * arrayLength);
		//texにdeviceArrayのhost?をバインド
		int k = 0;
		for (int i = 0; i < deviceArray.size(); i++)
			for (int j = 0; j < deviceArray[i]->arrayLength; j++, k++)
				UtilityForCUDA::setLinearArrayToTexture(deviceArray[i]->host[j], this->host[k], info, filterMode);
		cudaMemcpy(this->device, this->host, sizeof(cudaTextureObject_t) * arrayLength, cudaMemcpyDefault);

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

	////////////////////////////////////////////////////////////////////////////////




	////////////////////////////////////////////////////////////////////////////////

}