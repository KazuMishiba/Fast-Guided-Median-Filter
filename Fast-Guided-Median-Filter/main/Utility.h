#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "Utility.cuh"

#include <stdio.h>

#include <iostream>
#include <curand_kernel.h>
#include <assert.h>
#include<opencv2/opencv.hpp>


template<class TYPE> class DeviceArray;



////////////////////////////////////////////////////////////////////////////////
//GPUメモリの確認
void cu_memoryInfo();

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
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
	int width, height, range;
	dim3 blockSize, gridSize;
	SizeInfo();
	SizeInfo(int width, int height);
	SizeInfo(int width, int height, int range, dim3 blockSize);
	void setInfo(int width, int height, int range, dim3 blockSize);

	template<typename TYPE>	inline size_t pitch();

private:
	size_t pitchF1, pitchI1, pitchF2, pitchI2, pitchI3, pitchI4, pitchF4, pitchUC1, pitchUC3;
};

template<> inline size_t SizeInfo::pitch<int>() { return this->pitchI1;}
template<> inline size_t SizeInfo::pitch<int2>() { return this->pitchI2; }
template<> inline size_t SizeInfo::pitch<int3>() { return this->pitchI3; }
template<> inline size_t SizeInfo::pitch<int4>() { return this->pitchI4; }
template<> inline size_t SizeInfo::pitch<float>() { return this->pitchF1; }
template<> inline size_t SizeInfo::pitch<float2>() { return this->pitchF2; }
template<> inline size_t SizeInfo::pitch<float4>() { return this->pitchF4; }
template<> inline size_t SizeInfo::pitch<unsigned char>() { return this->pitchUC1; }
template<> inline size_t SizeInfo::pitch<uchar3>() { return this->pitchUC3; }

////////////////////////////////////////////////////////////////////////////////
//cuda
void cu_print_xy(int* src, int x, int y, SizeInfo& sizeInfo);
void cu_print_xy(float* src, int x, int y, SizeInfo& sizeInfo);


void cu_initializeWithValue(int* dst, int val, SizeInfo& sizeInfo, cudaStream_t stream);
void cu_initializeWithValue(float* dst, float val, SizeInfo& sizeInfo, cudaStream_t stream);



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
	gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width * sizeof(TYPE), info.height));
}
//メモリを確保しinitialValueMatで初期化
template<typename TYPE> void UtilityForCUDA::allocateDeviceMemory(TYPE*& dst, cv::Mat& initialValueMat, SizeInfo& info, cudaStream_t stream)
{
	size_t pitch;
	gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width * sizeof(TYPE), info.height));
	cudaMemcpy2DAsync(dst, pitch, initialValueMat.data, initialValueMat.step, info.width * sizeof(TYPE), info.height, cudaMemcpyDefault, stream);
}
//メモリを確保しinitialValueで初期化(float or int)
template<typename TYPE> void UtilityForCUDA::allocateDeviceMemory(TYPE*& dst, TYPE initialValue, SizeInfo& info, cudaStream_t stream)
{
	size_t pitch;
	gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width * sizeof(TYPE), info.height));
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
	gpuErrchk(cudaMallocPitch(&dst, &pitch, info.width * sizeof(TYPE), info.height));
	cudaMemset2DAsync(dst, pitch, 0, info.width * sizeof(TYPE), info.height, stream);
}
//0で初期化
template<typename TYPE> void UtilityForCUDA::initializeDeviceMemoryWithZero(TYPE*& dst, SizeInfo& info, cudaStream_t stream)
{
	cudaMemset2DAsync(dst, info.pitch<TYPE>(), 0, info.width * sizeof(TYPE), info.height, stream);
}

//copy
template<typename TYPE>	void UtilityForCUDA::copyDeviceMemory(TYPE*& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream)
{
	size_t pitch = info.pitch<TYPE>();
	cudaMemcpy2DAsync(dst, pitch, src, pitch, info.width * sizeof(TYPE), info.height, cudaMemcpyDefault, stream);
}
template<typename TYPE>	void UtilityForCUDA::copyDeviceMemory(DeviceArray<TYPE>*& src, DeviceArray<TYPE>*& dst, int arrayLength, SizeInfo& info, cudaStream_t stream)
{
	size_t pitch = info.pitch<TYPE>();
	for (int i = 0; i < arrayLength; i++)
	{
		cudaMemcpy2DAsync(dst->host[i], pitch, src->host[i], pitch, info.width * sizeof(TYPE), info.height, cudaMemcpyDefault, stream);
	}
}
//mat upload
template<typename TYPE>	void UtilityForCUDA::uploadMatToDevice(cv::Mat& src, TYPE*& dst, SizeInfo& info, cudaStream_t stream)
{
	size_t pitch = info.pitch<TYPE>();
	cudaMemcpy2DAsync(dst, pitch, src.data, src.step, info.width * sizeof(TYPE), info.height, cudaMemcpyDefault, stream);
}

//download array as mat
template<typename TYPE>	void UtilityForCUDA::downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info, cv::Mat& dst) 
{
	size_t pitch = info.pitch<TYPE>();
	cudaDeviceSynchronize();//同期
	cudaMemcpy2D(dst.data, dst.step, src, pitch, info.width * sizeof(TYPE), info.height, cudaMemcpyDefault);
}
template<typename TYPE>	cv::Mat UtilityForCUDA::downloadLinearArrayAsMat(TYPE*& src, SizeInfo& info)
{
	size_t pitch = info.pitch<TYPE>();
	cv::Mat dst(info.height, info.width, UtilityForCUDA::getCvType<TYPE>());
	cudaDeviceSynchronize();//同期
	cudaMemcpy2D(dst.data, dst.step, src, pitch, info.width * sizeof(TYPE), info.height, cudaMemcpyDefault);
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
		cv::Mat tmp(info.height, info.width, UtilityForCUDA::getCvType<TYPE>());
		cudaMemcpy2D(tmp.data, tmp.step, src->host[i], pitch, info.width * sizeof(TYPE), info.height, cudaMemcpyDefault);
		planes.push_back(tmp);
	}
	cv::merge(planes, dst);
	return dst;
}


////////////////////////////////////////////////////////////////////////////////
class PixelInfo
{
public:
	static void showPixelInfo(std::string title, cv::Mat& mat) {
		mouseParam mouseEvent = mouseParam();
		//コールバックの設定
		cv::setMouseCallback(title, PixelInfo::CallBackFunc, &mouseEvent);
		std::cout << "Click left button to get pixel info. Click right button or press any key to exit." << std::endl;
		int x = 0;
		int y = 0;
		while (1) {
			int key = cv::waitKey(100);
			bool flag = false;
			if (mouseEvent.event == cv::EVENT_LBUTTONDOWN)
			{
				x = mouseEvent.x;
				y = mouseEvent.y;
				flag = true;
			}
			else if (key == 2490368) {//up
				y--; flag = true;
			}
			else if (key == 2621440) {//down
				y++; flag = true;
			}
			else if (key == 2424832) {//left
				x--; flag = true;
			}
			else if (key == 2555904) {//right
				x++; flag = true;
			}
			else if (mouseEvent.event == cv::EVENT_RBUTTONDOWN || key != -1)
			{
				//終了
				break;
			}

			if (flag)
			{
				//座標取得
				std::cout << "(" << x << " , " << y << ") = ";
				dispPixelValue(mat, x, y);
				std::cout << std::endl;

				flag = false;
			}
		}
	}

private:
	//マウス入力用のパラメータ
	struct mouseParam {
		int x;
		int y;
		int event;
		int flags;
	};

	//コールバック関数
	static void CallBackFunc(int eventType, int x, int y, int flags, void* userdata)
	{
		mouseParam* ptr = static_cast<mouseParam*> (userdata);
		ptr->x = x;
		ptr->y = y;
		ptr->event = eventType;
		ptr->flags = flags;
	}

	static void dispPixelValue(cv::Mat mat, int x, int y)
	{
		switch (mat.type())
		{
		case CV_32FC1:
			std::cout << mat.at<float>(y, x);
			break;
		case CV_32SC1:
			std::cout << mat.at<int>(y, x);
			break;
		default:
			break;
		}
	}
};



////////////////////////////////////////////////////////////////////////////////
//show device
//single
template<typename TYPE>	void UtilityForCUDA::showDevice(TYPE*& src, SizeInfo& info, std::string title, bool useAbs, TYPE scalingFactor, bool showPixelInfo)
{
	cv::Mat mat = UtilityForCUDA::downloadLinearArrayAsMat(src, info);
	cv::Mat matDisp = mat * scalingFactor;
	cv::imshow(title, useAbs ? cv::abs(matDisp) : matDisp);
	if (showPixelInfo)
		PixelInfo::showPixelInfo(title, mat);
	else
		cv::waitKey(0);
}
//multiチャンネル全て表示
template<typename TYPE_SRC, typename TYPE_SCALING> void UtilityForCUDA::showDevice(DeviceArray<TYPE_SRC>*& src, SizeInfo& info, std::string title, bool useAbs, TYPE_SCALING scalingFactor, bool asImage, bool showPixelInfo)
{
	cv::Mat mat = UtilityForCUDA::downloadLinearArrayAsMat(src, info);
	if (asImage)
	{
		cv::Mat matDisp = mat * scalingFactor;
		cv::imshow(title, useAbs ? cv::abs(matDisp) : matDisp);
		if (showPixelInfo)
			PixelInfo::showPixelInfo(title, mat);
		else
			cv::waitKey(0);
	}
	else
	{
		std::vector<cv::Mat> planes;
		cv::split(mat, planes);
		for (int i = 0; i < planes.size(); i++)
		{
			cv::Mat matDisp = planes[i] * scalingFactor;
			cv::imshow(title + std::to_string(i), useAbs ? cv::abs(matDisp) : matDisp);
		}
		cv::waitKey(0);
	}
}
//マルチチャンネルの指定のチャンネル表示
template<typename TYPE_SRC, typename TYPE_SCALING> void UtilityForCUDA::showDevice(DeviceArray<TYPE_SRC>*& src, int channel, SizeInfo& info, std::string title, bool useAbs, TYPE_SCALING scalingFactor, bool showPixelInfo)
{
	cv::Mat mat = UtilityForCUDA::downloadLinearArrayAsMat(src, info);
	std::vector<cv::Mat> planes;
	cv::split(mat, planes);
	cv::Mat matDisp = planes[channel] * scalingFactor;
	cv::imshow(title, useAbs ? cv::abs(matDisp) : matDisp);
	if (showPixelInfo)
		PixelInfo::showPixelInfo(title, planes[channel]);
	else
		cv::waitKey(0);
}

/*
template<typename TYPE> void Utility::showDeviceForSingleChannel(TYPE*& src, SizeInfo& info, std::string title, bool useAbs, TYPE scalingFactor)
{
	cv::Mat mat = Utility::downloadLinearArrayAsMat(src, info) * scalingFactor;
	cv::imshow(title, useAbs ? cv::abs(mat) : mat);
	cv::waitKey(0);
}
//指定のチャンネルを表示
template<typename TYPE_SRC, typename TYPE_SCALING> void Utility::showDeviceForMultiChannels(TYPE_SRC*& src, int channel, SizeInfo& info, std::string title, bool useAbs, TYPE_SCALING scalingFactor)
{
	cv::Mat mat = Utility::downloadLinearArrayAsMat(src, info);
	std::vector<cv::Mat> planes;
	cv::split(mat, planes);
	cv::Mat matDisp = planes[channel] * scalingFactor;
	cv::imshow(title, useAbs ? cv::abs(matDisp) : matDisp);
	cv::waitKey(0);
}
*/






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
template<> inline cudaChannelFormatDesc UtilityForCUDA::getCudaChannelFormatDesc<int>() { return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);}
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
	resDesc.res.pitch2D.width = info.width;
	resDesc.res.pitch2D.height = info.height;
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
	DeviceArray(int arrayLength, SizeInfo info, bool initializeWithZero = false);
	DeviceArray(std::vector<cv::Mat>& mats, SizeInfo info);
	~DeviceArray();

	TYPE** host;
	TYPE** device;
	int arrayLength;
};
template<class TYPE>DeviceArray<TYPE>::DeviceArray() {}
template<class TYPE>
DeviceArray<TYPE>::DeviceArray(int arrayLength, SizeInfo info, bool initializeWithZero)
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
DeviceArray<TYPE>::DeviceArray(std::vector<cv::Mat>& mats, SizeInfo info)
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

