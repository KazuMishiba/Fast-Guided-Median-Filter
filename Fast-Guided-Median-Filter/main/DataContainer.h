#pragma once

#include<opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utility.h"
#include<iostream>
#include <fstream>
#include <iomanip>

enum DataType
{
	Image, Video, LightFieldImage, MultiSpectralImage
};
/*
class ImageRW
{
public:
	ImageRW(std::string filePath, DataType dataType, bool selfGuided);
	ImageRW(std::string filePathSrc, std::string filePathGuide, DataType dataType, bool selfGuided);
	void readImage();
	void readVideo();
	void readLightFieldImage();
	void readMultiSpectralImage();

	std::string filePath;
	DataType dataType;
	bool selfGuided;
	

	//Image
	Container_Image* image;


};
*/
struct ConvertImageFlag
{
	bool mat_8;
	bool mat_32;//100+, propCPU gray
	bool mat_32F;
	bool mat_8_color;
	bool mat_32_color;//100+, propCPU color
	bool mat_32F_color;
	bool mat_32_color_vector;
	bool device;//propGPU gray int
	bool device_color;//propGPU color int
	bool deviceF;//propGPU gray float
	bool deviceF_color;//propGPU color float
};

class Container_base
{
public:
	Container_base(std::string filePathSrc, ConvertImageFlag convertFlag) : filePathSrc(filePathSrc), convertFlag(convertFlag), selfGuided(true) {};
	Container_base(std::string filePathSrc, std::string filePathGuide, ConvertImageFlag convertFlag) : filePathSrc(filePathSrc), filePathGuide(filePathGuide), convertFlag(convertFlag), selfGuided(false) {};
	~Container_base();

	std::string filePathSrc;
	std::string filePathGuide;
	//DataType dataType;
	bool selfGuided;
	ConvertImageFlag convertFlag;
	cv::Size imageSize;


	//virtual void load() {};
	void convertImage(const cv::Mat& raw, cv::Mat& mat_8, cv::Mat& mat_32, cv::Mat& mat_32F, cv::Mat& mat_8_color, cv::Mat& mat_32_color, cv::Mat& mat_32F_color, std::vector<cv::Mat>& mat_32_color_vector, int*& device, DeviceArray<int>*& device_color, float *& deviceF, DeviceArray<float>*& deviceF_color, bool resize, cv::Size sz);
private:

};



class Container_Image : public Container_base
{
public:
	Container_Image::~Container_Image();

	// 入力画像
	cv::Mat I_raw;
	// 処理対象画像
	cv::Mat I;//CV_8UC1
	cv::Mat I32;//CV_32SC1
	cv::Mat I32F;//CV_32FC1
	cv::Mat I_color;//CV_8UC3
	cv::Mat I32_color;//CV_32SC3
	cv::Mat I32F_color;//CV_32FC3
	std::vector<cv::Mat> I32_color_vector;//vector<CV_32SC1>
	int* I_device;//for cuda
	DeviceArray<int>* I_device_color;//color for cuda
	float* I_deviceF;//for cuda
	DeviceArray<float>* I_deviceF_color;//color for cuda
	// ガイド画像
	cv::Mat G_raw;
	cv::Mat G;//CV_8UC1
	cv::Mat G32;//CV_32SC1
	cv::Mat G32F;//CV_32FC1
	cv::Mat G_color;//CV_8UC3
	cv::Mat G32_color;//CV_32SC3
	cv::Mat G32F_color;//CV_32FC3
	std::vector<cv::Mat> G32_color_vector;//vector<CV_32SC1>
	int* G_device;//for cuda
	DeviceArray<int>* G_device_color;//color for cuda
	float* G_deviceF;//for cuda
	DeviceArray<float>* G_deviceF_color;//color for cuda

	cv::Mat result;
	int* result_device;
	DeviceArray<int>* result_device_color;

	void load(bool resize = false, cv::Size sz = cv::Size(0,0), int divScale = 1);

	using Container_base::Container_base;
private:

};



class Container_Video : public Container_base
{
public:
	Container_Video::~Container_Video();

	// 入力画像
	std::vector<cv::Mat> I_raw;
	// 処理対象画像
	std::vector<cv::Mat> I;//CV_8UC1
	std::vector<cv::Mat> I32;//CV_32SC1
	std::vector<cv::Mat> I32F;//CV_32FC1
	std::vector<cv::Mat> I_color;//CV_8UC3
	std::vector<cv::Mat> I32_color;//CV_32SC3
	std::vector<cv::Mat> I32F_color;//CV_32FC3
	std::vector<std::vector<cv::Mat>> I32_color_vector;//vector<CV_32SC1>
	std::vector<int*> I_device;//for cuda
	std::vector<DeviceArray<int>*> I_device_color;//color for cuda
	std::vector<float*> I_deviceF;//for cuda
	std::vector<DeviceArray<float>*> I_deviceF_color;//color for cuda
	// ガイド画像
	std::vector<cv::Mat> G_raw;
	std::vector<cv::Mat> G;//CV_8UC1
	std::vector<cv::Mat> G32;//CV_32SC1
	std::vector<cv::Mat> G32F;//CV_32FC1
	std::vector<cv::Mat> G_color;//CV_8UC3
	std::vector<cv::Mat> G32_color;//CV_32SC3
	std::vector<cv::Mat> G32F_color;//CV_32FC3
	std::vector<std::vector<cv::Mat>> G32_color_vector;//vector<CV_32SC1>
	std::vector<int*> G_device;//for cuda
	std::vector<DeviceArray<int>*> G_device_color;//color for cuda
	std::vector<float*> G_deviceF;//for cuda
	std::vector<DeviceArray<float>*> G_deviceF_color;//color for cuda

	std::vector<cv::Mat> result;
	std::vector<int*> result_device;
	std::vector<DeviceArray<int>*> result_device_color;

	void load(int startNum = 0, int useNum = -1, int numOfDigit = 3, bool resize = false, cv::Size sz = cv::Size(0, 0));
	int frameNum;

	using Container_base::Container_base;
private:

};

/*
//Videoを継承してMultiSpectralImage用とする。読み込むのは1チャンネル1画像として分解したpngデータで、名前はfilePath + "imgXXX.png" (XXXは0から始まるチャンネル番号)
class Container_MultiSpectralImage : public Container_Video
{
public:
	// 入力画像
	std::vector<cv::Mat> I_raw;
	// 処理対象画像
	std::vector<cv::Mat> I;//CV_8UC1
	std::vector<cv::Mat> I32;//CV_32SC1
	std::vector<cv::Mat> I32F;//CV_32FC1
	std::vector<cv::Mat> I_color;//CV_8UC3
	std::vector<cv::Mat> I32_color;//CV_32SC3
	std::vector<cv::Mat> I32F_color;//CV_32FC3
	std::vector<std::vector<cv::Mat>> I32_color_vector;//vector<CV_32SC1>
	std::vector<int*> I_device;//for cuda
	std::vector<DeviceArray<int>*> I_device_color;//color for cuda
	std::vector<float*> I_deviceF;//for cuda
	std::vector<DeviceArray<float>*> I_deviceF_color;//color for cuda
	// ガイド画像
	std::vector<cv::Mat> G_raw;
	std::vector<cv::Mat> G;//CV_8UC1
	std::vector<cv::Mat> G32;//CV_32SC1
	std::vector<cv::Mat> G32F;//CV_32FC1
	std::vector<cv::Mat> G_color;//CV_8UC3
	std::vector<cv::Mat> G32_color;//CV_32SC3
	std::vector<cv::Mat> G32F_color;//CV_32FC3
	std::vector<std::vector<cv::Mat>> G32_color_vector;//vector<CV_32SC1>
	std::vector<int*> G_device;//for cuda
	std::vector<DeviceArray<int>*> G_device_color;//color for cuda
	std::vector<float*> G_deviceF;//for cuda
	std::vector<DeviceArray<float>*> G_deviceF_color;//color for cuda

	std::vector<cv::Mat> result;
	std::vector<int*> result_device;
	std::vector<DeviceArray<int>*> result_device_color;

	//void load(int startNum = 0, int useNum = -1, bool resize = false, cv::Size sz = cv::Size(0, 0));
	int frameNum;

	//
	//void getFrameAsDeviceArray(int startFrame, int endFrame);

	using Container_Video::Container_Video;
private:

};
*/