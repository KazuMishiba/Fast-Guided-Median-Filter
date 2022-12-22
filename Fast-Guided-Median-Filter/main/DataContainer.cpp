#include "DataContainer.h"

/*
ImageRW::ImageRW(std::string filePath, DataType dataType, bool selfGuided)
{
	this->filePath = filePath;
	this->dataType = dataType;
	this->selfGuided = selfGuided;
}

void ImageRW::readImage()
{
}
*/


Container_base::~Container_base()
{
}

void Container_base::convertImage(const cv::Mat & raw, cv::Mat & mat_8, cv::Mat & mat_32, cv::Mat & mat_32F, cv::Mat & mat_8_color, cv::Mat & mat_32_color, cv::Mat & mat_32F_color, std::vector<cv::Mat>& mat_32_color_vector, int *& device, DeviceArray<int>*& device_color, float *& deviceF, DeviceArray<float>*& deviceF_color, bool resize, cv::Size sz)
{
	if (resize)
		cv::resize(raw, mat_8_color, sz);
	else
		raw.copyTo(mat_8_color);

#if 0
	std::cout << "Debug: Color to Gray" << std::endl;
	cv::Mat temp;
	std::vector<cv::Mat> planes;
	cv::cvtColor(mat_8_color, temp, CV_BGR2GRAY);
	planes.push_back(temp);
	planes.push_back(temp);
	planes.push_back(temp);
	cv::merge(planes, mat_8_color);

	//cv::imwrite("testtest.png", mat_8_color);
#endif

	SizeInfo sizeInfo = SizeInfo(mat_8_color.cols, mat_8_color.rows, 0, dim3());
	if (mat_8_color.channels() == 3)
		cv::cvtColor(mat_8_color, mat_8, cv::COLOR_BGR2GRAY);
	else {
		//グレースケールならチャンネル複製してカラー化
		mat_8_color.copyTo(mat_8);
		std::vector<cv::Mat> planes;
		planes.push_back(mat_8);
		planes.push_back(mat_8);
		planes.push_back(mat_8);
		cv::merge(planes, mat_8_color);
	}

	if (this->convertFlag.mat_32)
		mat_8.convertTo(mat_32, CV_32SC1);
	if (this->convertFlag.mat_32F)
		mat_8.convertTo(mat_32F, CV_32FC1);
	if (this->convertFlag.mat_32_color)
		mat_8_color.convertTo(mat_32_color, CV_32SC3);
	if (this->convertFlag.mat_32F_color)
		mat_8_color.convertTo(mat_32F_color, CV_32FC3);
	if (this->convertFlag.mat_32_color_vector) {
		if (this->convertFlag.mat_32_color)
			cv::split(mat_32_color, mat_32_color_vector);
		else
		{
			cv::Mat temp;
			mat_8_color.convertTo(temp, CV_32SC3);
			cv::split(temp, mat_32_color_vector);
		}
	}
	if (this->convertFlag.device) {
		if (this->convertFlag.mat_32)
			UtilityForCUDA::allocateDeviceMemory(device, mat_32, sizeInfo);
		else {
			cv::Mat temp;
			mat_8.convertTo(temp, CV_32SC1);
			UtilityForCUDA::allocateDeviceMemory(device, temp, sizeInfo);
		}
	}
	if (this->convertFlag.device_color) {
		std::vector<cv::Mat> planes;
		if (this->convertFlag.mat_32_color)
			cv::split(mat_32_color, planes);
		else {
			cv::Mat temp;
			mat_8_color.convertTo(temp, CV_32SC3);
			cv::split(temp, planes);
		}
		device_color = new DeviceArray<int>(planes, sizeInfo);
	}
	if (this->convertFlag.deviceF) {
		if (this->convertFlag.mat_32F)
			UtilityForCUDA::allocateDeviceMemory(deviceF, mat_32F, sizeInfo);
		else {
			cv::Mat temp;
			mat_8.convertTo(temp, CV_32FC1);
			UtilityForCUDA::allocateDeviceMemory(deviceF, temp, sizeInfo);
		}
	}
	if (this->convertFlag.deviceF_color) {
		std::vector<cv::Mat> planes;
		if (this->convertFlag.mat_32F_color)
			cv::split(mat_32F_color, planes);
		else {
			cv::Mat temp;
			mat_8_color.convertTo(temp, CV_32FC3);
			cv::split(temp, planes);
		}
		deviceF_color = new DeviceArray<float>(planes, sizeInfo);
	}
}


Container_Image::~Container_Image()
{
	if (this->convertFlag.device) {
		cudaFree(I_device);
		if (!this->selfGuided)
			cudaFree(G_device);
	}
	if (this->convertFlag.device_color) {
		delete I_device_color;
		if (!this->selfGuided)
			delete G_device_color;
	}
	if (this->convertFlag.deviceF) {
		cudaFree(I_deviceF);
		if (!this->selfGuided)
			cudaFree(G_deviceF);
	}
	if (this->convertFlag.deviceF_color) {
		delete I_deviceF_color;
		if (!this->selfGuided)
			delete G_deviceF_color;
	}
	if (this->convertFlag.device || this->convertFlag.deviceF)
		cudaFree(result_device);
	if (this->convertFlag.device_color || this->convertFlag.deviceF_color)
		delete result_device_color;

}

/////////////////////////////////////////////////
//image
void Container_Image::load(bool resize, cv::Size sz, int divScale)
{
	this->I_raw = cv::imread(this->filePathSrc, -1);

	if (this->I_raw.empty())
	{
		std::cout << "Cannot read " << this->filePathSrc << std::endl;
		exit(-1);
	}

	int total = this->I_raw.total() * this->I_raw.channels();

	/*
	for (int i = 0; i < total; i++)
	{
		if (this->I_raw.data[i] >= 200)
		{
			((unsigned char*)this->I_raw.data)[i] = 200;
		}
	}
		*/
	/*
	std::vector<cv::Mat> planes;
	// 3つのチャネルB, G, Rに分離 (OpenCVではデフォルトでB, G, Rの順)
	cv::split(this->I_raw, planes);
	std::vector<cv::Mat> color_shuffle;
	color_shuffle.push_back(planes[2]);
	color_shuffle.push_back(planes[2]);
	color_shuffle.push_back(planes[2]);
	cv::merge(color_shuffle, this->I_raw);
	*/
	/*
	double mMin, mMax;
	cv::Point minP, maxP;
	cv::minMaxLoc(planes[0], &mMin, &mMax, &minP, &maxP);
	std::cout << "min: " << mMin << ", point " << minP << std::endl;
	std::cout << "max: " << mMax << ", point " << maxP << std::endl;
	cv::minMaxLoc(planes[1], &mMin, &mMax, &minP, &maxP);
	std::cout << "min: " << mMin << ", point " << minP << std::endl;
	std::cout << "max: " << mMax << ", point " << maxP << std::endl;
	cv::minMaxLoc(planes[2], &mMin, &mMax, &minP, &maxP);
	std::cout << "min: " << mMin << ", point " << minP << std::endl;
	std::cout << "max: " << mMax << ", point " << maxP << std::endl;
	//this->I_raw *= 3;
	*/

	if (divScale != 1)
	{
		//単に除算するとdouble型での演算が行われ、小数部が切り落とされない
		//this->I_raw /= divScale;

		unsigned int depth = this->I_raw.depth();
		switch (depth)
		{
		case CV_8U:
			for (int i = 0; i < total; i++)
				((unsigned char*)this->I_raw.data)[i] /= divScale;
			break;
		case CV_8S:
			for (int i = 0; i < total; i++)
				((signed char*)this->I_raw.data)[i] /= divScale;
			break;
		case CV_16U:
			for (int i = 0; i < total; i++)
				((unsigned short*)this->I_raw.data)[i] /= divScale;
			break;
		case CV_16S:
			for (int i = 0; i < total; i++)
				((short*)this->I_raw.data)[i] /= divScale;
			break;
		case CV_32S:
			for (int i = 0; i < total; i++)
				((int*)this->I_raw.data)[i] /= divScale;
			break;
		case CV_32F:
			for (int i = 0; i < total; i++)
				((float*)this->I_raw.data)[i] /= divScale;
			break;
		case CV_64F:
			for (int i = 0; i < total; i++)
				((double*)this->I_raw.data)[i] /= divScale;
			break;
		default:
			break;
		}
	}
	//cv::imshow("input", this->I_raw);
	//cv::waitKey(0);

	if (resize)
		this->imageSize = cv::Size(sz.width, sz.height);
	else
		this->imageSize = cv::Size(I_raw.cols, I_raw.rows);

	this->convertImage(I_raw, I, I32, I32F, I_color, I32_color, I32F_color, I32_color_vector, I_device, I_device_color, I_deviceF, I_deviceF_color, resize, sz);
	if (selfGuided)
	{
		//アドレスをコピー
		G_raw = I_raw;
		G = I;
		G32 = I32;
		G32F = I32F;
		G_color = I_color;
		G32_color = I32_color;
		G32F_color = I32F_color;
		G_device = I_device;
		G_device_color = I_device_color;
		G_deviceF = I_deviceF;
		G_deviceF_color = I_deviceF_color;
	}
	else
	{
		this->G_raw = cv::imread(this->filePathGuide);
		this->convertImage(G_raw, G, G32, G32F, G_color, G32_color, G32F_color, G32_color_vector, G_device, G_device_color, G_deviceF, G_deviceF_color, resize, sz);
	}


	SizeInfo sizeInfo = SizeInfo(this->imageSize.width, this->imageSize.height, 0, dim3());
	//result
	if (this->convertFlag.mat_32 || this->convertFlag.mat_32F || this->convertFlag.mat_32F_color || this->convertFlag.mat_32_color || this->convertFlag.mat_8 || this->convertFlag.mat_8_color)
		this->result = cv::Mat();
	if (this->convertFlag.device || this->convertFlag.deviceF)
		UtilityForCUDA::allocateDeviceMemory(this->result_device, sizeInfo);
	if (this->convertFlag.device_color || this->convertFlag.deviceF_color)
		this->result_device_color = new DeviceArray<int>(3, sizeInfo);

}


/////////////////////////////////////////////////
//video
void Container_Video::load(int startNum, int useNum, int numOfDigit, bool resize, cv::Size sz)
{
	if (useNum == -1)
	{
		//フォルダ情報から取得、未実装
		std::cout << "フレーム数自動取得未実装" << std::endl;
		exit(0);

	}
	this->frameNum = useNum;

	I_raw.resize(useNum);
	I.resize(useNum);
	I32.resize(useNum);
	I32F.resize(useNum);
	I_color.resize(useNum);
	I32_color.resize(useNum);
	I32F_color.resize(useNum);
	I32_color_vector.resize(useNum);
	I_device.resize(useNum);
	I_device_color.resize(useNum);
	I_deviceF.resize(useNum);
	I_deviceF_color.resize(useNum);
	G_raw.resize(useNum);
	G.resize(useNum);
	G32.resize(useNum);
	G32F.resize(useNum);
	G_color.resize(useNum);
	G32_color.resize(useNum);
	G32F_color.resize(useNum);
	G32_color_vector.resize(useNum);
	G_device.resize(useNum);
	G_device_color.resize(useNum);
	G_deviceF.resize(useNum);
	G_deviceF_color.resize(useNum);
	result.resize(useNum);
	result_device.resize(useNum);
	result_device_color.resize(useNum);

	for (int i = startNum, n = 0; i < startNum + useNum; i++, n++)
	{
		// filePath + "imgXXX.png" (XXXはフレーム番号)
		std::ostringstream ss;
		ss << std::setw(numOfDigit) << std::setfill('0') << i;
		//ss << std::setw(3) << std::setfill('0') << startNum;
		std::string numStr(ss.str());

		std::string fnSrc = this->filePathSrc + numStr + ".png";;
		I_raw[n] = cv::imread(fnSrc);

		if (this->I_raw[n].empty())
		{
			std::cout << "Cannot read " << fnSrc << std::endl;
			exit(-1);
		}

		if (n == 0)
		{
			if (resize)
				this->imageSize = cv::Size(sz.width, sz.height);
			else
				this->imageSize = cv::Size(I_raw[0].cols, I_raw[0].rows);
		}


		this->convertImage(I_raw[n], I[n], I32[n], I32F[n], I_color[n], I32_color[n], I32F_color[n], I32_color_vector[n], I_device[n], I_device_color[n], I_deviceF[n], I_deviceF_color[n], resize, sz);
		if (selfGuided)
		{
			//アドレスをコピー
			G_raw[n] = I_raw[n];
			G[n] = I[n];
			G32[n] = I32[n];
			G32F[n] = I32F[n];
			G_color[n] = I_color[n];
			G32_color[n] = I32_color[n];
			G32F_color[n] = I32F_color[n];
			G_device[n] = I_device[n];
			G_device_color[n] = I_device_color[n];
		}
		else
		{
			std::string fnGuide = this->filePathGuide + numStr + ".png";
			this->G_raw[n] = cv::imread(fnGuide);
			this->convertImage(G_raw[n], G[n], G32[n], G32F[n], G_color[n], G32_color[n], G32F_color[n], G32_color_vector[n], G_device[n], G_device_color[n], G_deviceF[n], G_deviceF_color[n], resize, sz);
		}

		SizeInfo sizeInfo = SizeInfo(this->imageSize.width, this->imageSize.height, 0, dim3());
		//result
		if (this->convertFlag.mat_32 || this->convertFlag.mat_32F || this->convertFlag.mat_32F_color || this->convertFlag.mat_32_color || this->convertFlag.mat_8 || this->convertFlag.mat_8_color)
			this->result[n] = cv::Mat();
		if (this->convertFlag.device)
			UtilityForCUDA::allocateDeviceMemory(this->result_device[n], sizeInfo);
		if (this->convertFlag.device_color)
			this->result_device_color [n]= new DeviceArray<int>(3, sizeInfo);

	}


}

Container_Video::~Container_Video()
{
	if (this->convertFlag.device) {
		for (const auto& e : I_device)
			cudaFree(e);
		if (!this->selfGuided)
		{
			for (const auto& e : G_device)
				cudaFree(e);
		}
	}
	if (this->convertFlag.device_color) {
		for (const auto& e : I_device_color)
			delete e;
		if (!this->selfGuided)
		{
			for (const auto& e : G_device_color)
				delete e;
		}
	}
	if (this->convertFlag.deviceF) {
		for (const auto& e : I_deviceF)
			cudaFree(e);
		if (!this->selfGuided)
		{
			for (const auto& e : G_deviceF)
				cudaFree(e);
		}
	}
	if (this->convertFlag.deviceF_color) {
		for (const auto& e : I_deviceF_color)
			delete e;
		if (!this->selfGuided)
		{
			for (const auto& e : G_deviceF_color)
				delete e;
		}
	}
	if (this->convertFlag.device || this->convertFlag.deviceF)
	{
		for (const auto& e : result_device)
			cudaFree(e);
	}
	if (this->convertFlag.device_color || this->convertFlag.deviceF_color)
	{
		for (const auto& e : result_device_color)
			delete e;
	}


}

/*
/////////////////////////////////////////////////
//MultiSpectralImage
void Container_MultiSpectralImage::load(int startNum, int useNum, bool resize, cv::Size sz)
{
	if (useNum == -1)
	{
		//フォルダ情報から取得、未実装
		std::cout << "フレーム数自動取得未実装" << std::endl;
		exit(0);

	}
	this->frameNum = useNum;

	I_raw.resize(useNum);
	I.resize(useNum);
	I32.resize(useNum);
	I32F.resize(useNum);
	I_color.resize(useNum);
	I32_color.resize(useNum);
	I32F_color.resize(useNum);
	I32_color_vector.resize(useNum);
	I_device.resize(useNum);
	I_device_color.resize(useNum);
	G_raw.resize(useNum);
	G.resize(useNum);
	G32.resize(useNum);
	G32F.resize(useNum);
	G_color.resize(useNum);
	G32_color.resize(useNum);
	G32F_color.resize(useNum);
	G32_color_vector.resize(useNum);
	G_device.resize(useNum);
	G_device_color.resize(useNum);
	result.resize(useNum);
	result_device.resize(useNum);
	result_device_color.resize(useNum);

	for (int i = startNum, n = 0; i < startNum + useNum; i++, n++)
	{
		// filePath + "imgXXX.png" (XXXはフレーム番号)
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << i;
		//ss << std::setw(3) << std::setfill('0') << startNum;
		std::string numStr(ss.str());

		std::string fnSrc = this->filePathSrc + numStr + ".png";;
		I_raw[n] = cv::imread(fnSrc, 0);//グレースケール(1ch)として読み込む

		if (n == 0)
		{
			if (resize)
				this->imageSize = cv::Size(sz.width, sz.height);
			else
				this->imageSize = cv::Size(I_raw[0].cols, I_raw[0].rows);
		}


		this->convertImage(I_raw[n], I[n], I32[n], I32F[n], I_color[n], I32_color[n], I32F_color[n], I32_color_vector[n], I_device[n], I_device_color[n], I_deviceF[n], I_deviceF_color[n], resize, sz);
		if (selfGuided)
		{
			//アドレスをコピー
			G_raw[n] = I_raw[n];
			G[n] = I[n];
			G32[n] = I32[n];
			G32F[n] = I32F[n];
			G_color[n] = I_color[n];
			G32_color[n] = I32_color[n];
			G32F_color[n] = I32F_color[n];
			G_device[n] = I_device[n];
			G_device_color[n] = I_device_color[n];
		}
		else
		{
			std::string fnGuide = this->filePathGuide + numStr + ".png";
			this->G_raw[n] = cv::imread(fnGuide, 0);//グレースケール(1ch)として読み込む
			this->convertImage(G_raw[n], G[n], G32[n], G32F[n], G_color[n], G32_color[n], G32F_color[n], G32_color_vector[n], G_device[n], G_device_color[n], G_deviceF[n], G_deviceF_color[n], resize, sz);
		}

		SizeInfo sizeInfo = SizeInfo(this->imageSize.width, this->imageSize.height, 0, dim3());
		//result
		if (this->convertFlag.mat_32 || this->convertFlag.mat_32F || this->convertFlag.mat_32F_color || this->convertFlag.mat_32_color || this->convertFlag.mat_8 || this->convertFlag.mat_8_color)
			this->result[n] = cv::Mat();
		if (this->convertFlag.device)
			Utility::allocateDeviceMemory(this->result_device[n], sizeInfo);
		if (this->convertFlag.device_color)
			this->result_device_color[n] = new DeviceArray<int>(3, sizeInfo);

	}


}
*/