#include "Experimenter.h"


#define debugTest false
#define debugTestScaling false



//何かしらのテストに使う
void Experimenter::test(std::string filePathSrc, std::string filePathGuide)
{
	int radius = 15;
	float eps2 = 25.5f * 25.5f;
	int Imax = 256;
	int threadNum = 12;

	bool useResize = false;
	const cv::Size sz = cv::Size(128 * 1, 128 * 1);
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true, false, false };
	if (filePathGuide == "")
		this->image = new Container_Image(filePathSrc, convertFlag);
	else
		this->image = new Container_Image(filePathSrc, filePathGuide, convertFlag);

	this->image->load(useResize, sz);
	SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1));

	std::cout << "h*w= " << this->image->imageSize.height << "×" << this->image->imageSize.width << std::endl;
	std::cout << "Radius: " << radius << std::endl;

	cv::Mat result1, result2;
	cv::Mat mat1;


	//result1 = FGMF2::filter2DWindow<gSum, fgSumUpToIndex, fg, int, float>(this->image->I32, this->image->G32, threadNum, radius, eps2, Imax);
	//cv::imshow("Result1", result1 * 256);

	/*

	cv::Mat result2 = cv::Mat(this->image->I32.size(), CV_32SC(1)); ;
	//FGMF1::filter2D<gSum, fgSumUpToIndex, fg_node, int, float>(this->image->I32, this->image->G32, result2, 0, this->image->I32.rows, 0, this->image->I32.cols, radius, eps2, Imax);
	//FGMF1::filter2D_2<gSum, fg_node, int, float>(this->image->I32, this->image->G32, result2, 0, this->image->I32.rows-1, 0, this->image->I32.cols-1, radius, eps2, Imax);
	result2 = FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, radius, eps2, Imax);
	//result2 = FGMF::filter2DInterfaceTest(this->image->I32, this->image->G32, threadNum, radius, eps2, Imax);
	result2.convertTo(result2, CV_8U);
	cv::imshow("Result2", result2);
	
	//
	//FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, radius, eps2, Imax, sizeInfo);
	//result1 = Utility::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);
	result1 = FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, radius, eps2, Imax);
	result1.convertTo(result1, CV_8U);
	cv::imshow("Result1", result1);

	cv::Mat dif = cv::abs(result1 - result2);
	cv::imshow("dif", dif * 10000);
	*/

	/*
	//
	mat1 = FGMF2::filter2DWindow<gSum, fgSumUpToIndex, fg, int, float>(this->image->I32, this->image->G32, threadNum, radius, eps2, Imax);
	mat1 *= 256;

	cv::imshow("Result1", mat1);
	*/


	//FGMF3::filter2DGPU_Test(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, radius, eps2, Imax, sizeInfo);

	   
	/*
	//多チャンネルテスト
	FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, radius, eps2, Imax, sizeInfo);
	//Utility::showDevice(this->image->result_device, sizeInfo, "test", 255 * 255);
	result2 = Utility::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);
	std::cout << "test" << std::endl;
	//FGMF3::filter2DGPU_MultiChannel<int>(this->image->I_device, this->image->G_device_color, this->image->result_device, radius, eps2, Imax, sizeInfo);
	//Utility::showDevice(this->image->I_device_color->host[2], sizeInfo, "test2", false, 255);
	FGMF3::filter2DGPU_MultiChannel<DeviceArray<int>>(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, radius, eps2, Imax, sizeInfo);
	result1 = Utility::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);
	//FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, radius, eps2, Imax, sizeInfo);
	//result1 = Utility::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);

	//result2 = FGMF2::filter2DInterface(this->image->I32, this->image->G32_color, threadNum, radius, eps2, Imax);
	//result2 = FGMF2::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, radius, eps2, Imax);
	//result2 = FGMF2::filter2DInterface(this->image->I32, this->image->G32, threadNum, radius, eps2, Imax);
	*/

	//OP sliding
	//result1 = FGMF2::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, radius, eps2, Imax);

	//list g3テスト
	threadNum = 12;
	result1 = FGMF1::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, radius, eps2, Imax);
	result2 = FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, radius, eps2, Imax);

	cv::imshow("Result1", result1 * 255);
	cv::imshow("Result2", result2 * 255);
	cv::waitKey(0);
	result1.convertTo(result1, CV_32S);
	result2.convertTo(result2, CV_32S);
	cv::Mat dif = cv::abs(result1 - result2);
	cv::imshow("dif", dif * 10000);
	cv::waitKey(0);

	return;



	//higher bit 読み込みテスト
	unsigned int depth = this->image->I_raw.depth();
	std::string strDepth =
		(
			depth == CV_8U ? "CV_8U" :
			depth == CV_8S ? "CV_8S" :
			depth == CV_16U ? "CV_16U" :
			depth == CV_16S ? "CV_16S" :
			depth == CV_32S ? "CV_32S" :
			depth == CV_32F ? "CV_32F" :
			depth == CV_64F ? "CV_64F" :
			"Other"
			);

	std::cout << strDepth << std::endl;

	cv::imshow("test1", this->image->I_raw*pow(2,4));
	
	double mMin, mMax;
	cv::Point minP, maxP;

	cv::minMaxLoc(this->image->I32, &mMin, &mMax, &minP, &maxP);
	//std::cout << mBar << std::endl;
	std::cout << "min: " << mMin << ", point " << minP << std::endl;
	std::cout << "max: " << mMax << ", point " << maxP << std::endl;

	cv::waitKey(0);







	
	//テンプレート展開確認用コード
	int r = radius;
	int radius_depth = r;

	//提案法CPU
	FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
	FGMF::filter2DInterface(this->image->I32, this->image->G32_color, threadNum, r, eps2, Imax);
	FGMF::filter2DInterface(this->image->I32_color, this->image->G32, threadNum, r, eps2, Imax);
	FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
	//提案法Window
	//2D
	FGMF2::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
	FGMF2::filter2DInterface(this->image->I32, this->image->G32_color, threadNum, r, eps2, Imax);
	FGMF2::filter2DInterface(this->image->I32_color, this->image->G32, threadNum, r, eps2, Imax);
	FGMF2::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
	//3D
	FGMF2::filter3DI1<gSum, fgSumUpToIndex, fg, int, float>(this->video->I32, this->video->G32, threadNum, radius, radius_depth, eps2, Imax);

	//提案法GPU
	//2D
	FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);
	FGMF3::filter2DGPU(this->image->I_device, this->image->G_device_color, this->image->result_device, r, eps2, Imax, sizeInfo);
	FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
	FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device, this->image->result_device_color, r, eps2, Imax, sizeInfo);
	//3D
	FGMF3::filter3DGPU(video->I_device, video->G_device, video->result_device, radius, radius_depth, eps2, Imax, sizeInfo);
	FGMF3::filter3DGPU(video->I_device, video->G_device_color, video->result_device, radius, radius_depth, eps2, Imax, sizeInfo);
	FGMF3::filter3DGPU(video->I_device_color, video->G_device, video->result_device_color, radius, radius_depth, eps2, Imax, sizeInfo);
	FGMF3::filter3DGPU(video->I_device_color, video->G_device_color, video->result_device_color, radius, radius, eps2, Imax, sizeInfo);

	//提案法 linked list
	//FGMF1::filter2D<gSum, fgSumUpToIndex, fg_node, int, float>(this->image->I32, this->image->G32, this->image->result, 0, this->image->I32.rows, 0, this->image->I32.cols, radius, eps2, Imax);
	FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);

}

void Experimenter::testForAblationStudy()
{
	//
	std::string guideImageName = R"(E:\LargeData\LightField\dataset_evaluation\evaluation-toolkit-master\data\additional\antinous\input_Cam040.png)";
	std::string inputImageName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\dataset\input\antinous\img040.png)";

	int radius = 15;
	float eps2 = 25.5f * 25.5f;
	int Imax = 256;
	int threadNum = 12;

	bool useResize = false;
	const cv::Size sz = cv::Size(128 * 1, 128 * 1);
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true, false, false };
	if (guideImageName == "")
		this->image = new Container_Image(inputImageName, convertFlag);
	else
		this->image = new Container_Image(inputImageName, guideImageName, convertFlag);

	this->image->load(useResize, sz);
	SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1));

	FGMF3::filter2D_CPU_Or_AblationStudy<gSum, fgSumUpToIndex, fg, int, float>(this->image->I32, this->image->G32, threadNum, radius, eps2, Imax);

}

void Experimenter::jointUpsamplingTest()
{
	std::string filePathSrc;
	std::string filePathGuide;

	std::string fileDir = R"(E:\MATLAB\GuidedFilterExtension\WienerFilterInterpretation\experiments\depthRestoration\images\)";
	std::string fileName = "pens";

	std::string scale = "8";
	std::string noiseSigma = "0.0001";

	filePathSrc = fileDir + fileName + "_depth.png";
	filePathGuide = fileDir + fileName + "_guide.png";

	std::string filePathSrcDegraded = fileDir + fileName + "_depth_upsampled_" + scale + "_" + noiseSigma + ".png";
	std::string filePathGuideDegraded = fileDir + fileName + "_guide_upsampled_" + scale + ".png";


	int radius = 4;
	//float eps2 = 25.5f * 25.5f;
	float eps2 = 0.5f * 0.5f;
	int Imax = 256;
	int threadNum = 12;

	bool useResize = false;
	const cv::Size sz = cv::Size(128 * 1, 128 * 1);
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true, false, false };
	this->image = new Container_Image(filePathSrc, filePathGuide, convertFlag);
	this->image->load(useResize, sz);
	SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1));

	std::cout << "h*w= " << this->image->imageSize.height << "×" << this->image->imageSize.width << std::endl;
	std::cout << "Radius: " << radius << std::endl;

	//劣化画像の読み込み
	Container_Image* degradedImage;
	degradedImage = new Container_Image(filePathSrcDegraded, filePathGuideDegraded, convertFlag);
	degradedImage->load(useResize, sz);

	
	UtilityForCUDA::showDevice(this->image->I_device, sizeInfo, "I", false, 255);
	//Utility::showDevice(this->image->G_device, sizeInfo, "G", false, 255);
	UtilityForCUDA::showDevice(degradedImage->I_device, sizeInfo, "I deg", false, 255);
	//Utility::showDevice(degradedImage->G_device, sizeInfo, "G deg", false, 255);
	

	FGMF3::upsamplingFilter2DGPU(degradedImage->I_device, this->image->G_device, degradedImage->G_device, this->image->result_device, radius, eps2, Imax, sizeInfo);
	//FGMF3::upsamplingFilter2DGPU(degradedImage->I_device, this->image->G_device_color, degradedImage->G_device_color, this->image->result_device, radius, eps2, Imax, sizeInfo);
	//FGMF3::upsamplingFilter2DGPU(degradedImage->I_device, this->image->G_device, this->image->G_device, this->image->result_device, radius, eps2, Imax, sizeInfo);

	UtilityForCUDA::showDevice(this->image->result_device, sizeInfo, "result", false, 255);


}


#define SPEED_TEST_MODE_I1G1
//#define SPEED_TEST_MODE_I1G3
//#define SPEED_TEST_MODE_I3G1
//#define SPEED_TEST_MODE_I3G3

void Experimenter::performSpeedTest(std::string filePathSrc)
{
	const int loopNum = 10;

	float eps2 = 25.5f * 25.5f;
	int Imax = 256 * 1;
	int threadNum = 12;
	const int r = 15;

	const cv::Size sz = cv::Size(128 * 4, 128 * 4);
	//const cv::Size sz = cv::Size(1920*4, 1080*4);
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true, false, false };
	this->image = new Container_Image(filePathSrc, convertFlag);
	this->image->load(true, sz);

	std::cout << "h*w= " << sz.height << "×" << sz.width << std::endl;
	double time1 = 0.0;
	double time2 = 0.0;
	double time3 = 0.0;
	double time4 = 0.0;
	std::cout << "Radius: " << r << std::endl;

	//GMF_GPU_MODE mode = GMF_GPU_MODE::MEDIAN_PREPARE;
	//GMF_GPU gmf_gpu = GMF_GPU(this->I.size(), blockSize, mode, r);
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

	std::chrono::system_clock::time_point  start, end;

	bool perform100faster = false;
	bool performPropCPU = false;
	bool performPropCPUwindow = false;
	bool performPropGPU = true;



	//従来法
	if (perform100faster)
	{
		start = std::chrono::system_clock::now();
		for (int j = 0; j < loopNum; j++)
		{
#if defined(SPEED_TEST_MODE_I1G1)
			l1Solver::filter(this->image->I32, this->image->G, r);
#elif defined(SPEED_TEST_MODE_I1G3)
			l1Solver::filter(this->image->I32, this->image->G_color, r);
#elif defined(SPEED_TEST_MODE_I3G1)
			l1Solver::filter(this->image->I32_color, this->image->G, r);
#elif defined(SPEED_TEST_MODE_I3G3)
			l1Solver::filter(this->image->I32_color, this->image->G_color, r);
#endif
		}
		end = std::chrono::system_clock::now();
		time1 += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "100+fast: ";
		std::cout << time1 << " [ms] / " << loopNum << "[times]" << std::endl;
	}

	//提案法CPU
	if (performPropCPU)
	{
		start = std::chrono::system_clock::now();
		for (int j = 0; j < loopNum; j++)
		{
#if defined(SPEED_TEST_MODE_I1G1)
			FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
#elif defined(SPEED_TEST_MODE_I1G3)
			FGMF::filter2DInterface(this->image->I32, this->image->G32_color, threadNum, r, eps2, Imax);
#elif defined(SPEED_TEST_MODE_I3G1)
			FGMF::filter2DInterface(this->image->I32_color, this->image->G32, threadNum, r, eps2, Imax);
#elif defined(SPEED_TEST_MODE_I3G3)
			FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
#endif
		}
		end = std::chrono::system_clock::now();
		time2 += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Prop CPU: ";
		std::cout << time2 << " [ms] / " << loopNum << "[times]" << std::endl;
	}
	//提案法CPU window
	if (performPropCPUwindow)
	{
		start = std::chrono::system_clock::now();
		for (int j = 0; j < loopNum; j++)
		{
#if defined(SPEED_TEST_MODE_I1G1)
			FGMF2::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			//FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			//FGMF::filter2DInterfaceTest(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
#elif defined(SPEED_TEST_MODE_I1G3)
#elif defined(SPEED_TEST_MODE_I3G1)
#elif defined(SPEED_TEST_MODE_I3G3)
#endif
		}
		end = std::chrono::system_clock::now();
		time4 += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Prop Win: ";
		std::cout << time4 << " [ms] / " << loopNum << "[times]" << std::endl;
	}
	//GuidedFilter::filterSimplified(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, sizeInfo);
	
	//提案法GPU
	if (performPropGPU)
	{
		start = std::chrono::system_clock::now();
		for (int j = 0; j < loopNum; j++)
		{
#if defined(SPEED_TEST_MODE_I1G1)
			FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);
#elif defined(SPEED_TEST_MODE_I1G3)
			std::cout << j << " ";
			start = std::chrono::system_clock::now();
			//FGMF3::filter2DGPU(this->image->I_device, this->image->G_device_color, this->image->result_device, r, eps2, Imax, sizeInfo);
			FGMF3::filter2DGPU_MultiChannel(this->image->I_device, this->image->G_device_color, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();
			time3 += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//cv::Mat result1 = Utility::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
			//cv::imshow("Result1", result1 * 255);
			//cv::waitKey(0);
			//Utility::showDevice(this->image->result_device, sizeInfo, "test", 255*255);
#elif defined(SPEED_TEST_MODE_I3G1)
#elif defined(SPEED_TEST_MODE_I3G3)
			FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
#endif
		}
		end = std::chrono::system_clock::now();
		time3 += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Prop GPU: ";
		std::cout << time3 << " [ms] / " << loopNum << "[times]" << std::endl;
	}

				/*
				std::chrono::system_clock::time_point  start, end; // 型は auto で可
				start = std::chrono::system_clock::now(); // 計測開始時間
				gmf_gpu.filterWithConstantTime(this->I32, this->G32F, r, eps2, Imax);
				end = std::chrono::system_clock::now();  // 計測終了時間
				time4 += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //処理に要した時間をミリ秒に変換
				*/
				//cu_memoryInfo();


				/*
				gmf_gpu.prepareMemory(this->I32, this->G32F);
				gmf_gpu.filter();
				gmf_gpu.clearMemory();
				*/


}

/*
Experimenter::Experimenter()
{
}
*/
//ノイズ除去の実施
void Experimenter::performNoiseReduction2D()
{
	//時間計測用
	std::chrono::system_clock::time_point  start, end;


	if (true)
	{
		//例えばある手法について
		//for (const auto& fn : this->fileNames) 
		{
			//真値画像読み込み
			cv::Mat raw = cv::imread(fileName);
			raw.convertTo(raw, CV_32S);
			//ノイズ付与画像読み込み　gauss, implus,  sigma,
			cv::Mat I = cv::imread(fileName);

			//計算時間測定開始して
			start = std::chrono::system_clock::now(); // 計測開始時間

			//処理して (入力画像をガイドとして用いる)
			int threadNum = 12;
			cv::Mat result = FGMF::filter2DI3G1_MultiThread(I, I, threadNum, this->param_fgmf.radius, this->param_fgmf.eps2, this->param_fgmf.Imax);

			//計算時間測定終了して
			end = std::chrono::system_clock::now();  // 計測終了時間
			double calculationTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//後処理して
			this->postProcessForNoiseReduction2D(raw, result, calculationTime, fileName);

			//次の画像へ
		}
	}
}

void Experimenter::postProcessForNoiseReduction2D(const cv::Mat& raw, const cv::Mat& result, double calculationTime, std::string fileName)
{
	std::string saveFileName, windowName;
	//評価指標を計算して
	//double psnr = calculatePSNR(raw, result);
	//保存して
	//saveResult(result, saveFileName);
	//計算時間保存

	//表示して
	//showResult(result, windowName);
}

double Experimenter::calculatePSNR(const cv::Mat& I1, const cv::Mat& I2, int maxVal)
{
	int channelNum = I1.channels();
	cv::Mat J1, J2;
	I1.convertTo(J1, CV_32S);
	I2.convertTo(J2, CV_32S);
	cv::Mat s1;
	cv::absdiff(J1, J2, s1);
	s1.convertTo(s1, CV_32F);
	s1 = s1.mul(s1);
	cv::Scalar s = sum(s1);
	if (channelNum == 1)
	{
		double sse = s.val[0];
		double mse = sse / (double)(J1.channels() * J1.total());
		double psnr = 10.0 * log10((maxVal * maxVal) / mse);
		return psnr;
	}
	else if (channelNum == 3)
	{
		double sse = s.val[0] + s.val[1] + s.val[2];
		double mse = sse / (double)(J1.channels() * J1.total());
		double psnr = 10.0 * log10((maxVal * maxVal) / mse);
		return psnr;
	}
	else
	{
		std::cout << "PSNR channel error." << std::endl;
		exit(0);
	}
}

double Experimenter::calculateSSIM(const cv::Mat & i1, const cv::Mat & i2)
{
	const double C1 = 6.5025, C2 = 58.5225;//for 0-255
	/***************************** INITS **********************************/
	cv::Mat I1, I2;
	i1.convertTo(I1, CV_32F);
	i2.convertTo(I2, CV_32F);
	cv::Mat I2_2 = I2.mul(I2);        // I2^2
	cv::Mat I1_2 = I1.mul(I1);        // I1^2
	cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2
	/*************************** END INITS **********************************/
	cv::Mat mu1, mu2;                   // PRELIMINARY COMPUTING
	cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu1_mu2 = mu1.mul(mu2);
	cv::Mat sigma1_2, sigma2_2, sigma12;
	cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;
	cv::Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
	cv::Mat ssim_map;
	cv::divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
	cv::Scalar mssim = cv::mean(ssim_map);   // mssim = average of ssim map
	double mmssim = cv::sum(mssim)[0] / I1.channels();
	return mmssim;
}

double Experimenter::calculateEKI(const cv::Mat & i1, const cv::Mat & i2)
{
	/*
	z1 = Σ(d1 - mean(d1))(d2 - mean(d2))
	z2 = Σ(d1 - mean(d1))^2 Σ(d2 - mean(d2))^2
	epi = z1 / sqrt(z2)
	*/
	cv::Mat I1, I2;
	i1.convertTo(I1, CV_32F);
	i2.convertTo(I2, CV_32F);
	//laplacian filter
	cv::Mat d1, d2;
	cv::Laplacian(I1, d1, CV_32F);
	cv::Laplacian(I2, d2, CV_32F);
	cv::Scalar m1, m2;
	m1 = cv::mean(d1);
	m2 = cv::mean(d2);
	//
	cv::Mat dd1, dd2;
	dd1 = d1 - m1;
	dd2 = d2 - m2;
	//
	cv::Scalar z1, z2;
	z1 = cv::sum(dd1.mul(dd2));
	z2 = cv::sum(dd1.mul(dd1)).mul(cv::sum(dd2.mul(dd2)));
	cv::Scalar epi;
	cv::sqrt(z2, z2);
	cv::divide(z1, z2, epi);
	double mepi = cv::sum(epi)[0] / I1.channels();

	return mepi;
}

//動画に対する単なるフィルタリング
void Experimenter::performFilteringForVideo(std::string filePathSrc, std::string filePathGuide)
{
	float eps2 = 25.5f * 25.5f * 1;
	int Imax = 256;
	int startFrameNum = 0;
	int frameNum = 7;
	int radius = 15;
	int radius_depth = 2;
	//CPU
	int threadNum = 12;


	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true };
	if (filePathGuide == "")
		this->video = new Container_Video(filePathSrc, convertFlag);
	else
		this->video = new Container_Video(filePathSrc, filePathGuide, convertFlag);

	video->load(startFrameNum, frameNum);

	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	SizeInfo sizeInfo = SizeInfo(video->imageSize.width, video->imageSize.height, Imax, blockSize);
	


	//std::vector<cv::Mat> results2 = FGMF2::filter3DI1<gSum, fgSumUpToIndex, fg, int, float>(this->video->I32, this->video->G32, threadNum, radius, radius_depth, eps2, Imax);

	//GPU


	//std::vector<cv::Mat> results =
	//I1G1
	//FGMF3::filter3DGPU(video->I_device, video->G_device, video->result_device, radius, radius_depth, eps2, Imax, sizeInfo);
	//I1G3
	//FGMF3::filter3DGPU(video->I_device, video->G_device_color, video->result_device, radius, radius_depth, eps2, Imax, sizeInfo);
	//I3G1
	//FGMF3::filter3DGPU(video->I_device_color, video->G_device, video->result_device_color, radius, radius_depth, eps2, Imax, sizeInfo);
	//I3G3
	FGMF3::filter3DGPU(video->I_device_color, video->G_device_color, video->result_device_color, radius, radius_depth, eps2, Imax, sizeInfo);

	// = FGMF::filter3DWindow<gSum, fgSumUpToIndex, fg, int, float>(data3d, data3d, threadNum, radius, radius_depth);


	for (int i = 0; i < video->frameNum; i++)
	{
		cv::Mat mat1, mat2;

		//cv::imshow("CPU", results2[i]*256);

		//Utility::showDevice(video->result_device[i], sizeInfo, std::to_string(i), true, 256);
		//Utility::showDevice(video->result_device[i], sizeInfo, "result", true, 256);
		UtilityForCUDA::showDevice(video->result_device_color[i], sizeInfo, "result", true, 256, true, false);
		//Utility::showDevice(video->I_device[i], sizeInfo, "I" + std::to_string(i), true, 256);
		//Utility::showDevice(video->G_device[i], sizeInfo, "G" + std::to_string(i), true, 256);

		/*
		mat1 = Utility::downloadLinearArrayAsMat(video->result_device[i], sizeInfo);
		mat2 = results2[i];

		cv::Mat diff = cv::abs(mat1 - mat2) * 1000;
		cv::imshow("Diff", diff);
		*/


		/*
		results[i].convertTo(result, CV_8U);
		cv::imshow("Result", result);
		results2[i].convertTo(result2, CV_8U);
		cv::imshow("Result2", result2);
		*/



		/*
		cv::Mat result2 = FGMF::filter2DI1G1_MultiThread(data3d[i], data3d[i], threadNum, radius, eps2, Imax);


		cv::Mat diff = cv::abs(results[i] - result2) * 1000;
		cv::imshow("Diff", diff);

		results[i].convertTo(result, CV_8U);
		cv::imshow("Result", result);
		result2.convertTo(result2, CV_8U);
		cv::imshow("Result2", result2);
		*/

		//cv::waitKey(0);
	}

}

//#define SpeedTestConstantTime
//#define SpeedTestFaster
#define SpeedTestPropCPU
//#define SpeedTestPropList
//#define SpeedTestPropGPU

//半径変化か　offなら画像サイズ変化
//#define performRadiusChange
//論文用2D 8bit画像速度テスト
void Experimenter::speedTest2D8bitForPaper()
{
	//各手法の切り替えは#if　スイッチで行う OpenMP使わない手法を試す場合はOpenMPのスイッチ自体を切っておくことに注意

	/*
	実験手順
	threadNum=1;
	OpenMPスイッチ切る
	SpeedTest...を1つずつ変えて実行
	#define performRadiusChange　を有効にしてまたSpeedTest...を1つずつ変えて実行
	threadNum=12
	OpenMPスイッチ入れる
	#define performRadiusChange　を無効にする
	#define SpeedTestFaster, #define SpeedTestPropCPUを1つずつ実行
	#define performRadiusChange　を有効にしてまた1つずつ変えて実行

	*/

	//設定
	float eps2 = 25.5f * 25.5f;
	int Imax = 256 * 1;
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);


	//計測時間の保存
	std::string saveTimeName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Speed Test\result.txt)";
	//ファイルの作成
	std::ofstream writing_file;
	writing_file.open(saveTimeName, std::ios::app);





	std::string methodName;
#ifdef SpeedTestConstantTime
	methodName = "ConstantTime";
	ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };
#endif // SpeedTestConstantTime

#ifdef SpeedTestFaster
	methodName = "100times";
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };
#endif // SpeedTestFaster

#ifdef SpeedTestPropCPU
	methodName = "PropCPU";
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };
#endif // SpeedTestPropCPU

#ifdef SpeedTestPropList
	methodName = "PropList";
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };
#endif // SpeedTestPropList

#ifdef SpeedTestPropGPU
	methodName = "PropGPU";
	ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, false, false };
#endif // SpeedTestPropGPU

	//ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, false, false, false, false };


#ifdef performRadiusChange
	std::vector<int> rs = { 2,7,17,32,52};
	cv::Size sz = cv::Size(1920, 1080);
	//std::vector<int> rs = { 2, 7 };
	//cv::Size sz = cv::Size(640, 480);
#else
	int r = 7;
	std::vector<cv::Size> sizes = { cv::Size(640, 480), cv::Size(1024, 768), cv::Size(1920, 1080) };
#endif



	//タイマー準備
	std::chrono::system_clock::time_point start, end;
#ifdef performRadiusChange
	std::vector<double> timer_gray(rs.size(), 0.0);
	std::vector<double> timer_color(rs.size(), 0.0);
#else
	std::vector<double> timer_gray(sizes.size(), 0.0);
	std::vector<double> timer_color(sizes.size(), 0.0);
#endif

	//画像データ読み込み(複数)
	//まずリストの読み込み
	std::string imageListFile = R"(E:\MATLAB\FastMedianFiltering\Experiments\Speed Test\HD_List.txt)";
	std::string fileName;
	std::ifstream fs(imageListFile);

	int n = 0;

	while (!fs.eof())
	{
		n++;
		std::cout << n << " ";
		//ファイル名の読み込み
		fs >> fileName;
		//std::cout << fileName << std::endl;

#ifdef performRadiusChange
		//半径変化
		for (int i = 0; i < rs.size(); i++)
		{
			int r = rs[i];
#else
		//画像サイズ変化
		for (int i = 0; i < sizes.size(); i++)
		{
			cv::Size sz = sizes[i];
#endif

			//画像読み込み、サイズ指定
			this->image = new Container_Image(fileName, convertFlag);
			this->image->load(true, sz);


			//手法実行
#ifdef SpeedTestConstantTime
			//constant timeのフィルタにはガイデッドフィルタを用いており、その場合アルゴリズムの特性上、半径を指定したときには実際に使用される半径はその2倍になる
			//一方でガイデッドフィルタの論文ではボックスフィルタの半径をフィルタの半径と呼んでいるっぽいので、そのまま使う
			//cu_memoryInfo();
			//constant time の実行
			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			//グレースケール（入力・ガイド共に）
			start = std::chrono::system_clock::now();//タイマー作動
			ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//I3G1 test
			//ConstantTimeWMF::filter2DGPU(this->image->I_device_color, this->image->G_deviceF, this->image->result_device_color, r, eps2, Imax, sizeInfo);
			//I1G3 test
			//ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF_color, this->image->result_device, r, eps2, Imax, sizeInfo);
			//Utility::showDevice(this->image->result_device, sizeInfo, "result", false, 256, false);
			//カラー（入力・ガイド共に）
			start = std::chrono::system_clock::now();//タイマー作動
			ConstantTimeWMF::filter2DGPU(this->image->I_device_color, this->image->G_deviceF_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//Utility::showDevice(this->image->result_device_color, sizeInfo, "result", false, 256, false);

#endif // SpeedTestConstantTime

#ifdef SpeedTestFaster
			//100+times faster WMFの実行

			//cv::Mat result, mat;
			//グレースケール（入力・ガイド共に）
			start = std::chrono::system_clock::now();//タイマー作動
			l1Solver::filter(this->image->I32, this->image->G, r);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result1", mat);

			//カラー（入力・ガイド共に）
			start = std::chrono::system_clock::now();//タイマー作動
			l1Solver::filter(this->image->I32_color, this->image->G_color, r);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result2", mat);
			//cv::waitKey(0);

#endif // SpeedTestFaster

#ifdef SpeedTestPropCPU
			//提案法CPU（MD sliding window）の実行

			//グレースケール（入力・ガイド共に）
			//cv::Mat result, mat;
			start = std::chrono::system_clock::now();//タイマー作動
			FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result1", mat);
			//カラー（入力・ガイド共に）
			start = std::chrono::system_clock::now();//タイマー作動
			FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result2", mat);
			//cv::waitKey(0);

#endif // SpeedTestPropCPU

#ifdef SpeedTestPropList
			//提案法CPU（List）の実行

			//グレースケール（入力・ガイド共に）
			//cv::Mat result, mat;
			start = std::chrono::system_clock::now();//タイマー作動
			//result = 
			FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			/*
			result.convertTo(mat, CV_8U);
			cv::imshow("Result1", mat);
			cv::waitKey(0);
			*/
			//カラー（入力・ガイド共に）
			start = std::chrono::system_clock::now();//タイマー作動
			//result = 
			FGMF1::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			/*
			result.convertTo(mat, CV_8U);
			cv::imshow("Result2", mat);
			cv::waitKey(0);
			*/
#endif // SpeedTestPropList

#ifdef SpeedTestPropGPU
			//提案法GPU（1D sliding window）の実行
			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			//Utility::showDevice(this->image->I_device, sizeInfo, "input", false, 256, false);

			//グレースケール（入力・ガイド共に）
			start = std::chrono::system_clock::now();//タイマー作動
			FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//Utility::showDevice(this->image->result_device, sizeInfo, "result", false, 256, false);

			//カラー（入力・ガイド共に）
			start = std::chrono::system_clock::now();//タイマー作動
			FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//Utility::showDevice(this->image->result_device_color, sizeInfo, "result", false, 256, false);

#endif // SpeedTestPropGPU


		
			

			delete this->image;

		}



	}

	//出力
	std::cout << std::endl;

	writing_file << methodName << std::endl;

#ifdef performRadiusChange
	writing_file << "Size = (" << sz.width << ", " << sz.height << "), Radius = ";
	for (int i = 0; i < rs.size(); i++)
	{
		writing_file << rs[i] << ",";
	}
	writing_file << std::endl;
#else
	writing_file << "Radius = " << r << ", Size = ";
	for (int i = 0; i < sizes.size(); i++)
	{
		writing_file << "(" << sizes[i].width << "," << sizes[i].height << "), ";
	}
	writing_file << std::endl;
#endif

	std::cout << "Gray" << std::endl;
	//writing_file << "Gray" << std::endl;
	for (int i = 0; i < timer_gray.size(); i++)
	{
		std::cout << timer_gray[i] / float(n) << "[ms/image]" << std::endl;
		writing_file << timer_gray[i] / float(n) << "\t";
	}
	writing_file << std::endl;
	std::cout << "Color" << std::endl;
	for (int i = 0; i < timer_color.size(); i++)
	{
		std::cout << timer_color[i] / float(n) << "[ms/image]" << std::endl;
		writing_file << timer_color[i] / float(n) << "\t";
	}
	writing_file << std::endl;
}




//#define HigherBitTestConstantTime
//#define HigherBitPropCPU
#define HigherBitPropGPU
//#define HigherBitPropList

//論文用2D higher bit画像速度テスト
void Experimenter::speedTest2DHigherBitForPaper()
{
	//各手法の切り替えは#if　スイッチで行う OpenMP使わない手法を試す場合はOpenMPのスイッチ自体を切っておくことに注意


	//設定
	float eps2 = 25.5f * 25.5f;
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int r = 7;

	//propGPUは12までしか動作しない
	//constant timeは8ビットでしか動かなかった
	//propCPUとListは全部動作
	std::vector<int> bits = { 8,10,12,14,16 };
	//std::vector<int> bits = { 10 };


	//計測時間の保存
	std::string saveTimeName = R"(E:\MATLAB\FastMedianFiltering\Experiments\HighPrecision\result.txt)";
	//ファイルの作成
	std::ofstream writing_file;
	writing_file.open(saveTimeName, std::ios::app);


	std::string methodName;
#ifdef HigherBitTestConstantTime
	methodName = "ConstantTime";
	ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };
#endif

#ifdef HigherBitPropList
	methodName = "PropList";
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };
#endif

#ifdef HigherBitPropCPU
	methodName = "PropCPU";
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };
#endif

#ifdef HigherBitPropGPU
	methodName = "PropGPU";
	ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, false, false };
#endif 

	

	//タイマー準備
	std::chrono::system_clock::time_point start, end;
	std::vector<double> timer(bits.size(), 0.0);

	//画像データ読み込み(複数)
//まずリストの読み込み
	std::string imageListFile = R"(E:\MATLAB\FastMedianFiltering\Experiments\HighPrecision\HP_List.txt)";
	std::string fileName;
	std::ifstream fs(imageListFile);

	int n = 0;

	while (!fs.eof())
	{
		n++;
		std::cout << n << " ";
		//ファイル名の読み込み
		fs >> fileName;

		//bit深度変化
		for (int i = 0; i < bits.size(); i++)
		{
			int bit = bits[i];
			int divScale = pow(2, 16 - bit);//入力16bit画像をスケーリングして疑似的に8～16bit画像を生成
			int Imax = pow(2, bit);

			//画像読み込み
			this->image = new Container_Image(fileName, convertFlag);
			this->image->load(false, cv::Size(0,0), divScale);


			//手法実行
#ifdef HigherBitTestConstantTime
			//constant time の実行
			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			start = std::chrono::system_clock::now();//タイマー作動
			ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			Utility::showDevice(this->image->result_device, sizeInfo, "result", false, divScale * 16, false);

#endif

#ifdef HigherBitPropList
			//提案法CPU　Listの実行

			//cv::imshow("Input", this->image->I32 * divScale * 16);

			start = std::chrono::system_clock::now();//タイマー作動
			FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			//cv::Mat result = FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//cv::imshow("Result1", result * divScale * 16);
			//cv::waitKey(0);


#endif

#ifdef HigherBitPropCPU
			//提案法CPU（MD sliding window）の実行

			//cv::Mat result, mat;
			start = std::chrono::system_clock::now();//タイマー作動
			FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result1", result * divScale * 16);
			//cv::waitKey(0);

#endif

#ifdef HigherBitPropGPU
			//提案法GPU（1D sliding window）の実行
			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


			//Utility::showDevice(this->image->I_device, sizeInfo, "input", false, 256, false);

			start = std::chrono::system_clock::now();//タイマー作動
			FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//タイマーに時間追加
			timer[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//Utility::showDevice(this->image->result_device, sizeInfo, "result", false, divScale * 16, false);
#endif


			delete this->image;

		}



	}


	//出力
	std::cout << std::endl;

	writing_file << methodName << std::endl;

	writing_file << "Radius = " << r << ", Bits = ";
	for (int i = 0; i < bits.size(); i++)
	{
		writing_file << bits[i] << ",";
	}
	writing_file << std::endl;

	for (int i = 0; i < timer.size(); i++)
	{
		std::cout << timer[i] / float(n) << "[ms/image]" << std::endl;
		writing_file << timer[i] / float(n) << "\t";
	}
	writing_file << std::endl;


}




//論文用2D multispectral image ノイズ除去テスト
//ノイズレベル固定
void Experimenter::noiseRemovalForMultispectralImageForPaper()
{
	//設定
	//float eps2 = 75.5f * 75.5f;
	//float eps2 = 10.0f * 10.0f;
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;
	//int r = 5;
	//int channelRadius = 0;
	int frameNum = 31;

	//データフォルダ
	std::string dataDir = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\createTestData\data\toy\)";

	//PSNRの保存
	std::string saveTimeName = dataDir + "result.txt";
	//結果画像の保存
	std::string saveImageDir = dataDir + R"(result\)";

	//ファイルの作成
	std::ofstream writing_file;
	writing_file.open(saveTimeName, std::ios::app);



	//2つめのtrueによりCV_8UC1を確保し、これをガイド画像の生成に用いる
	ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

	//ノイズ画像フォルダ
	std::string fileDirNoise = dataDir + R"(noise\)";
	//真値画像フォルダ
	std::string fileDirGT = dataDir + R"(img\)";

	//ノイズレベル（このプログラム上ではノイズは加えない。すでに加わっている画像を読み込む）
	std::vector<int> nSigrange = { 10, 30, 50, 100 };
	//std::vector<int> nSigrange = { 30 };

	//データ測定用変数
	//チャンネル半径
	std::vector<int> channelRs = {0,1,2,3,4,5,6,7,8,9 };
	//std::vector<int> channelRs = { 10,11,12,13,14,15 };
	//std::vector<int> channelRs = { 0,9 };
	//半径
	//std::vector<int> rs = { 2,3,4,5,6,7,8,9 };
	std::vector<int> rs = { 7 };
	//eps
	std::vector<float> eps2s = {75.5f * 75.5f };
	//std::vector<float> eps2s = { 25.5f * 25.5f };

	
	//真値
	//mat_8とmat_8_colorを確保
	ConvertImageFlag convertFlagGT = { true, false, false, true, false, false, false, false, false, false, false };
	std::string fileDirGT2 = fileDirGT + "img";
	Container_Video gt = Container_Video(fileDirGT2, convertFlagGT);
	gt.load(0, frameNum);

	int i = 1;//ノイズレベル30使用固定
	int sig = nSigrange[i];

	std::string fileDirNoise2 = fileDirNoise + std::to_string(i + 1) + "\\img";

	//画像読み込み チャンネルを動画フレームとして読み込み
	this->video = new Container_Video(fileDirNoise2, convertFlag);
	this->video->load(0, frameNum);
	SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1));

	for(float &eps2 : eps2s)
	{
		std::cout << "eps2 = " << eps2 << std::endl;
		//出力
		writing_file << "eps2 = " << eps2 << std::endl;
		for (int &channelRadius : channelRs)
		{
			//出力
			std::cout << "channelRadius = " << channelRadius << std::endl;
			writing_file << "channelRadius = " << channelRadius << std::endl;
			//出力
			std::cout << "r = ";
			writing_file << "r = ";
			for (int &r : rs)
				writing_file << r << "\t";
			writing_file << std::endl;
			for (int &r : rs)
			{
				std::cout << r << " ";

				double psnrAve = 0.0;
				//1channelごとに処理
				for (int f = 0; f < frameNum; f++)
				{
					std::string fileNumber = "";
					if (f < 10)
						fileNumber += "00" + std::to_string(f);
					else
						fileNumber += "0" + std::to_string(f);

					if (channelRadius == 0)
					{
						//1チャンネルそのまま処理
						int* result;
						UtilityForCUDA::allocateDeviceMemory(result, sizeInfo);
						FGMF3::filter2DGPU(this->video->I_device[f], this->video->G_device[f], result, r, eps2, Imax, sizeInfo);
						//FGMF3::filter2DGPU(this->video->I_device[f], this->video->G_device[f], result, r, eps2 * channelRadius, Imax, sizeInfo);
						//PSNR計算
						cv::Mat resultForPSNR = UtilityForCUDA::downloadLinearArrayAsMat(result, sizeInfo);
						double psnr = this->calculatePSNR(resultForPSNR, gt.I[f], Imax - 1);
						//std::cout << psnr << std::endl;
						psnrAve += psnr;

						//画像保存(フォルダ固定　一時的)
						//std::string saveImageName = saveImageDir + "2\\r" + std::to_string(channelRadius) + "\\img" + fileNumber + ".png";
						//cv::imwrite(saveImageName, resultForPSNR);

						cudaFree(result);
					}
					else
					{
						int startFrame = std::max(0, f - channelRadius);
						int endFrame = std::min(frameNum - 1, f + channelRadius);
						//videoから処理対象のIを1フレーム、Gをチャンネル半径内分、DeviceArray<int>として取得
						int* targetI = this->video->I_device[f];
						std::vector<cv::Mat> targetGsMat{ this->video->G32.begin() + startFrame, this->video->G32.begin() + endFrame + 1 };
						//cv::imshow("in",targetGsMat[0]*255);
						//cv::waitKey(0);

						DeviceArray<int>* targetGs = new DeviceArray<int>(targetGsMat, sizeInfo);
						//Utility::showDevice(targetGs->host[0], sizeInfo, "test", false, 255, true);
						int* result;
						UtilityForCUDA::allocateDeviceMemory(result, sizeInfo);

						FGMF3::filter2DGPU_MultiChannel<int>(targetI, targetGs, result, r, eps2, Imax, sizeInfo);
						//FGMF3::filter2DGPU_MultiChannel<int>(targetI, targetGs, result, r, eps2 * channelRadius, Imax, sizeInfo);

						//Utility::showDevice(targetI, sizeInfo, "Input", false, 255, false);
						//Utility::showDevice(result, sizeInfo, "Result", false, 255, false);

						//PSNR計算
						cv::Mat resultForPSNR = UtilityForCUDA::downloadLinearArrayAsMat(result, sizeInfo);
						double psnr = this->calculatePSNR(resultForPSNR, gt.I[f], Imax - 1);
						//std::cout << psnr << std::endl;
						psnrAve += psnr;

						//画像保存(フォルダ固定　一時的)
						//std::string saveImageName = saveImageDir + "2\\r" + std::to_string(channelRadius) + "\\img" + fileNumber + ".png";
						//cv::imwrite(saveImageName, resultForPSNR);



						delete targetGs;
						cudaFree(result);
					}

				}

				//PSNR平均
				psnrAve /= (double)frameNum;
				//std::cout << psnrAve << std::endl;

				writing_file << psnrAve << "\t";
			}
			std::cout << std::endl;
			writing_file << std::endl;
		}
		std::cout << std::endl;
		writing_file << std::endl;
	}
	std::cout << std::endl;
	writing_file << std::endl;
}

//Multispectral image denoising のjson使用版
void Experimenter::noiseRemovalForMultispectralImageForPaperNew()
{
	//設定
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	//bool doProp3 = false;
	//bool doProp3_ex = false;
	//bool doProp6 = false;
	//std::vector<bool> doProp = { true, true, true, true, true };//0,1,3,5,7
	std::vector<bool> doProp = { false, false, false, false, false };//0,1,3,5,7
	std::vector<int> channnelRadius_doProp = { 0,1,3,5,7 };
	std::vector<bool> doProp_ex = { false, false, false, false };//1,3,5,7
	std::vector<int> channnelRadius_doProp_ex = { 1,3,5,7 };
	bool do100times0 = false;
	bool do100times1 = false;
	bool doMedian = false;
	bool doCT = true;

	//設定jsonファイル
	//std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\New\list\dev_experiments.json)";
	//インパルスノイズ用
	//std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\New_Pepper\list\dev_experiments.json)";
	//int numOfDigit = 2;
	//PaviaU
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\New_remote\PaviaU\list\dev_experiments_6.json)";
	int frameNum = 103;
	//WDCM
	//std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\New_remote\WDCM\list\dev_experiments.json)";
	//int frameNum = 191;

	//チャンネル数に比例してepsを大きくするか
	bool epsIncrementalMode = true;
	//読み込まれるepsはint型か
	bool epsIsInt = true;


	int numOfDigit = 3;

	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);




	//ファイル名
	std::string inputFileName = "img";
	std::string resultFileName = "result";



	std::string methodName;

	std::cout << "Data Num: " << em.data.dataNames.size() << std::endl;

	//提案法　チャンネル自身含む
	for (int i = 0; i < doProp.size(); i++)
	{
		if (doProp[i])
		{
			//チャンネル半径
			int channelRadius = channnelRadius_doProp[i];
			methodName = "PropGPU_cr" + std::to_string(channelRadius);
			std::cout << methodName << std::endl;
			ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

			for (int i = 0; i < em.data.dataNames.size(); i++)
			{
				std::cout << " " << i + 1;
				std::string inputName = em.data.dataNames[i];
				std::string resultName = em.data.resultNames[i];

				std::string inputPath = inputName + "\\" + inputFileName;

				//画像読み込み チャンネルを動画フレームとして読み込み
				this->video = new Container_Video(inputPath, convertFlag);
				this->video->load(1, frameNum, numOfDigit);

				SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

				for (auto& param : em.methods.at(methodName).parameters)
				{
					float eps2;
					if (epsIsInt)
					{
						int eps = param.get<int>("eps");
						eps2 = eps * eps;
					}
					else
					{
						float eps = param.get<float>("eps");
						eps2 = eps * eps;
					}
					if (epsIncrementalMode)
					{
						eps2 *= (2 * channelRadius + 1);
					}

					int r = param.get<int>("radius");
					std::string id = param.get<std::string>("id");

					//1channelごとに処理
					for (int f = 0; f < frameNum; f++)
					{
						//チャンネル半径内使用フレーム開始終了番号(vector配列内の番号なので常に0始まり)
						int startFrame = std::max(0, f - channelRadius);
						int endFrame = std::min(frameNum - 1, f + channelRadius);
						//videoから処理対象のIを1フレーム、Gをチャンネル半径内分、DeviceArray<int>として取得
						int* targetI = this->video->I_device[f];
						std::vector<cv::Mat> targetGsMat{ this->video->G32.begin() + startFrame, this->video->G32.begin() + endFrame + 1 };

						DeviceArray<int>* targetGs = new DeviceArray<int>(targetGsMat, sizeInfo);
						int* result;
						UtilityForCUDA::allocateDeviceMemory(result, sizeInfo);

						FGMF3::filter2DGPU_MultiChannel<int>(targetI, targetGs, result, r, eps2, Imax, sizeInfo);

						cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(result, sizeInfo);
						resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

						std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
						std::ostringstream ss;
						ss << std::setw(2) << std::setfill('0') << (f + 1);
						std::string numStr(ss.str());
						cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);

						delete targetGs;
						cudaFree(result);
					}
#if debugTest == true
					break;
#endif
				}
				delete this->video;
#if debugTest == true
				break;
#endif
			}
			std::cout << std::endl;
		}
	}


	//提案法　チャンネル自身除く
	for (int i = 0; i < doProp_ex.size(); i++)
	{
		if (doProp_ex[i])
		{
			//チャンネル半径
			int channelRadius = channnelRadius_doProp_ex[i];
			methodName = "PropGPU_cr" + std::to_string(channelRadius) + "_ex";

			std::cout << methodName << std::endl;
			ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

			for (int i = 0; i < em.data.dataNames.size(); i++)
			{
				std::cout << " " << i + 1;
				std::string inputName = em.data.dataNames[i];
				std::string resultName = em.data.resultNames[i];

				std::string inputPath = inputName + "\\" + inputFileName;

				//画像読み込み チャンネルを動画フレームとして読み込み
				this->video = new Container_Video(inputPath, convertFlag);
				this->video->load(1, frameNum, numOfDigit);

				SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

				for (auto& param : em.methods.at(methodName).parameters)
				{
					float eps2;
					if (epsIsInt)
					{
						int eps = param.get<int>("eps");
						eps2 = eps * eps;
					}
					else
					{
						float eps = param.get<float>("eps");
						eps2 = eps * eps;
					}
					if (epsIncrementalMode)
					{
						eps2 *= (2 * channelRadius);
					}

					int r = param.get<int>("radius");
					std::string id = param.get<std::string>("id");



					//1channelごとに処理
					for (int f = 0; f < frameNum; f++)
					{
						//チャンネル半径内使用フレーム開始終了番号(vector配列内の番号なので常に0始まり)
						int startFrame = std::max(0, f - channelRadius);
						int endFrame = std::min(frameNum - 1, f + channelRadius);
						//videoから処理対象のIを1フレーム、Gをチャンネル半径内分、DeviceArray<int>として取得
						int* targetI = this->video->I_device[f];
						std::vector<cv::Mat> targetGsMat;
						for (int ff = startFrame; ff <= endFrame; ff++)
						{
							if (ff != f)
							{
								targetGsMat.push_back(this->video->G32[ff]);
							}
						}

						DeviceArray<int>* targetGs = new DeviceArray<int>(targetGsMat, sizeInfo);
						int* result;
						UtilityForCUDA::allocateDeviceMemory(result, sizeInfo);

						FGMF3::filter2DGPU_MultiChannel<int>(targetI, targetGs, result, r, eps2, Imax, sizeInfo);

						cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(result, sizeInfo);
						resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

						std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
						std::ostringstream ss;
						ss << std::setw(2) << std::setfill('0') << (f + 1);
						std::string numStr(ss.str());
						cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);

						delete targetGs;
						cudaFree(result);



					}


#if debugTest == true
					break;
#endif
				}

				delete this->video;
#if debugTest == true
				break;
#endif
			}
			std::cout << std::endl;
		}
	}


	//ConstantTime
	if (doCT)
	{
		methodName = "ConstantTime";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };
		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;

			//メモリ的にチャンネルを全部読み込むと無理っぽい。
			//画像として連続で処理するしかない。



			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				//1channelごとに処理
				for (int f = 0; f < frameNum; f++)
				{
					//画像読み込み
					std::ostringstream ss0;
					ss0 << std::setw(3) << std::setfill('0') << (f + 1);

					std::string dataPath = inputPath + ss0.str() + ".png";

					this->image = new Container_Image(dataPath, convertFlag);
					this->image->load();


					SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


					ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF, this->image->result_device, r, eps2, Imax, sizeInfo);

					cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::ostringstream ss;
					ss << std::setw(2) << std::setfill('0') << (f + 1);
					std::string numStr(ss.str());
					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);

					delete this->image;

					//cu_memoryInfo();
				}


#if debugTest == true
				break;
#endif
			}

#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}




	//100times0
	if (do100times0)
	{
		//チャンネル半径0
		methodName = "100times_cr0";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, true, false, false, true, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;

			//画像読み込み チャンネルを動画フレームとして読み込み
			this->video = new Container_Video(inputPath, convertFlag);
			this->video->load(1, frameNum, numOfDigit);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");



				//1channelごとに処理
				for (int f = 0; f < frameNum; f++)
				{
					//videoから処理対象のIを1フレーム、同じ物をGとして使用
					cv::Mat resultForEvaluation = l1Solver::filter(this->video->I32[f], this->video->I[f], r, sigma);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					std::ostringstream ss;
					ss << std::setw(2) << std::setfill('0') << (f + 1);
					std::string numStr(ss.str());
					cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);
				}


#if debugTest == true
				break;
#endif
			}
			delete this->video;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}


	//100times1
	if (do100times1)
	{
		//チャンネル半径1(=3チャンネル使用)
		methodName = "100times_cr1";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, true, false, false, true, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;

			//画像読み込み チャンネルを動画フレームとして読み込み
			this->video = new Container_Video(inputPath, convertFlag);
			this->video->load(1, frameNum, numOfDigit);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");



				//1channelごとに処理
				for (int f = 0; f < frameNum; f++)
				{
					//videoから処理対象のIを1フレーム
					//Gは周囲±1チャンネルでカラーと同じもの作るが、チャンネル方向折り返し境界処理をする
					cv::Mat G;
					std::vector<cv::Mat> Gs;
					if (f == 0)
					{
						Gs.push_back(this->video->I[f + 1]);
						Gs.push_back(this->video->I[f]);
						Gs.push_back(this->video->I[f + 1]);
					}
					else if (f == frameNum - 1)
					{
						Gs.push_back(this->video->I[f - 1]);
						Gs.push_back(this->video->I[f]);
						Gs.push_back(this->video->I[f - 1]);
					}
					else
					{
						Gs.push_back(this->video->I[f - 1]);
						Gs.push_back(this->video->I[f]);
						Gs.push_back(this->video->I[f + 1]);
					}
					cv::merge(Gs, G);
					cv::Mat resultForEvaluation = l1Solver::filter(this->video->I32[f], G, r, sigma);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					std::ostringstream ss;
					ss << std::setw(2) << std::setfill('0') << (f + 1);
					std::string numStr(ss.str());
					cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);
				}


#if debugTest == true
				break;
#endif
			}
			delete this->video;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}

	//median
	if (doMedian)
	{
		//チャンネル半径0
		methodName = "median";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, false, false, true, false, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;

			//画像読み込み チャンネルを動画フレームとして読み込み
			this->video = new Container_Video(inputPath, convertFlag);
			this->video->load(1, frameNum, numOfDigit);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				int r = param.get<int>("radius");
				int cr = param.get<int>("channelRadius");
				std::string id = param.get<std::string>("id");


				//1channelごとに処理
				for (int f = 0; f < frameNum; f++)
				{
					cv::Mat resultForEvaluation;
					if (cr == 0)
					{
						cv::medianBlur(this->video->I[f], resultForEvaluation, r * 2 + 1);
					}
					else
					{
						//提案法のガイドを均一の画像にすることで、medianの多チャンネル化対応とする
						// と思ったが、そうはならない。やる必要ないのか。
						// 
						//未実装
						/*
						//1channelごとに処理
						for (int f = 0; f < frameNum; f++)
						{
							//videoから処理対象のIを1フレーム、同じ物をGとして使用
							cv::Mat resultForEvaluation = l1Solver::filter(this->video->I32[f], this->video->I[f], r, sigma);
						}
						*/
					}
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					std::ostringstream ss;
					ss << std::setw(2) << std::setfill('0') << (f + 1);
					std::string numStr(ss.str());
					cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);
				}

#if debugTest == true
				break;
#endif
			}
			delete this->video;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}

}

/*
//異なるノイズレベルに対するテスト
void Experimenter::noiseRemovalForMultispectralImageForPaper()
{
	//設定
	float eps2 = 75.5f * 75.5f;
	//float eps2 = 10.0f * 10.0f;
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;
	int r = 5;
	int channelRadius = 1;
	int frameNum = 31;

	//データフォルダ
	std::string dataDir = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\createTestData\data\toy\)";

	//PSNRの保存
	std::string saveTimeName = dataDir + "result.txt";
	//結果画像の保存
	std::string saveImageDir = dataDir + R"(result\)";

	//ファイルの作成
	std::ofstream writing_file;
	writing_file.open(saveTimeName, std::ios::app);
	//出力
	writing_file << "Radius = " << r << ", ChannelRadius = " << channelRadius << std::endl;


	//2つめのtrueによりCV_8UC1を確保し、これをガイド画像の生成に用いる
	ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

	//ノイズ画像フォルダ
	std::string fileDirNoise = dataDir + R"(noise\)";
	//真値画像フォルダ
	std::string fileDirGT = dataDir + R"(img\)";

	//ノイズレベル（このプログラム上ではノイズは加えない。すでに加わっている画像を読み込む）
	std::vector<int> nSigrange = { 10, 30, 50, 100 };

	//真値
	//mat_8とmat_8_colorを確保
	ConvertImageFlag convertFlagGT = { true, false, false, true, false, false, false, false, false, false, false };
	std::string fileDirGT2 = fileDirGT + "img";
	Container_Video gt = Container_Video(fileDirGT2, convertFlagGT);
	gt.load(0, frameNum);

	for (int i = 0; i < nSigrange.size(); i++)
	{
		int sig = nSigrange[i];

		std::string fileDirNoise2 = fileDirNoise + std::to_string(i + 1) + "\\img";

		//画像読み込み チャンネルを動画フレームとして読み込み
		this->video = new Container_Video(fileDirNoise2, convertFlag);
		this->video->load(0, frameNum);
		SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1));

		double psnrAve = 0.0;
		std::cout << i << ": ";
		//1channelごとに処理
		for (int f = 0; f < frameNum; f++)
		{
			//std::cout << f << ": ";
			int startFrame = std::max(0, f - channelRadius);
			int endFrame = std::min(frameNum - 1, f + channelRadius);
			//videoから処理対象のIを1フレーム、Gをチャンネル半径内分、DeviceArray<int>として取得
			int* targetI = this->video->I_device[f];
			std::vector<cv::Mat> targetGsMat{ this->video->G32.begin() + startFrame, this->video->G32.begin() + endFrame + 1 };
			//cv::imshow("in",targetGsMat[0]*255);
			//cv::waitKey(0);

			DeviceArray<int>* targetGs = new DeviceArray<int>(targetGsMat, sizeInfo);
			//Utility::showDevice(targetGs->host[0], sizeInfo, "test", false, 255, true);
			int* result;
			Utility::allocateDeviceMemory(result, sizeInfo);

			FGMF3::filter2DGPU_MultiChannel<int>(targetI, targetGs, result, r, eps2, Imax, sizeInfo);

			//Utility::showDevice(targetI, sizeInfo, "Input", false, 255, false);
			//Utility::showDevice(result, sizeInfo, "Result", false, 255, false);

			//PSNR計算
			cv::Mat resultForPSNR = Utility::downloadLinearArrayAsMat(result, sizeInfo);
			double psnr = this->calculatePSNR(resultForPSNR, gt.I[f], Imax - 1);
			//std::cout << psnr << std::endl;
			psnrAve += psnr;

			delete targetGs;
			cudaFree(result);
		}

		//PSNR平均
		psnrAve /= (double)frameNum;
		std::cout << psnrAve << std::endl;

		writing_file << psnrAve << "\t";

	}
	writing_file << std::endl;
}
*/


void Experimenter::noiseRemovalForLightFieldForPaper()
{
	//設定
	float eps2 = 25.5f * 25.5f;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;
	int r = 5;
	int viewRadius = 1;
	int viewCols = 9;
	int viewRows = 9;


}

//4D ライトフィールドディスパリティリファインメント
void Experimenter::disparityRefinementForLightFieldForPaper()
{
	//設定
	//float eps2 = 25.5f * 25.5f;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;
	//int r = 15;
	//int viewRadius = 1;
	int viewCols = 9;
	int viewRows = 9;

	//std::vector<float> eps2s = { 25.5f * 25.5f, 55.5f * 55.5f, 75.5f * 75.5f};
	std::vector<float> eps2s = { 25.5f * 25.5f };
	std::vector<int> rs = { 17, 22, 27 };
	std::vector<int> viewRs = { 0, 1, 2, 3, 4 };
	//std::vector<int> viewRs = { 2, 3, 4 };



	std::string filePathSrc = R"(E:\LargeData\fastAllTest\DispAllViewInitialPNG_mishiba\antinous\img)";
	std::string filePathGuide = R"(E:\LargeData\LightField\dataset_evaluation\evaluation-toolkit-master\data\additional\antinous\input_Cam)";

	std::string saveDir = R"(E:\MATLAB\FastMedianFiltering\Experiments\LightFieldDisparityRefinement\result\antinous\)";


	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true };
	this->video = new Container_Video(filePathSrc, filePathGuide, convertFlag);

	this->video->load(0, 81);

	SizeInfo sizeInfo = SizeInfo(video->imageSize.width, video->imageSize.height, Imax, blockSize);






	for (float &eps2 : eps2s)
	{
		std::cout << "eps2 = " << eps2 << std::endl;
		for (int &r : rs)
		{
			std::cout << "spatial radius = " << r << std::endl;
			std::cout << "view radius = ";
			for (int &viewRadius : viewRs)
			{
				std::cout << viewRadius << " ";



				FGMF3::filter4DGPU(video->I_device, video->G_device, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);

				//パラメータ名
				std::string paramName = std::to_string(eps2) + "_" + std::to_string(r) + "_" + std::to_string(viewRadius);
				std::string saveDir2 = saveDir + paramName + "\\";


				//保存フォルダ生成
				struct stat statBuf;
				if (stat(saveDir2.c_str(), &statBuf) != 0) {
					
					if (_mkdir(saveDir2.c_str()) == 0) {
						std::cout << "Create output folder:" << saveDir2 << std::endl;
					}
					else {
						std::cout << "Fail to create output folder: " << saveDir2 << std::endl;
					}
					
				}



				for (int i = 0; i < 81; i++)
				{
					/*
					std::string numStr;
					if (i < 10)
						numStr = "00" + std::to_string(i);
					else
						numStr = "0" + std::to_string(i);
					*/
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << i;
					std::string numStr(ss.str());

					cv::Mat tmp = UtilityForCUDA::downloadLinearArrayAsMat(video->result_device[i], sizeInfo);
					std::string saveFileName = saveDir + paramName + "\\img" + numStr + ".png";
					cv::imwrite(saveFileName, tmp);
					//Utility::showDevice(video->result_device[i], sizeInfo, "result", true, 256);
				}
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

}


//4D ライトフィールドディスパリティリファインメント json版
void Experimenter::disparityRefinementForLightFieldForPaperNew()
{
	//設定
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	//std::vector<bool> doProps = { true, false, false, false, false };//0,1,2,3,4
	//std::vector<bool> doProp = { false, false, false, false, false };//0,1,2,3,4
	bool doProp = false;
	int doViewRadius = 2;
	bool do100times = false;
	bool doMedian = false;
	bool doCT = true;

	//設定jsonファイル
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\LightFieldDisparityRefinement\New\list\dev_experiments_2.json)";
	int viewCols = 9;
	int viewRows = 9;
	int viewNum = viewCols * viewRows;


	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);




	//ファイル名
	std::string inputFileName = "img";
	std::string guideFileName = "input_Cam";
	std::string resultFileName = "result";



	std::string methodName;

	std::cout << "Data Num: " << em.data.dataNames.size() << std::endl;

	/*
	//提案法 (r固定)
	methodName = "PropGPU";
	std::cout << methodName << std::endl;
	if (doProp)
	{
		ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];




			std::string inputPath = inputName + "\\" + inputFileName;
			std::string guidePath = inputName + "\\" + guideFileName;
			this->video = new Container_Video(inputPath, guidePath, convertFlag);
			this->video->load(0, 81);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int viewRadius = param.get<int>("angularRadius");
				int r = 18;//固定
				std::string id = param.get<std::string>("id");

				if (viewRadius == doViewRadius)
				{

					FGMF3::filter4DGPU(video->I_device, video->G_device, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);


					//全視点保存
					for (int f = 0; f < viewNum; f++)
					{
						cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(video->result_device[f], sizeInfo);
						std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
						std::ostringstream ss;
						ss << std::setw(3) << std::setfill('0') << f;
						std::string numStr(ss.str());
						cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);
					}

				}
#if debugTest == true
				break;
#endif
			}
			delete this->video;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}
	*/

	//提案法 (angularRadius固定)
	methodName = "PropGPU";
	std::cout << methodName << std::endl;
	if (doProp)
	{
		ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];




			std::string inputPath = inputName + "\\" + inputFileName;
			std::string guidePath = inputName + "\\" + guideFileName;
			this->video = new Container_Video(inputPath, guidePath, convertFlag);
			this->video->load(0, 81);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				int viewRadius = 2;//固定
				std::string id = param.get<std::string>("id");

				FGMF3::filter4DGPU(video->I_device, video->G_device, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);


				//全視点保存
				for (int f = 0; f < viewNum; f++)
				{
					cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(video->result_device[f], sizeInfo);
					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << f;
					std::string numStr(ss.str());
					cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);
				}

#if debugTest == true
				break;
#endif
			}
			delete this->video;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}


	//ConstantTime
	if (doCT)
	{
		methodName = "ConstantTime";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };
		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;


			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				//1視点ごとに処理
				for (int f = 0; f < viewNum; f++)
				{
					//画像読み込み
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << f;

					std::string dataPath = inputName + "\\" + inputFileName + ss.str() + ".png";
					std::string guidePath = inputName + "\\" + guideFileName + ss.str() + ".png";

					this->image = new Container_Image(dataPath, guidePath, convertFlag);
					this->image->load();


					SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


					ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF, this->image->result_device, r, eps2, Imax, sizeInfo);

					cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					cv::imwrite(resultPath + ss.str() + ".png", resultForEvaluation);

					delete this->image;

					//cu_memoryInfo();
				}


#if debugTest == true
				break;
#endif
			}

#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}




	//100times
	if (do100times)
	{
		methodName = "100times";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, true, false, false, true, false, false, false, false, false, false };


		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;


			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				//1視点ごとに処理
				for (int f = 0; f < viewNum; f++)
				{
					//画像読み込み
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << f;

					std::string dataPath = inputName + "\\" + inputFileName + ss.str() + ".png";
					std::string guidePath = inputName + "\\" + guideFileName + ss.str() + ".png";

					this->image = new Container_Image(dataPath, guidePath, convertFlag);
					this->image->load();


					SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


					cv::Mat resultForEvaluation = l1Solver::filter(this->image->I32, this->image->G, r, sigma);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					cv::imwrite(resultPath + ss.str() + ".png", resultForEvaluation);

					delete this->image;
				}
#if debugTest == true
				break;
#endif
			}

#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}



	//median
	if (doMedian)
	{
		methodName = "median";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, false, false, true, false, false, false, false, false, false, false };


		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;


			for (auto& param : em.methods.at(methodName).parameters)
			{
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				//1視点ごとに処理
				for (int f = 0; f < viewNum; f++)
				{
					//画像読み込み
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << f;

					std::string dataPath = inputName + "\\" + inputFileName + ss.str() + ".png";
					std::string guidePath = inputName + "\\" + guideFileName + ss.str() + ".png";

					this->image = new Container_Image(dataPath, guidePath, convertFlag);
					this->image->load();

					cv::Mat resultForEvaluation;
					cv::medianBlur(this->image->I, resultForEvaluation, r * 2 + 1);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					cv::imwrite(resultPath + ss.str() + ".png", resultForEvaluation);

					delete this->image;
				}
#if debugTest == true
				break;
#endif
			}
#if debugTest == true
			break;
#endif
		}

		std::cout << std::endl;
	}
}


//4D ライトフィールドディスパリティリファインメント json版 カラーガイド
void Experimenter::disparityRefinementForLightFieldForPaperNewColor()
{
	//設定
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	//std::vector<bool> doProps = { true, false, false, false, false };//0,1,2,3,4
	//std::vector<bool> doProp = { false, false, false, false, false };//0,1,2,3,4
	bool doProp = true;
	int doViewRadius = 4;
	bool do100times = false;
	bool doCT = false;
	bool doMedian = false;

	//設定jsonファイル
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\LightFieldDisparityRefinement\New\list\dev_experiments_6.json)";
	int viewCols = 9;
	int viewRows = 9;
	int viewNum = viewCols * viewRows;


	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);




	//ファイル名
	std::string inputFileName = "img";
	std::string guideFileName = "input_Cam";
	std::string resultFileName = "result";



	std::string methodName;

	std::cout << "Data Num: " << em.data.dataNames.size() << std::endl;

	
	//提案法 (r固定)
	methodName = "PropGPU";
	std::cout << methodName << std::endl;
	if (doProp)
	{
		ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];




			std::string inputPath = inputName + "\\" + inputFileName;
			std::string guidePath = inputName + "\\" + guideFileName;
			this->video = new Container_Video(inputPath, guidePath, convertFlag);
			this->video->load(0, 81);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int viewRadius = param.get<int>("angularRadius");
				int r = 20;//固定
				std::string id = param.get<std::string>("id");

				if (viewRadius == doViewRadius)
				{

					FGMF3::filter4DGPU(video->I_device, video->G_device_color, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);


					//全視点保存
					for (int f = 0; f < viewNum; f++)
					{
						cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(video->result_device[f], sizeInfo);
						std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
						std::ostringstream ss;
						ss << std::setw(3) << std::setfill('0') << f;
						std::string numStr(ss.str());
						cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);
					}

				}
#if debugTest == true
				break;
#endif
			}
			delete this->video;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}
	

	/*
	//提案法 (angularRadius固定)
	methodName = "PropGPU";
	std::cout << methodName << std::endl;
	if (doProp)
	{
		ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			if (i + 1 < 2)
			{
				//continue;
			}

			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];




			std::string inputPath = inputName + "\\" + inputFileName;
			std::string guidePath = inputName + "\\" + guideFileName;
			this->video = new Container_Video(inputPath, guidePath, convertFlag);
			this->video->load(0, 81);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			int tt = 0;
			for (auto& param : em.methods.at(methodName).parameters)
			{
				tt++;
				if (tt < 2 && (i + 1) ==2)
				{
					//continue;
				}
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				int viewRadius = 2;//固定
				std::string id = param.get<std::string>("id");

				FGMF3::filter4DGPU(video->I_device, video->G_device_color, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);


				//全視点保存
				for (int f = 0; f < viewNum; f++)
				{
					cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(video->result_device[f], sizeInfo);
					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << f;
					std::string numStr(ss.str());
					cv::imwrite(resultPath + numStr + ".png", resultForEvaluation);
				}
#if debugTest == true
				break;
#endif
			}
			delete this->video;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}
	*/

	//ConstantTime
	if (doCT)
	{
		methodName = "ConstantTime";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };
		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;


			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				//1視点ごとに処理
				for (int f = 0; f < viewNum; f++)
				{
					//画像読み込み
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << f;

					std::string dataPath = inputName + "\\" + inputFileName + ss.str() + ".png";
					std::string guidePath = inputName + "\\" + guideFileName + ss.str() + ".png";

					this->image = new Container_Image(dataPath, guidePath, convertFlag);
					this->image->load();


					SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


					ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF_color, this->image->result_device, r, eps2, Imax, sizeInfo);

					cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					cv::imwrite(resultPath + ss.str() + ".png", resultForEvaluation);

					delete this->image;

					//cu_memoryInfo();
				}


#if debugTest == true
				break;
#endif
			}

#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}




	//100times
	if (do100times)
	{
		methodName = "100times";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, true, false, false, true, false, false, false, false, false, false };


		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;


			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				//1視点ごとに処理
				for (int f = 0; f < viewNum; f++)
				{
					//画像読み込み
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << f;

					std::string dataPath = inputName + "\\" + inputFileName + ss.str() + ".png";
					std::string guidePath = inputName + "\\" + guideFileName + ss.str() + ".png";

					this->image = new Container_Image(dataPath, guidePath, convertFlag);
					this->image->load();


					SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


					cv::Mat resultForEvaluation = l1Solver::filter(this->image->I32, this->image->G_color, r, sigma);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					cv::imwrite(resultPath + ss.str() + ".png", resultForEvaluation);

					delete this->image;
				}
#if debugTest == true
				break;
#endif
			}

#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}



	//median
	if (doMedian)
	{
		methodName = "median";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, false, false, true, false, false, false, false, false, false, false };


		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;


			for (auto& param : em.methods.at(methodName).parameters)
			{
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				//1視点ごとに処理
				for (int f = 0; f < viewNum; f++)
				{
					//画像読み込み
					std::ostringstream ss;
					ss << std::setw(3) << std::setfill('0') << f;

					std::string dataPath = inputName + "\\" + inputFileName + ss.str() + ".png";
					std::string guidePath = inputName + "\\" + guideFileName + ss.str() + ".png";

					this->image = new Container_Image(dataPath, guidePath, convertFlag);
					this->image->load();

					cv::Mat resultForEvaluation;
					cv::medianBlur(this->image->I, resultForEvaluation, r * 2 + 1);
					resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

					std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
					cv::imwrite(resultPath + ss.str() + ".png", resultForEvaluation);

					delete this->image;
				}
#if debugTest == true
				break;
#endif
			}
#if debugTest == true
			break;
#endif
		}

		std::cout << std::endl;
	}

}

void Experimenter::flashNoFlashForPaper()
{
	int r = 15;
	float eps2 = 25.5f * 25.5f;
	int Imax = 256;
	int threadNum = 12;

	bool useResize = false;
	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true, false, false };

	std::string filePathSrc = R"(E:\ProgramCode\FastGuidedMedianFilter\image\img_flash\cave-noflash.bmp)";
	std::string filePathGuide = R"(E:\ProgramCode\FastGuidedMedianFilter\image\img_flash\cave-flash.bmp)";

	this->image = new Container_Image(filePathSrc, filePathGuide, convertFlag);

	this->image->load();
	SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1));



	FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);

	cv::Mat ourResult = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);
	cv::Mat convResult = l1Solver::filter(this->image->I32_color, this->image->G_color, r);

	ourResult.convertTo(ourResult, CV_8U);
	convResult.convertTo(convResult, CV_8U);

	cv::imshow("Ours", ourResult);
	cv::imshow("100+", convResult);
	cv::waitKey(0);

	std::string saveNameOurs = R"(E:\MATLAB\FastMedianFiltering\Experiments\FlashNoFlash\ours.png)";
	std::string saveNameConv = R"(E:\MATLAB\FastMedianFiltering\Experiments\FlashNoFlash\fwmf.png)";

	cv::imwrite(saveNameOurs, ourResult);
	cv::imwrite(saveNameConv, convResult);

}


//カラー画像のノイズ除去　PSNR、SSIM、EKI測定 (json版)
void Experimenter::noiseRemovalEvaluationForColorImage(std::string settingFileName)
{
	//設定
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	bool doProp = true;
	bool doCT = true;
	bool do100times = true;
	bool doMedian = false;


	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);




	ConvertImageFlag convertFlagForGT = { true, false, false, true, false, false, false, false, false, false, false };

#if debugTestScaling == true
	bool useScaling = false;
	cv::Size scalingSize = cv::Size(800, 600);
#else	
	bool useScaling = false;
	cv::Size scalingSize;
#endif


	//ファイルモード
	//auto fileMode = std::ios::trunc;//上書き保存

	std::string dataFileName = "input.png";
	std::string gtFileName = "gt.png";
	std::string resultFileName = "result.png";

	//ループの順は
	/*
	nSig
	 "PropGPU"
	  imgs
	   rs
		eps2s
	 "ConstantTime"
	  imgs
	   rs
		eps2s
	 "100times"
	  imgs
	   rs
		sigmas
	 "median"(単なるメジアン)
	  imgs
	   rs
		eps2

	   保存ファイル名は
	   ~\[nSig]\[手法名]_result.txt
	   で、内部の構造は
	   1行目rs
	   2行目eps or sigmas or 0
	   以降の行　rs*2行目の要素（なしの場合は1）個分、画像ごとに評価値を1行に並べる

	*/

	std::string fileNameGT;
	std::string fileNameNoised;
	std::string methodName;

	//Prop
	if (doProp)
	{
		methodName = "PropGPU";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, false, false };


		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::string dataName = em.data.dataNames[i];
			std::string gtName = em.data.gtNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dataPath = dataName + "\\" + dataFileName;
			std::string gtPath = gtName + "\\" + gtFileName;

			//ノイズ画像読み込み
			this->image = new Container_Image(dataPath, convertFlag);	this->image->load(useScaling, scalingSize);
			//真値画像読み込み
			Container_Image* gtImage = new Container_Image(gtPath, convertFlagForGT); gtImage->load(useScaling, scalingSize);

			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			//カラー（入力・ガイド共に）
			for (auto &param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
				cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
				/*
				float psnr = this->calculatePSNR(resultForEvaluation, gtImage->I_color, Imax - 1);
				float ssim = this->calculateSSIM(resultForEvaluation, gtImage->I_color);
				float eki = this->calculateEKI(resultForEvaluation, gtImage->I_color);
#if debugTest == true
				std::cout << psnr << " " << ssim << " " << eki << " ";
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
#endif
				std::map<std::string, float> resultValues;
				resultValues.emplace("PSNR", psnr);
				resultValues.emplace("SSIM", ssim);
				resultValues.emplace("EKI", eki);
				em.saveResultInJSON(resultName, methodName, id, "result.json", resultValues);
				*/
#if debugTest == true
				break;
#endif
			}
			delete this->image;
			delete gtImage;
#if debugTest == true
			break;
#endif
		}

		std::cout << std::endl;
	}

	//ConstantTime
	if (doCT)
	{
		methodName = "ConstantTime";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };
		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::string dataName = em.data.dataNames[i];
			std::string gtName = em.data.gtNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dataPath = dataName + "\\" + dataFileName;
			std::string gtPath = gtName + "\\" + gtFileName;


			//ノイズ画像読み込み
			this->image = new Container_Image(dataPath, convertFlag);	this->image->load(useScaling, scalingSize);
			//真値画像読み込み
			Container_Image* gtImage = new Container_Image(gtPath, convertFlagForGT); gtImage->load(useScaling, scalingSize);

			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			//カラー（入力・ガイド共に）

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				ConstantTimeWMF::filter2DGPU(this->image->I_device_color, this->image->G_deviceF_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
				cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
				/*
				float psnr = this->calculatePSNR(resultForEvaluation, gtImage->I_color, Imax - 1);
				float ssim = this->calculateSSIM(resultForEvaluation, gtImage->I_color);
				float eki = this->calculateEKI(resultForEvaluation, gtImage->I_color);
#if debugTest == true
				std::cout << psnr << " " << ssim << " " << eki << " ";
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
#endif
				std::map<std::string, float> resultValues;
				resultValues.emplace("PSNR", psnr);
				resultValues.emplace("SSIM", ssim);
				resultValues.emplace("EKI", eki);
				em.saveResultInJSON(resultName, methodName, id, "result.json", resultValues);
				*/
#if debugTest == true
				break;
#endif
			}

			delete this->image;
			delete gtImage;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}


	//100times
	if (do100times)
	{
		methodName = "100times";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };


		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::string dataName = em.data.dataNames[i];
			std::string gtName = em.data.gtNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dataPath = dataName + "\\" + dataFileName;
			std::string gtPath = gtName + "\\" + gtFileName;

			//ノイズ画像読み込み
			this->image = new Container_Image(dataPath, convertFlag);	this->image->load(useScaling, scalingSize);
			//真値画像読み込み
			Container_Image* gtImage = new Container_Image(gtPath, convertFlagForGT); gtImage->load(useScaling, scalingSize);


			//カラー（入力・ガイド共に）
			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				cv::Mat resultForEvaluation = l1Solver::filter(this->image->I32_color, this->image->G_color, r, sigma);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
				/*
				float psnr = this->calculatePSNR(resultForEvaluation, gtImage->I_color, Imax - 1);
				float ssim = this->calculateSSIM(resultForEvaluation, gtImage->I_color);
				float eki = this->calculateEKI(resultForEvaluation, gtImage->I_color);
#if debugTest == true
				std::cout << psnr << " " << ssim << " " << eki << " ";
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
#endif
				std::map<std::string, float> resultValues;
				resultValues.emplace("PSNR", psnr);
				resultValues.emplace("SSIM", ssim);
				resultValues.emplace("EKI", eki);
				em.saveResultInJSON(resultName, methodName, id, "result.json", resultValues);
				*/
#if debugTest == true
				break;
#endif
			}
			delete this->image;
			delete gtImage;
#if debugTest == true
			break;
#endif
		}

		std::cout << std::endl;
	}


	//median
	if (doMedian)
	{
		methodName = "median";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, false, false, true, false, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::string dataName = em.data.dataNames[i];
			std::string gtName = em.data.gtNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dataPath = dataName + "\\" + dataFileName;
			std::string gtPath = gtName + "\\" + gtFileName;

			//ノイズ画像読み込み
			this->image = new Container_Image(dataPath, convertFlag);	this->image->load(useScaling, scalingSize);
			//真値画像読み込み
			Container_Image* gtImage = new Container_Image(gtPath, convertFlagForGT); gtImage->load(useScaling, scalingSize);


			//カラー（入力・ガイド共に）
			for (auto& param : em.methods.at(methodName).parameters)
			{
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");


				cv::Mat resultForEvaluation;
				cv::medianBlur(this->image->I_color, resultForEvaluation, r * 2 + 1);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
				/*
				float psnr = this->calculatePSNR(resultForEvaluation, gtImage->I_color, Imax - 1);
				float ssim = this->calculateSSIM(resultForEvaluation, gtImage->I_color);
				float eki = this->calculateEKI(resultForEvaluation, gtImage->I_color);
#if debugTest == true
				std::cout << psnr << " " << ssim << " " << eki << " ";
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
#endif
				std::map<std::string, float> resultValues;
				resultValues.emplace("PSNR", psnr);
				resultValues.emplace("SSIM", ssim);
				resultValues.emplace("EKI", eki);
				em.saveResultInJSON(resultName, methodName, id, "result.json", resultValues);
				*/
#if debugTest == true
				break;
#endif
			}
			delete this->image;
			delete gtImage;
#if debugTest == true
			break;
#endif
		}

		std::cout << std::endl;
	}
	
}


//視差画像中央視点のリファインメント
void Experimenter::noiseRemovalEvaluationForDepthImageGuideColor()
{
	//設定
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	bool doProp = true;
	bool doCT = false;
	bool do100times = false;
	bool doMedian = false;

	//設定jsonファイル
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\list\dev_experiments.json)";
	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);


	//ファイル名
	std::string dispFileName = "img040.png";
	std::string colorFileName = "input_Cam040.png";
	std::string resultFileName = "result.png";



	std::string fileNameColor;
	std::string fileNameDisp;
	std::string methodName;


	std::cout << "Data Num: " << em.data.dataNames.size() << std::endl;
	//Prop
	if (doProp)
	{
		methodName = "PropGPU";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string dispName = em.data.dataNames[i];
			std::string colorName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dispPath = dispName + "\\" + dispFileName;
			std::string colorPath = colorName + "\\" + colorFileName;

			//推定視差画像を入力用、カラー画像をガイド用として読み込み
			this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				FGMF3::filter2DGPU(this->image->I_device, this->image->G_device_color, this->image->result_device, r, eps2, Imax, sizeInfo);
				cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
#if debugTest == true
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
				break;
#endif
			}
			delete this->image;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}


	//ConstantTime
	if (doCT)
	{
		methodName = "ConstantTime";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };
		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			if (i < -1)
			{
				continue;
			}

			std::string dispName = em.data.dataNames[i];
			std::string colorName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dispPath = dispName + "\\" + dispFileName;
			std::string colorPath = colorName + "\\" + colorFileName;

			//推定視差画像を入力用、カラー画像をガイド用として読み込み
			this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF_color, this->image->result_device, r, eps2, Imax, sizeInfo);
				cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
#if debugTest == true
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
				break;
#endif
			}
			delete this->image;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}




	//100times
	if (do100times)
	{
		methodName = "100times";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string dispName = em.data.dataNames[i];
			std::string colorName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dispPath = dispName + "\\" + dispFileName;
			std::string colorPath = colorName + "\\" + colorFileName;

			//推定視差画像を入力用、カラー画像をガイド用として読み込み
			this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				cv::Mat resultForEvaluation = l1Solver::filter(this->image->I32_color, this->image->G_color, r, sigma);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
#if debugTest == true
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
				break;
#endif
			}
			delete this->image;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}


	//median
	if (doMedian)
	{
		methodName = "median";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, false, false, true, false, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string dispName = em.data.dataNames[i];
			std::string colorName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dispPath = dispName + "\\" + dispFileName;
			std::string colorPath = colorName + "\\" + colorFileName;

			//推定視差画像を入力用、カラー画像をガイド用として読み込み
			this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

			for (auto& param : em.methods.at(methodName).parameters)
			{
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");


				cv::Mat resultForEvaluation;
				cv::medianBlur(this->image->I_color, resultForEvaluation, r * 2 + 1);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
#if debugTest == true
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
				break;
#endif
			}
			delete this->image;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}



}

//視差画像中央視点のリファインメント(ガイド画像グレースケース)
void Experimenter::noiseRemovalEvaluationForDepthImageGuideGray()
{
	//設定
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	bool doProp = true;
	bool doCT = true;
	bool do100times = true;
	bool doMedian = false;

	//設定jsonファイル
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\list\dev_experiments_3.json)";
	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);


	//ファイル名
	std::string dispFileName = "img040.png";
	std::string colorFileName = "input_Cam040.png";
	std::string resultFileName = "result.png";



	std::string fileNameColor;
	std::string fileNameDisp;
	std::string methodName;


	std::cout << "Data Num: " << em.data.dataNames.size() << std::endl;
	//Prop
	if (doProp)
	{
		methodName = "PropGPU";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string dispName = em.data.dataNames[i];
			std::string colorName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dispPath = dispName + "\\" + dispFileName;
			std::string colorPath = colorName + "\\" + colorFileName;

			//推定視差画像を入力用、カラー画像をガイド用として読み込み
			this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);
				cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
#if debugTest == true
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
				break;
#endif
			}
			delete this->image;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}


	//ConstantTime
	if (doCT)
	{
		methodName = "ConstantTime";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };
		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			if (i < -1)
			{
				continue;
			}

			std::string dispName = em.data.dataNames[i];
			std::string colorName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dispPath = dispName + "\\" + dispFileName;
			std::string colorPath = colorName + "\\" + colorFileName;

			//推定視差画像を入力用、カラー画像をガイド用として読み込み
			this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF, this->image->result_device, r, eps2, Imax, sizeInfo);
				cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
#if debugTest == true
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
				break;
#endif
			}
			delete this->image;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}




	//100times
	if (do100times)
	{
		methodName = "100times";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string dispName = em.data.dataNames[i];
			std::string colorName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dispPath = dispName + "\\" + dispFileName;
			std::string colorPath = colorName + "\\" + colorFileName;

			//推定視差画像を入力用、カラー画像をガイド用として読み込み
			this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				cv::Mat resultForEvaluation = l1Solver::filter(this->image->I32, this->image->G, r, sigma);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
#if debugTest == true
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
				break;
#endif
			}
			delete this->image;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}


	//median
	if (doMedian)
	{
		methodName = "median";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, false, false, true, false, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string dispName = em.data.dataNames[i];
			std::string colorName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string dispPath = dispName + "\\" + dispFileName;
			std::string colorPath = colorName + "\\" + colorFileName;

			//推定視差画像を入力用、カラー画像をガイド用として読み込み
			this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

			for (auto& param : em.methods.at(methodName).parameters)
			{
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");


				cv::Mat resultForEvaluation;
				cv::medianBlur(this->image->I_color, resultForEvaluation, r * 2 + 1);
				resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

				std::string resultPath = em.getResultPath(resultName, methodName, id, resultFileName);
				cv::imwrite(resultPath, resultForEvaluation);
#if debugTest == true
				cv::imshow(methodName, resultForEvaluation);
				cv::waitKey(0);
				break;
#endif
			}
			delete this->image;
#if debugTest == true
			break;
#endif
		}
		std::cout << std::endl;
	}



}
void Experimenter::testForConstantTimeWMF()
{
	//設定
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;




	std::string fileNameColor;
	std::string fileNameDisp;
	std::string methodName;




	//ConstantTime
	methodName = "ConstantTime";
	std::cout << methodName << std::endl;
	ConvertImageFlag convertFlag = { false, false, false, false, false, false, false, true, true, true, true };

	std::string dispPath = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\dataset\result\antinous\ConstantTime\5_3\result.png)";
	std::string colorPath = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\dataset\input\antinous\input_Cam040.png)";

	//推定視差画像読み込み
	this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

	SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

	float eps = 25.5f;
	float eps2 = eps * eps;
	int r = 10;

	//ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF_color, this->image->result_device, r, eps2, Imax, sizeInfo);
	ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF, this->image->result_device, r, eps2, Imax, sizeInfo);
	cv::Mat resultForEvaluation = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
	resultForEvaluation.convertTo(resultForEvaluation, CV_8U);

	cv::imshow(methodName, resultForEvaluation);
	cv::waitKey(0);
	delete this->image;
	std::cout << std::endl;
}

void Experimenter::testForProp4DColor()
{
	//設定
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	int viewCols = 9;
	int viewRows = 9;
	int viewNum = viewCols * viewRows;

	std::string fileNameColor;
	std::string fileNameDisp;
	std::string methodName;


	//Prop
	methodName = "Prop";
	std::cout << methodName << std::endl;
	ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

	std::string dispPath = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\dataset\input\antinous\img040.png)";
	std::string colorPath = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\dataset\input\antinous\input_Cam040.png)";

	std::string gtPath = R"(E:\LargeData\LightField\dataset_evaluation\evaluation-toolkit-master\data\additional\antinous\gt_disp_Cam040.png)";
	cv::Mat gt = cv::imread(gtPath, 0);
	//推定視差画像読み込み
	this->image = new Container_Image(dispPath, colorPath, convertFlag);	this->image->load();

	SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

	float eps = 25.5f;
	float eps2 = eps * eps;
	int r = 10;

	int viewRadius = 1;

	FGMF3::filter2DGPU(this->image->I_device, this->image->G_device_color, this->image->result_device, r, eps2, Imax, sizeInfo);
	cv::Mat result2D = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
	result2D.convertTo(result2D, CV_8U);
	std::cout << Experimenter::calculatePSNR(gt, result2D, 256) << std::endl;


	std::string inputName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\dataset\input\antinous)";
	std::string inputFileName = "img";
	std::string guideFileName = "input_Cam";
	std::string inputPath = inputName + "\\" + inputFileName;
	std::string guidePath = inputName + "\\" + guideFileName;
	this->video = new Container_Video(inputPath, guidePath, convertFlag);
	this->video->load(0, 81);
	FGMF3::filter4DGPU(video->I_device, video->G_device_color, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);
	cv::Mat result4D = UtilityForCUDA::downloadLinearArrayAsMat(this->video->result_device[40], sizeInfo);
	result4D.convertTo(result4D, CV_8U);


	//std::cout << Experimenter::calculatePSNR(result2D, result4D, 256) << std::endl;
	std::cout << Experimenter::calculatePSNR(gt, result4D, 256) << std::endl;


	cv::imshow("2D", result2D);
	cv::imshow("4D", result4D);
	cv::waitKey(0);
	delete this->image;
	delete this->video;
	std::cout << std::endl;
}

///////////////////////
//論文用


//100+とかでガイド色を圧縮したときの影響が出るか見る （⇒でなかった）
void Experimenter::colorQuantizationDifference()
{
	float eps2 = 25.5f * 25.5f;
	int Imax = 256;
	int r = 15;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);


	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true, false, false };



	cv::Size sz = cv::Size(900, 600);

	//差が出る画像を探す用

	//画像データ読み込み(複数)
	//まずリストの読み込み
	std::string imageListFile = R"(E:\MATLAB\FastMedianFiltering\Experiments\Speed Test\HD_List.txt)";
	std::string fileName;
	std::ifstream fs(imageListFile);


	int n = 0;

	while (!fs.eof())
	{
		n++;
		//ファイル名の読み込み
		fs >> fileName;
		std::cout << fileName << std::endl;


		this->image = new Container_Image(fileName, convertFlag);
		this->image->load(true, sz);
		//this->image->load();
		SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


		//提案法GPU
		FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);


		//100+
		cv::Mat result = l1Solver::filter(this->image->I32_color, this->image->G_color, r);

		cv::Mat resultProp = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);

		cv::imshow("100+", result * 255.0f);
		cv::imshow("Prop", resultProp * 255.0f);
		cv::waitKey(0);
	}
}

//fminとmedian trackingで差が生じる画素％を計算する
void Experimenter::trackingDifference1()
{
	float eps2 = 25.5f * 25.5f;
	int Imax = 256;
	int r = 7;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int threadNum = 12;


	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };


	//cv::Size sz = cv::Size(900, 600);

	//画像データ読み込み(複数)
	//まずリストの読み込み
	std::string imageListFile = R"(E:\MATLAB\FastMedianFiltering\Experiments\Speed Test\HD_List.txt)";
	std::string fileName;
	std::ifstream fs(imageListFile);


	int n = 0;

	int numOfDif = 0;
	int totalPix = 0;

	while (!fs.eof())
	{
		n++;
		//ファイル名の読み込み
		fs >> fileName;
		//std::cout << fileName << std::endl;


		this->image = new Container_Image(fileName, convertFlag);
		//this->image->load(true, sz);
		this->image->load();
		SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

		//提案法CPU（median tracking）
		cv::Mat result_tracking = FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);

		//提案法List（fminから探す）
		cv::Mat result_fmin = FGMF1::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
		
		//差分
		cv::Mat dif;
		cv::absdiff(result_tracking, result_fmin, dif);
		//差のある画素数を計算
		std::vector<cv::Mat> planes;
		cv::split(dif, planes);
		dif = (planes[0] + planes[1] + planes[2]);
		//int num = (int)(cv::sum(dif)[0]);
		int num = cv::countNonZero(dif);

		std::cout << num << " ";

		totalPix += this->image->imageSize.width * this->image->imageSize.height;
		numOfDif += num;

		//cv::imshow("dif", dif * 255.0f * 255.0f);
		//cv::imshow("tracking", result_tracking * 255.0f);
		//cv::imshow("fmin", result_fmin * 255.0f);
		//cv::waitKey(0);
	}

	std::cout << std::endl << numOfDif << " / " << totalPix << std::endl;
}

















std::string Experimenter::getConfigFilePath(std::string fileNameWithoutExtension)
{
	//プログラム実行ディレクトリ取得
	// 実行ファイルのパス
	std::string modulePath = "";
	/*
	// ドライブ名、ディレクトリ名、ファイル名、拡張子
	char path[MAX_PATH], drive[MAX_PATH], dir[MAX_PATH], fname[MAX_PATH], ext[MAX_PATH];

	// 実行ファイルのファイルパスを取得
	if (::GetModuleFileNameA(NULL, path, MAX_PATH) != 0)
	{
		// ファイルパスを分割
		::_splitpath_s(path, drive, dir, fname, ext);
		// ドライブとディレクトリ名を結合して実行ファイルパスとする
		modulePath = std::string(drive) + std::string(dir);
	}

	//そのディレクトリにconfigフォルダがあるか
	std::string currentPath = modulePath;
	std::string searchFileName;

	while (true)
	{
		searchFileName = currentPath + "config\\" + fileNameWithoutExtension + ".json";
		std::ifstream ifs(searchFileName);
		if (ifs.is_open())
		{
			return searchFileName;
		}
		else
		{
			//なければ何度か以下を実行
			//一つ上のディレクトリにconfigフォルダあるか
			currentPath.erase(--currentPath.end());
			int last_pos = currentPath.find_last_of('\\');
			currentPath = currentPath.erase(last_pos + 1);
			if (currentPath == "")
			{
				//見つからず
				std::cerr << "Can not find config\\default.json\n";
				exit(EXIT_FAILURE);
			}
		}
	}
	*/
	return modulePath;
}

void Experimenter::loadSettings(std::string settingFileName)
{
	//jsonファイルから設定読み込み
	boost::property_tree::ptree pt;
	boost::property_tree::read_json(settingFileName, pt);

	std::string baseStr;



}
