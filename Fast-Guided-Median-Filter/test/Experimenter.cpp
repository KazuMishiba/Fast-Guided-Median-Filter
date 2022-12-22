#include "Experimenter.h"


#define debugTest false
#define debugTestScaling false



//��������̃e�X�g�Ɏg��
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

	std::cout << "h*w= " << this->image->imageSize.height << "�~" << this->image->imageSize.width << std::endl;
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
	//���`�����l���e�X�g
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

	//list g3�e�X�g
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



	//higher bit �ǂݍ��݃e�X�g
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







	
	//�e���v���[�g�W�J�m�F�p�R�[�h
	int r = radius;
	int radius_depth = r;

	//��Ė@CPU
	FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
	FGMF::filter2DInterface(this->image->I32, this->image->G32_color, threadNum, r, eps2, Imax);
	FGMF::filter2DInterface(this->image->I32_color, this->image->G32, threadNum, r, eps2, Imax);
	FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
	//��Ė@Window
	//2D
	FGMF2::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
	FGMF2::filter2DInterface(this->image->I32, this->image->G32_color, threadNum, r, eps2, Imax);
	FGMF2::filter2DInterface(this->image->I32_color, this->image->G32, threadNum, r, eps2, Imax);
	FGMF2::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
	//3D
	FGMF2::filter3DI1<gSum, fgSumUpToIndex, fg, int, float>(this->video->I32, this->video->G32, threadNum, radius, radius_depth, eps2, Imax);

	//��Ė@GPU
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

	//��Ė@ linked list
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

	std::cout << "h*w= " << this->image->imageSize.height << "�~" << this->image->imageSize.width << std::endl;
	std::cout << "Radius: " << radius << std::endl;

	//�򉻉摜�̓ǂݍ���
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

	std::cout << "h*w= " << sz.height << "�~" << sz.width << std::endl;
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



	//�]���@
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

	//��Ė@CPU
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
	//��Ė@CPU window
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
	
	//��Ė@GPU
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
				std::chrono::system_clock::time_point  start, end; // �^�� auto �ŉ�
				start = std::chrono::system_clock::now(); // �v���J�n����
				gmf_gpu.filterWithConstantTime(this->I32, this->G32F, r, eps2, Imax);
				end = std::chrono::system_clock::now();  // �v���I������
				time4 += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); //�����ɗv�������Ԃ��~���b�ɕϊ�
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
//�m�C�Y�����̎��{
void Experimenter::performNoiseReduction2D()
{
	//���Ԍv���p
	std::chrono::system_clock::time_point  start, end;


	if (true)
	{
		//�Ⴆ�΂����@�ɂ���
		//for (const auto& fn : this->fileNames) 
		{
			//�^�l�摜�ǂݍ���
			cv::Mat raw = cv::imread(fileName);
			raw.convertTo(raw, CV_32S);
			//�m�C�Y�t�^�摜�ǂݍ��݁@gauss, implus,  sigma,
			cv::Mat I = cv::imread(fileName);

			//�v�Z���ԑ���J�n����
			start = std::chrono::system_clock::now(); // �v���J�n����

			//�������� (���͉摜���K�C�h�Ƃ��ėp����)
			int threadNum = 12;
			cv::Mat result = FGMF::filter2DI3G1_MultiThread(I, I, threadNum, this->param_fgmf.radius, this->param_fgmf.eps2, this->param_fgmf.Imax);

			//�v�Z���ԑ���I������
			end = std::chrono::system_clock::now();  // �v���I������
			double calculationTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//�㏈������
			this->postProcessForNoiseReduction2D(raw, result, calculationTime, fileName);

			//���̉摜��
		}
	}
}

void Experimenter::postProcessForNoiseReduction2D(const cv::Mat& raw, const cv::Mat& result, double calculationTime, std::string fileName)
{
	std::string saveFileName, windowName;
	//�]���w�W���v�Z����
	//double psnr = calculatePSNR(raw, result);
	//�ۑ�����
	//saveResult(result, saveFileName);
	//�v�Z���ԕۑ�

	//�\������
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
	z1 = ��(d1 - mean(d1))(d2 - mean(d2))
	z2 = ��(d1 - mean(d1))^2 ��(d2 - mean(d2))^2
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

//����ɑ΂���P�Ȃ�t�B���^�����O
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

//���a�ω����@off�Ȃ�摜�T�C�Y�ω�
//#define performRadiusChange
//�_���p2D 8bit�摜���x�e�X�g
void Experimenter::speedTest2D8bitForPaper()
{
	//�e��@�̐؂�ւ���#if�@�X�C�b�`�ōs�� OpenMP�g��Ȃ���@�������ꍇ��OpenMP�̃X�C�b�`���̂�؂��Ă������Ƃɒ���

	/*
	�����菇
	threadNum=1;
	OpenMP�X�C�b�`�؂�
	SpeedTest...��1���ς��Ď��s
	#define performRadiusChange�@��L���ɂ��Ă܂�SpeedTest...��1���ς��Ď��s
	threadNum=12
	OpenMP�X�C�b�`�����
	#define performRadiusChange�@�𖳌��ɂ���
	#define SpeedTestFaster, #define SpeedTestPropCPU��1�����s
	#define performRadiusChange�@��L���ɂ��Ă܂�1���ς��Ď��s

	*/

	//�ݒ�
	float eps2 = 25.5f * 25.5f;
	int Imax = 256 * 1;
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);


	//�v�����Ԃ̕ۑ�
	std::string saveTimeName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Speed Test\result.txt)";
	//�t�@�C���̍쐬
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



	//�^�C�}�[����
	std::chrono::system_clock::time_point start, end;
#ifdef performRadiusChange
	std::vector<double> timer_gray(rs.size(), 0.0);
	std::vector<double> timer_color(rs.size(), 0.0);
#else
	std::vector<double> timer_gray(sizes.size(), 0.0);
	std::vector<double> timer_color(sizes.size(), 0.0);
#endif

	//�摜�f�[�^�ǂݍ���(����)
	//�܂����X�g�̓ǂݍ���
	std::string imageListFile = R"(E:\MATLAB\FastMedianFiltering\Experiments\Speed Test\HD_List.txt)";
	std::string fileName;
	std::ifstream fs(imageListFile);

	int n = 0;

	while (!fs.eof())
	{
		n++;
		std::cout << n << " ";
		//�t�@�C�����̓ǂݍ���
		fs >> fileName;
		//std::cout << fileName << std::endl;

#ifdef performRadiusChange
		//���a�ω�
		for (int i = 0; i < rs.size(); i++)
		{
			int r = rs[i];
#else
		//�摜�T�C�Y�ω�
		for (int i = 0; i < sizes.size(); i++)
		{
			cv::Size sz = sizes[i];
#endif

			//�摜�ǂݍ��݁A�T�C�Y�w��
			this->image = new Container_Image(fileName, convertFlag);
			this->image->load(true, sz);


			//��@���s
#ifdef SpeedTestConstantTime
			//constant time�̃t�B���^�ɂ̓K�C�f�b�h�t�B���^��p���Ă���A���̏ꍇ�A���S���Y���̓�����A���a���w�肵���Ƃ��ɂ͎��ۂɎg�p����锼�a�͂���2�{�ɂȂ�
			//����ŃK�C�f�b�h�t�B���^�̘_���ł̓{�b�N�X�t�B���^�̔��a���t�B���^�̔��a�ƌĂ�ł�����ۂ��̂ŁA���̂܂܎g��
			//cu_memoryInfo();
			//constant time �̎��s
			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			//�O���[�X�P�[���i���́E�K�C�h���Ɂj
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//I3G1 test
			//ConstantTimeWMF::filter2DGPU(this->image->I_device_color, this->image->G_deviceF, this->image->result_device_color, r, eps2, Imax, sizeInfo);
			//I1G3 test
			//ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF_color, this->image->result_device, r, eps2, Imax, sizeInfo);
			//Utility::showDevice(this->image->result_device, sizeInfo, "result", false, 256, false);
			//�J���[�i���́E�K�C�h���Ɂj
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			ConstantTimeWMF::filter2DGPU(this->image->I_device_color, this->image->G_deviceF_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//Utility::showDevice(this->image->result_device_color, sizeInfo, "result", false, 256, false);

#endif // SpeedTestConstantTime

#ifdef SpeedTestFaster
			//100+times faster WMF�̎��s

			//cv::Mat result, mat;
			//�O���[�X�P�[���i���́E�K�C�h���Ɂj
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			l1Solver::filter(this->image->I32, this->image->G, r);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result1", mat);

			//�J���[�i���́E�K�C�h���Ɂj
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			l1Solver::filter(this->image->I32_color, this->image->G_color, r);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result2", mat);
			//cv::waitKey(0);

#endif // SpeedTestFaster

#ifdef SpeedTestPropCPU
			//��Ė@CPU�iMD sliding window�j�̎��s

			//�O���[�X�P�[���i���́E�K�C�h���Ɂj
			//cv::Mat result, mat;
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result1", mat);
			//�J���[�i���́E�K�C�h���Ɂj
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result2", mat);
			//cv::waitKey(0);

#endif // SpeedTestPropCPU

#ifdef SpeedTestPropList
			//��Ė@CPU�iList�j�̎��s

			//�O���[�X�P�[���i���́E�K�C�h���Ɂj
			//cv::Mat result, mat;
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			//result = 
			FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			/*
			result.convertTo(mat, CV_8U);
			cv::imshow("Result1", mat);
			cv::waitKey(0);
			*/
			//�J���[�i���́E�K�C�h���Ɂj
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			//result = 
			FGMF1::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			/*
			result.convertTo(mat, CV_8U);
			cv::imshow("Result2", mat);
			cv::waitKey(0);
			*/
#endif // SpeedTestPropList

#ifdef SpeedTestPropGPU
			//��Ė@GPU�i1D sliding window�j�̎��s
			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			//Utility::showDevice(this->image->I_device, sizeInfo, "input", false, 256, false);

			//�O���[�X�P�[���i���́E�K�C�h���Ɂj
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_gray[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//Utility::showDevice(this->image->result_device, sizeInfo, "result", false, 256, false);

			//�J���[�i���́E�K�C�h���Ɂj
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer_color[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//Utility::showDevice(this->image->result_device_color, sizeInfo, "result", false, 256, false);

#endif // SpeedTestPropGPU


		
			

			delete this->image;

		}



	}

	//�o��
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

//�_���p2D higher bit�摜���x�e�X�g
void Experimenter::speedTest2DHigherBitForPaper()
{
	//�e��@�̐؂�ւ���#if�@�X�C�b�`�ōs�� OpenMP�g��Ȃ���@�������ꍇ��OpenMP�̃X�C�b�`���̂�؂��Ă������Ƃɒ���


	//�ݒ�
	float eps2 = 25.5f * 25.5f;
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int r = 7;

	//propGPU��12�܂ł������삵�Ȃ�
	//constant time��8�r�b�g�ł��������Ȃ�����
	//propCPU��List�͑S������
	std::vector<int> bits = { 8,10,12,14,16 };
	//std::vector<int> bits = { 10 };


	//�v�����Ԃ̕ۑ�
	std::string saveTimeName = R"(E:\MATLAB\FastMedianFiltering\Experiments\HighPrecision\result.txt)";
	//�t�@�C���̍쐬
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

	

	//�^�C�}�[����
	std::chrono::system_clock::time_point start, end;
	std::vector<double> timer(bits.size(), 0.0);

	//�摜�f�[�^�ǂݍ���(����)
//�܂����X�g�̓ǂݍ���
	std::string imageListFile = R"(E:\MATLAB\FastMedianFiltering\Experiments\HighPrecision\HP_List.txt)";
	std::string fileName;
	std::ifstream fs(imageListFile);

	int n = 0;

	while (!fs.eof())
	{
		n++;
		std::cout << n << " ";
		//�t�@�C�����̓ǂݍ���
		fs >> fileName;

		//bit�[�x�ω�
		for (int i = 0; i < bits.size(); i++)
		{
			int bit = bits[i];
			int divScale = pow(2, 16 - bit);//����16bit�摜���X�P�[�����O���ċ^���I��8�`16bit�摜�𐶐�
			int Imax = pow(2, bit);

			//�摜�ǂݍ���
			this->image = new Container_Image(fileName, convertFlag);
			this->image->load(false, cv::Size(0,0), divScale);


			//��@���s
#ifdef HigherBitTestConstantTime
			//constant time �̎��s
			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			ConstantTimeWMF::filter2DGPU(this->image->I_device, this->image->G_deviceF, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			Utility::showDevice(this->image->result_device, sizeInfo, "result", false, divScale * 16, false);

#endif

#ifdef HigherBitPropList
			//��Ė@CPU�@List�̎��s

			//cv::imshow("Input", this->image->I32 * divScale * 16);

			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			//cv::Mat result = FGMF1::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//cv::imshow("Result1", result * divScale * 16);
			//cv::waitKey(0);


#endif

#ifdef HigherBitPropCPU
			//��Ė@CPU�iMD sliding window�j�̎��s

			//cv::Mat result, mat;
			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//result.convertTo(mat, CV_8U);
			//cv::imshow("Result1", result * divScale * 16);
			//cv::waitKey(0);

#endif

#ifdef HigherBitPropGPU
			//��Ė@GPU�i1D sliding window�j�̎��s
			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


			//Utility::showDevice(this->image->I_device, sizeInfo, "input", false, 256, false);

			start = std::chrono::system_clock::now();//�^�C�}�[�쓮
			FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);
			end = std::chrono::system_clock::now();//�^�C�}�[�Ɏ��Ԓǉ�
			timer[i] += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			//Utility::showDevice(this->image->result_device, sizeInfo, "result", false, divScale * 16, false);
#endif


			delete this->image;

		}



	}


	//�o��
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




//�_���p2D multispectral image �m�C�Y�����e�X�g
//�m�C�Y���x���Œ�
void Experimenter::noiseRemovalForMultispectralImageForPaper()
{
	//�ݒ�
	//float eps2 = 75.5f * 75.5f;
	//float eps2 = 10.0f * 10.0f;
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;
	//int r = 5;
	//int channelRadius = 0;
	int frameNum = 31;

	//�f�[�^�t�H���_
	std::string dataDir = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\createTestData\data\toy\)";

	//PSNR�̕ۑ�
	std::string saveTimeName = dataDir + "result.txt";
	//���ʉ摜�̕ۑ�
	std::string saveImageDir = dataDir + R"(result\)";

	//�t�@�C���̍쐬
	std::ofstream writing_file;
	writing_file.open(saveTimeName, std::ios::app);



	//2�߂�true�ɂ��CV_8UC1���m�ۂ��A������K�C�h�摜�̐����ɗp����
	ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

	//�m�C�Y�摜�t�H���_
	std::string fileDirNoise = dataDir + R"(noise\)";
	//�^�l�摜�t�H���_
	std::string fileDirGT = dataDir + R"(img\)";

	//�m�C�Y���x���i���̃v���O������ł̓m�C�Y�͉����Ȃ��B���łɉ�����Ă���摜��ǂݍ��ށj
	std::vector<int> nSigrange = { 10, 30, 50, 100 };
	//std::vector<int> nSigrange = { 30 };

	//�f�[�^����p�ϐ�
	//�`�����l�����a
	std::vector<int> channelRs = {0,1,2,3,4,5,6,7,8,9 };
	//std::vector<int> channelRs = { 10,11,12,13,14,15 };
	//std::vector<int> channelRs = { 0,9 };
	//���a
	//std::vector<int> rs = { 2,3,4,5,6,7,8,9 };
	std::vector<int> rs = { 7 };
	//eps
	std::vector<float> eps2s = {75.5f * 75.5f };
	//std::vector<float> eps2s = { 25.5f * 25.5f };

	
	//�^�l
	//mat_8��mat_8_color���m��
	ConvertImageFlag convertFlagGT = { true, false, false, true, false, false, false, false, false, false, false };
	std::string fileDirGT2 = fileDirGT + "img";
	Container_Video gt = Container_Video(fileDirGT2, convertFlagGT);
	gt.load(0, frameNum);

	int i = 1;//�m�C�Y���x��30�g�p�Œ�
	int sig = nSigrange[i];

	std::string fileDirNoise2 = fileDirNoise + std::to_string(i + 1) + "\\img";

	//�摜�ǂݍ��� �`�����l���𓮉�t���[���Ƃ��ēǂݍ���
	this->video = new Container_Video(fileDirNoise2, convertFlag);
	this->video->load(0, frameNum);
	SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1));

	for(float &eps2 : eps2s)
	{
		std::cout << "eps2 = " << eps2 << std::endl;
		//�o��
		writing_file << "eps2 = " << eps2 << std::endl;
		for (int &channelRadius : channelRs)
		{
			//�o��
			std::cout << "channelRadius = " << channelRadius << std::endl;
			writing_file << "channelRadius = " << channelRadius << std::endl;
			//�o��
			std::cout << "r = ";
			writing_file << "r = ";
			for (int &r : rs)
				writing_file << r << "\t";
			writing_file << std::endl;
			for (int &r : rs)
			{
				std::cout << r << " ";

				double psnrAve = 0.0;
				//1channel���Ƃɏ���
				for (int f = 0; f < frameNum; f++)
				{
					std::string fileNumber = "";
					if (f < 10)
						fileNumber += "00" + std::to_string(f);
					else
						fileNumber += "0" + std::to_string(f);

					if (channelRadius == 0)
					{
						//1�`�����l�����̂܂܏���
						int* result;
						UtilityForCUDA::allocateDeviceMemory(result, sizeInfo);
						FGMF3::filter2DGPU(this->video->I_device[f], this->video->G_device[f], result, r, eps2, Imax, sizeInfo);
						//FGMF3::filter2DGPU(this->video->I_device[f], this->video->G_device[f], result, r, eps2 * channelRadius, Imax, sizeInfo);
						//PSNR�v�Z
						cv::Mat resultForPSNR = UtilityForCUDA::downloadLinearArrayAsMat(result, sizeInfo);
						double psnr = this->calculatePSNR(resultForPSNR, gt.I[f], Imax - 1);
						//std::cout << psnr << std::endl;
						psnrAve += psnr;

						//�摜�ۑ�(�t�H���_�Œ�@�ꎞ�I)
						//std::string saveImageName = saveImageDir + "2\\r" + std::to_string(channelRadius) + "\\img" + fileNumber + ".png";
						//cv::imwrite(saveImageName, resultForPSNR);

						cudaFree(result);
					}
					else
					{
						int startFrame = std::max(0, f - channelRadius);
						int endFrame = std::min(frameNum - 1, f + channelRadius);
						//video���珈���Ώۂ�I��1�t���[���AG���`�����l�����a�����ADeviceArray<int>�Ƃ��Ď擾
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

						//PSNR�v�Z
						cv::Mat resultForPSNR = UtilityForCUDA::downloadLinearArrayAsMat(result, sizeInfo);
						double psnr = this->calculatePSNR(resultForPSNR, gt.I[f], Imax - 1);
						//std::cout << psnr << std::endl;
						psnrAve += psnr;

						//�摜�ۑ�(�t�H���_�Œ�@�ꎞ�I)
						//std::string saveImageName = saveImageDir + "2\\r" + std::to_string(channelRadius) + "\\img" + fileNumber + ".png";
						//cv::imwrite(saveImageName, resultForPSNR);



						delete targetGs;
						cudaFree(result);
					}

				}

				//PSNR����
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

//Multispectral image denoising ��json�g�p��
void Experimenter::noiseRemovalForMultispectralImageForPaperNew()
{
	//�ݒ�
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

	//�ݒ�json�t�@�C��
	//std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\New\list\dev_experiments.json)";
	//�C���p���X�m�C�Y�p
	//std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\New_Pepper\list\dev_experiments.json)";
	//int numOfDigit = 2;
	//PaviaU
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\New_remote\PaviaU\list\dev_experiments_6.json)";
	int frameNum = 103;
	//WDCM
	//std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\New_remote\WDCM\list\dev_experiments.json)";
	//int frameNum = 191;

	//�`�����l�����ɔ�Ⴕ��eps��傫�����邩
	bool epsIncrementalMode = true;
	//�ǂݍ��܂��eps��int�^��
	bool epsIsInt = true;


	int numOfDigit = 3;

	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);




	//�t�@�C����
	std::string inputFileName = "img";
	std::string resultFileName = "result";



	std::string methodName;

	std::cout << "Data Num: " << em.data.dataNames.size() << std::endl;

	//��Ė@�@�`�����l�����g�܂�
	for (int i = 0; i < doProp.size(); i++)
	{
		if (doProp[i])
		{
			//�`�����l�����a
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

				//�摜�ǂݍ��� �`�����l���𓮉�t���[���Ƃ��ēǂݍ���
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

					//1channel���Ƃɏ���
					for (int f = 0; f < frameNum; f++)
					{
						//�`�����l�����a���g�p�t���[���J�n�I���ԍ�(vector�z����̔ԍ��Ȃ̂ŏ��0�n�܂�)
						int startFrame = std::max(0, f - channelRadius);
						int endFrame = std::min(frameNum - 1, f + channelRadius);
						//video���珈���Ώۂ�I��1�t���[���AG���`�����l�����a�����ADeviceArray<int>�Ƃ��Ď擾
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


	//��Ė@�@�`�����l�����g����
	for (int i = 0; i < doProp_ex.size(); i++)
	{
		if (doProp_ex[i])
		{
			//�`�����l�����a
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

				//�摜�ǂݍ��� �`�����l���𓮉�t���[���Ƃ��ēǂݍ���
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



					//1channel���Ƃɏ���
					for (int f = 0; f < frameNum; f++)
					{
						//�`�����l�����a���g�p�t���[���J�n�I���ԍ�(vector�z����̔ԍ��Ȃ̂ŏ��0�n�܂�)
						int startFrame = std::max(0, f - channelRadius);
						int endFrame = std::min(frameNum - 1, f + channelRadius);
						//video���珈���Ώۂ�I��1�t���[���AG���`�����l�����a�����ADeviceArray<int>�Ƃ��Ď擾
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

			//�������I�Ƀ`�����l����S���ǂݍ��ނƖ������ۂ��B
			//�摜�Ƃ��ĘA���ŏ������邵���Ȃ��B



			for (auto& param : em.methods.at(methodName).parameters)
			{
				float eps = param.get<float>("eps");
				float eps2 = eps * eps;
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");

				//1channel���Ƃɏ���
				for (int f = 0; f < frameNum; f++)
				{
					//�摜�ǂݍ���
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
		//�`�����l�����a0
		methodName = "100times_cr0";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, true, false, false, true, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;

			//�摜�ǂݍ��� �`�����l���𓮉�t���[���Ƃ��ēǂݍ���
			this->video = new Container_Video(inputPath, convertFlag);
			this->video->load(1, frameNum, numOfDigit);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");



				//1channel���Ƃɏ���
				for (int f = 0; f < frameNum; f++)
				{
					//video���珈���Ώۂ�I��1�t���[���A��������G�Ƃ��Ďg�p
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
		//�`�����l�����a1(=3�`�����l���g�p)
		methodName = "100times_cr1";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, true, false, false, true, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;

			//�摜�ǂݍ��� �`�����l���𓮉�t���[���Ƃ��ēǂݍ���
			this->video = new Container_Video(inputPath, convertFlag);
			this->video->load(1, frameNum, numOfDigit);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				float sigma = param.get<float>("sigma");
				int r = param.get<int>("radius");
				std::string id = param.get<std::string>("id");



				//1channel���Ƃɏ���
				for (int f = 0; f < frameNum; f++)
				{
					//video���珈���Ώۂ�I��1�t���[��
					//G�͎��́}1�`�����l���ŃJ���[�Ɠ������̍�邪�A�`�����l�������܂�Ԃ����E����������
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
		//�`�����l�����a0
		methodName = "median";
		std::cout << methodName << std::endl;
		ConvertImageFlag convertFlag = { true, false, false, true, false, false, false, false, false, false, false };

		for (int i = 0; i < em.data.dataNames.size(); i++)
		{
			std::cout << " " << i + 1;
			std::string inputName = em.data.dataNames[i];
			std::string resultName = em.data.resultNames[i];

			std::string inputPath = inputName + "\\" + inputFileName;

			//�摜�ǂݍ��� �`�����l���𓮉�t���[���Ƃ��ēǂݍ���
			this->video = new Container_Video(inputPath, convertFlag);
			this->video->load(1, frameNum, numOfDigit);

			SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, blockSize);

			for (auto& param : em.methods.at(methodName).parameters)
			{
				int r = param.get<int>("radius");
				int cr = param.get<int>("channelRadius");
				std::string id = param.get<std::string>("id");


				//1channel���Ƃɏ���
				for (int f = 0; f < frameNum; f++)
				{
					cv::Mat resultForEvaluation;
					if (cr == 0)
					{
						cv::medianBlur(this->video->I[f], resultForEvaluation, r * 2 + 1);
					}
					else
					{
						//��Ė@�̃K�C�h���ψ�̉摜�ɂ��邱�ƂŁAmedian�̑��`�����l�����Ή��Ƃ���
						// �Ǝv�������A�����͂Ȃ�Ȃ��B���K�v�Ȃ��̂��B
						// 
						//������
						/*
						//1channel���Ƃɏ���
						for (int f = 0; f < frameNum; f++)
						{
							//video���珈���Ώۂ�I��1�t���[���A��������G�Ƃ��Ďg�p
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
//�قȂ�m�C�Y���x���ɑ΂���e�X�g
void Experimenter::noiseRemovalForMultispectralImageForPaper()
{
	//�ݒ�
	float eps2 = 75.5f * 75.5f;
	//float eps2 = 10.0f * 10.0f;
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;
	int r = 5;
	int channelRadius = 1;
	int frameNum = 31;

	//�f�[�^�t�H���_
	std::string dataDir = R"(E:\MATLAB\FastMedianFiltering\Experiments\Hyperspectral Image Denoise\createTestData\data\toy\)";

	//PSNR�̕ۑ�
	std::string saveTimeName = dataDir + "result.txt";
	//���ʉ摜�̕ۑ�
	std::string saveImageDir = dataDir + R"(result\)";

	//�t�@�C���̍쐬
	std::ofstream writing_file;
	writing_file.open(saveTimeName, std::ios::app);
	//�o��
	writing_file << "Radius = " << r << ", ChannelRadius = " << channelRadius << std::endl;


	//2�߂�true�ɂ��CV_8UC1���m�ۂ��A������K�C�h�摜�̐����ɗp����
	ConvertImageFlag convertFlag = { false, true, false, false, false, false, false, true, true, false, false };

	//�m�C�Y�摜�t�H���_
	std::string fileDirNoise = dataDir + R"(noise\)";
	//�^�l�摜�t�H���_
	std::string fileDirGT = dataDir + R"(img\)";

	//�m�C�Y���x���i���̃v���O������ł̓m�C�Y�͉����Ȃ��B���łɉ�����Ă���摜��ǂݍ��ށj
	std::vector<int> nSigrange = { 10, 30, 50, 100 };

	//�^�l
	//mat_8��mat_8_color���m��
	ConvertImageFlag convertFlagGT = { true, false, false, true, false, false, false, false, false, false, false };
	std::string fileDirGT2 = fileDirGT + "img";
	Container_Video gt = Container_Video(fileDirGT2, convertFlagGT);
	gt.load(0, frameNum);

	for (int i = 0; i < nSigrange.size(); i++)
	{
		int sig = nSigrange[i];

		std::string fileDirNoise2 = fileDirNoise + std::to_string(i + 1) + "\\img";

		//�摜�ǂݍ��� �`�����l���𓮉�t���[���Ƃ��ēǂݍ���
		this->video = new Container_Video(fileDirNoise2, convertFlag);
		this->video->load(0, frameNum);
		SizeInfo sizeInfo = SizeInfo(this->video->imageSize.width, this->video->imageSize.height, Imax, dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1));

		double psnrAve = 0.0;
		std::cout << i << ": ";
		//1channel���Ƃɏ���
		for (int f = 0; f < frameNum; f++)
		{
			//std::cout << f << ": ";
			int startFrame = std::max(0, f - channelRadius);
			int endFrame = std::min(frameNum - 1, f + channelRadius);
			//video���珈���Ώۂ�I��1�t���[���AG���`�����l�����a�����ADeviceArray<int>�Ƃ��Ď擾
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

			//PSNR�v�Z
			cv::Mat resultForPSNR = Utility::downloadLinearArrayAsMat(result, sizeInfo);
			double psnr = this->calculatePSNR(resultForPSNR, gt.I[f], Imax - 1);
			//std::cout << psnr << std::endl;
			psnrAve += psnr;

			delete targetGs;
			cudaFree(result);
		}

		//PSNR����
		psnrAve /= (double)frameNum;
		std::cout << psnrAve << std::endl;

		writing_file << psnrAve << "\t";

	}
	writing_file << std::endl;
}
*/


void Experimenter::noiseRemovalForLightFieldForPaper()
{
	//�ݒ�
	float eps2 = 25.5f * 25.5f;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;
	int r = 5;
	int viewRadius = 1;
	int viewCols = 9;
	int viewRows = 9;


}

//4D ���C�g�t�B�[���h�f�B�X�p���e�B���t�@�C�������g
void Experimenter::disparityRefinementForLightFieldForPaper()
{
	//�ݒ�
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

				//�p�����[�^��
				std::string paramName = std::to_string(eps2) + "_" + std::to_string(r) + "_" + std::to_string(viewRadius);
				std::string saveDir2 = saveDir + paramName + "\\";


				//�ۑ��t�H���_����
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


//4D ���C�g�t�B�[���h�f�B�X�p���e�B���t�@�C�������g json��
void Experimenter::disparityRefinementForLightFieldForPaperNew()
{
	//�ݒ�
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

	//�ݒ�json�t�@�C��
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\LightFieldDisparityRefinement\New\list\dev_experiments_2.json)";
	int viewCols = 9;
	int viewRows = 9;
	int viewNum = viewCols * viewRows;


	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);




	//�t�@�C����
	std::string inputFileName = "img";
	std::string guideFileName = "input_Cam";
	std::string resultFileName = "result";



	std::string methodName;

	std::cout << "Data Num: " << em.data.dataNames.size() << std::endl;

	/*
	//��Ė@ (r�Œ�)
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
				int r = 18;//�Œ�
				std::string id = param.get<std::string>("id");

				if (viewRadius == doViewRadius)
				{

					FGMF3::filter4DGPU(video->I_device, video->G_device, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);


					//�S���_�ۑ�
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

	//��Ė@ (angularRadius�Œ�)
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
				int viewRadius = 2;//�Œ�
				std::string id = param.get<std::string>("id");

				FGMF3::filter4DGPU(video->I_device, video->G_device, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);


				//�S���_�ۑ�
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

				//1���_���Ƃɏ���
				for (int f = 0; f < viewNum; f++)
				{
					//�摜�ǂݍ���
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

				//1���_���Ƃɏ���
				for (int f = 0; f < viewNum; f++)
				{
					//�摜�ǂݍ���
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

				//1���_���Ƃɏ���
				for (int f = 0; f < viewNum; f++)
				{
					//�摜�ǂݍ���
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


//4D ���C�g�t�B�[���h�f�B�X�p���e�B���t�@�C�������g json�� �J���[�K�C�h
void Experimenter::disparityRefinementForLightFieldForPaperNewColor()
{
	//�ݒ�
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

	//�ݒ�json�t�@�C��
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\LightFieldDisparityRefinement\New\list\dev_experiments_6.json)";
	int viewCols = 9;
	int viewRows = 9;
	int viewNum = viewCols * viewRows;


	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);




	//�t�@�C����
	std::string inputFileName = "img";
	std::string guideFileName = "input_Cam";
	std::string resultFileName = "result";



	std::string methodName;

	std::cout << "Data Num: " << em.data.dataNames.size() << std::endl;

	
	//��Ė@ (r�Œ�)
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
				int r = 20;//�Œ�
				std::string id = param.get<std::string>("id");

				if (viewRadius == doViewRadius)
				{

					FGMF3::filter4DGPU(video->I_device, video->G_device_color, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);


					//�S���_�ۑ�
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
	//��Ė@ (angularRadius�Œ�)
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
				int viewRadius = 2;//�Œ�
				std::string id = param.get<std::string>("id");

				FGMF3::filter4DGPU(video->I_device, video->G_device_color, video->result_device, r, viewRadius, eps2, Imax, sizeInfo, viewCols);


				//�S���_�ۑ�
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

				//1���_���Ƃɏ���
				for (int f = 0; f < viewNum; f++)
				{
					//�摜�ǂݍ���
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

				//1���_���Ƃɏ���
				for (int f = 0; f < viewNum; f++)
				{
					//�摜�ǂݍ���
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

				//1���_���Ƃɏ���
				for (int f = 0; f < viewNum; f++)
				{
					//�摜�ǂݍ���
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


//�J���[�摜�̃m�C�Y�����@PSNR�ASSIM�AEKI���� (json��)
void Experimenter::noiseRemovalEvaluationForColorImage(std::string settingFileName)
{
	//�ݒ�
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


	//�t�@�C�����[�h
	//auto fileMode = std::ios::trunc;//�㏑���ۑ�

	std::string dataFileName = "input.png";
	std::string gtFileName = "gt.png";
	std::string resultFileName = "result.png";

	//���[�v�̏���
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
	 "median"(�P�Ȃ郁�W�A��)
	  imgs
	   rs
		eps2

	   �ۑ��t�@�C������
	   ~\[nSig]\[��@��]_result.txt
	   �ŁA�����̍\����
	   1�s��rs
	   2�s��eps or sigmas or 0
	   �ȍ~�̍s�@rs*2�s�ڂ̗v�f�i�Ȃ��̏ꍇ��1�j���A�摜���Ƃɕ]���l��1�s�ɕ��ׂ�

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

			//�m�C�Y�摜�ǂݍ���
			this->image = new Container_Image(dataPath, convertFlag);	this->image->load(useScaling, scalingSize);
			//�^�l�摜�ǂݍ���
			Container_Image* gtImage = new Container_Image(gtPath, convertFlagForGT); gtImage->load(useScaling, scalingSize);

			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			//�J���[�i���́E�K�C�h���Ɂj
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


			//�m�C�Y�摜�ǂݍ���
			this->image = new Container_Image(dataPath, convertFlag);	this->image->load(useScaling, scalingSize);
			//�^�l�摜�ǂݍ���
			Container_Image* gtImage = new Container_Image(gtPath, convertFlagForGT); gtImage->load(useScaling, scalingSize);

			SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

			//�J���[�i���́E�K�C�h���Ɂj

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

			//�m�C�Y�摜�ǂݍ���
			this->image = new Container_Image(dataPath, convertFlag);	this->image->load(useScaling, scalingSize);
			//�^�l�摜�ǂݍ���
			Container_Image* gtImage = new Container_Image(gtPath, convertFlagForGT); gtImage->load(useScaling, scalingSize);


			//�J���[�i���́E�K�C�h���Ɂj
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

			//�m�C�Y�摜�ǂݍ���
			this->image = new Container_Image(dataPath, convertFlag);	this->image->load(useScaling, scalingSize);
			//�^�l�摜�ǂݍ���
			Container_Image* gtImage = new Container_Image(gtPath, convertFlagForGT); gtImage->load(useScaling, scalingSize);


			//�J���[�i���́E�K�C�h���Ɂj
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


//�����摜�������_�̃��t�@�C�������g
void Experimenter::noiseRemovalEvaluationForDepthImageGuideColor()
{
	//�ݒ�
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	bool doProp = true;
	bool doCT = false;
	bool do100times = false;
	bool doMedian = false;

	//�ݒ�json�t�@�C��
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\list\dev_experiments.json)";
	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);


	//�t�@�C����
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

			//���莋���摜����͗p�A�J���[�摜���K�C�h�p�Ƃ��ēǂݍ���
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

			//���莋���摜����͗p�A�J���[�摜���K�C�h�p�Ƃ��ēǂݍ���
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

			//���莋���摜����͗p�A�J���[�摜���K�C�h�p�Ƃ��ēǂݍ���
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

			//���莋���摜����͗p�A�J���[�摜���K�C�h�p�Ƃ��ēǂݍ���
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

//�����摜�������_�̃��t�@�C�������g(�K�C�h�摜�O���[�X�P�[�X)
void Experimenter::noiseRemovalEvaluationForDepthImageGuideGray()
{
	//�ݒ�
	int threadNum = 12;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int Imax = 256;

	bool doProp = true;
	bool doCT = true;
	bool do100times = true;
	bool doMedian = false;

	//�ݒ�json�t�@�C��
	std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\LightField\New\list\dev_experiments_3.json)";
	//ExperimentManager
	ExperimentManager em = ExperimentManager(settingFileName);


	//�t�@�C����
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

			//���莋���摜����͗p�A�J���[�摜���K�C�h�p�Ƃ��ēǂݍ���
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

			//���莋���摜����͗p�A�J���[�摜���K�C�h�p�Ƃ��ēǂݍ���
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

			//���莋���摜����͗p�A�J���[�摜���K�C�h�p�Ƃ��ēǂݍ���
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

			//���莋���摜����͗p�A�J���[�摜���K�C�h�p�Ƃ��ēǂݍ���
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
	//�ݒ�
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

	//���莋���摜�ǂݍ���
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
	//�ݒ�
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
	//���莋���摜�ǂݍ���
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
//�_���p


//100+�Ƃ��ŃK�C�h�F�����k�����Ƃ��̉e�����o�邩���� �i�˂łȂ������j
void Experimenter::colorQuantizationDifference()
{
	float eps2 = 25.5f * 25.5f;
	int Imax = 256;
	int r = 15;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);


	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, true, true, false, false };



	cv::Size sz = cv::Size(900, 600);

	//�����o��摜��T���p

	//�摜�f�[�^�ǂݍ���(����)
	//�܂����X�g�̓ǂݍ���
	std::string imageListFile = R"(E:\MATLAB\FastMedianFiltering\Experiments\Speed Test\HD_List.txt)";
	std::string fileName;
	std::ifstream fs(imageListFile);


	int n = 0;

	while (!fs.eof())
	{
		n++;
		//�t�@�C�����̓ǂݍ���
		fs >> fileName;
		std::cout << fileName << std::endl;


		this->image = new Container_Image(fileName, convertFlag);
		this->image->load(true, sz);
		//this->image->load();
		SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);


		//��Ė@GPU
		FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);


		//100+
		cv::Mat result = l1Solver::filter(this->image->I32_color, this->image->G_color, r);

		cv::Mat resultProp = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);

		cv::imshow("100+", result * 255.0f);
		cv::imshow("Prop", resultProp * 255.0f);
		cv::waitKey(0);
	}
}

//fmin��median tracking�ō����������f�����v�Z����
void Experimenter::trackingDifference1()
{
	float eps2 = 25.5f * 25.5f;
	int Imax = 256;
	int r = 7;
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	int threadNum = 12;


	ConvertImageFlag convertFlag = { false, true, false, false, true, false, false, false, false, false, false };


	//cv::Size sz = cv::Size(900, 600);

	//�摜�f�[�^�ǂݍ���(����)
	//�܂����X�g�̓ǂݍ���
	std::string imageListFile = R"(E:\MATLAB\FastMedianFiltering\Experiments\Speed Test\HD_List.txt)";
	std::string fileName;
	std::ifstream fs(imageListFile);


	int n = 0;

	int numOfDif = 0;
	int totalPix = 0;

	while (!fs.eof())
	{
		n++;
		//�t�@�C�����̓ǂݍ���
		fs >> fileName;
		//std::cout << fileName << std::endl;


		this->image = new Container_Image(fileName, convertFlag);
		//this->image->load(true, sz);
		this->image->load();
		SizeInfo sizeInfo = SizeInfo(this->image->imageSize.width, this->image->imageSize.height, Imax, blockSize);

		//��Ė@CPU�imedian tracking�j
		cv::Mat result_tracking = FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);

		//��Ė@List�ifmin����T���j
		cv::Mat result_fmin = FGMF1::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
		
		//����
		cv::Mat dif;
		cv::absdiff(result_tracking, result_fmin, dif);
		//���̂����f�����v�Z
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
	//�v���O�������s�f�B���N�g���擾
	// ���s�t�@�C���̃p�X
	std::string modulePath = "";
	/*
	// �h���C�u���A�f�B���N�g�����A�t�@�C�����A�g���q
	char path[MAX_PATH], drive[MAX_PATH], dir[MAX_PATH], fname[MAX_PATH], ext[MAX_PATH];

	// ���s�t�@�C���̃t�@�C���p�X���擾
	if (::GetModuleFileNameA(NULL, path, MAX_PATH) != 0)
	{
		// �t�@�C���p�X�𕪊�
		::_splitpath_s(path, drive, dir, fname, ext);
		// �h���C�u�ƃf�B���N�g�������������Ď��s�t�@�C���p�X�Ƃ���
		modulePath = std::string(drive) + std::string(dir);
	}

	//���̃f�B���N�g����config�t�H���_�����邩
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
			//�Ȃ���Ή��x���ȉ������s
			//���̃f�B���N�g����config�t�H���_���邩
			currentPath.erase(--currentPath.end());
			int last_pos = currentPath.find_last_of('\\');
			currentPath = currentPath.erase(last_pos + 1);
			if (currentPath == "")
			{
				//�����炸
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
	//json�t�@�C������ݒ�ǂݍ���
	boost::property_tree::ptree pt;
	boost::property_tree::read_json(settingFileName, pt);

	std::string baseStr;



}
