#include "Method.h"


Method::Method()
{	//画像読み込み
	std::string filePathSrc_2d;
	std::string filePathGuide_2d;
	std::string filePathSrc_video;
	std::string filePathGuide_video;
#ifdef HOME
	//2d
	filePathSrc_2d = R"(C:\Users\Kazu\source\repos\GuidedMedianFilter\GuidedMedianFilter\image\src.png)";
	filePathGuide_2d = R"(C:\Users\Kazu\source\repos\GuidedMedianFilter\GuidedMedianFilter\image\guide.png)";
	//video
	filePathSrc_video = R"(C:\Users\Kazu\source\repos\GuidedMedianFilter\GuidedMedianFilter\image\result of flownet2\floForWMF\flo1_)";
	filePathGuide_video = R"(C:\Users\Kazu\source\repos\GuidedMedianFilter\GuidedMedianFilter\image\result of flownet2\rgb\frame)";

	filePathSrc_video = R"(C:\Users\Kazu\source\repos\GuidedMedianFilter\GuidedMedianFilter\image\data\dishes\img)";
#else
	//2d
	//filePathSrc_2d = R"(src.png)";
	filePathSrc_2d = R"(E:\ProgramCode\\FastGuidedMedianFilter_save\image\src.png)";
	filePathGuide_2d = R"(E:\ProgramCode\\FastGuidedMedianFilter_save\image\guide.png)";
	//video
	filePathSrc_video = R"(E:\MATLAB\FastMedianFiltering\Experiments\optical flow\middlebury\result of flownet2\floForWMF\flo1_)";
	filePathGuide_video = R"(E:\MATLAB\FastMedianFiltering\Experiments\optical flow\middlebury\result of flownet2\rgb\frame)";
	filePathSrc_video = R"(E:\MATLAB\FastMedianFiltering\Experiments\LightField Denoise\createTestData\data\dishes\img)";
	#endif // HOME
#ifdef HPC
	filePathSrc_2d = "src.png";
	//filePathSrc = "E:\ProgramCode\FastGuidedMedianFilter\image\input_init.png";
	filePathGuide_2d = "guide.png";
#endif
	
	//cv::Mat::setDefaultAllocator(cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

/*
	ConvertImageFlag convertFlag = {false, true, false, false, true, false, true, true};
	filePathSrc = R"(E:\MATLAB\FastMedianFiltering\Experiments\LightField Denoise\createTestData\data\dishes\img000.png)";
	//this->image = new Container_Image(filePathSrc, filePathGuide, convertFlag);
	//filePathSrc = R"(E:\MATLAB\FastMedianFiltering\Experiments\LightField Denoise\createTestData\data\dishes\img000.png)";
	this->image = new Container_Image(filePathSrc, convertFlag);
	//this->image->load();
	*/

	Experimenter ex;
	//ex.performFilteringForVideo(filePathSrc_video);
	ex.performSpeedTest(filePathSrc_2d);
	//ex.test(filePathSrc_2d);
	
	//論文用
	//ex.colorQuantizationDifference();
	//ex.trackingDifference1();
	

	//std::string filePathSrc_higher = R"(E:\LargeData\HighPrecision\iphd_test_depth\images\vid00227_lJx7hpNV.png)";
	//ex.test(filePathSrc_higher);

	//論文用実験
	//速度テスト用
	//ex.speedTest2D8bitForPaper();
	//高ビットテスト用
	//ex.speedTest2DHigherBitForPaper();
	//マルチスペクトラルテスト用
	//ex.noiseRemovalForMultispectralImageForPaper();
	//4D ライトフィールドディスパリティリファインメント
	//ex.disparityRefinementForLightFieldForPaper();
	//フラッシュ/ノンフラッシュによるreversal artifact確認用
	//ex.flashNoFlashForPaper();
	// カラーノイズ除去
	//std::string settingFileName = R"(E:\MATLAB\FastMedianFiltering\Experiments\Denoise\ColorImage\New\list\dev_experiments_5.json)";
	//ex.noiseRemovalEvaluationForColorImage(settingFileName);


	//ex.jointUpsamplingTest();
	// 
	//ex.noiseRemovalEvaluationForDepthImage();
	//ex.testForConstantTimeWMF();
	//ex.noiseRemovalForMultispectralImageForPaperNew();
	// 
	//ex.disparityRefinementForLightFieldForPaperNew();
	//ex.noiseRemovalEvaluationForDepthImageGuideGray();
	//ex.testForProp4DColor();
	//ex.disparityRefinementForLightFieldForPaperNewColor();
	//ex.testForAblationStudy();
}


void Method::test()
{
	this->image->load();
	int r = 3;


	int Imax = 256;
	float eps2 = 25.5f * 25.5f;
	int threadNum = 12;
	std::cout << "(h,w)= " << this->image->I.rows << "," << this->image->I.cols << std::endl;


	//debugモード時はレジスタ容量変わるのでサイズを小さくとらないとoutofresourceになる
	dim3 blockSize = dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
	//dim3 blockSize = dim3(32, 32, 1);
	SizeInfo sizeInfo = SizeInfo(this->image->I.cols, this->image->I.rows, Imax, blockSize);
	
	//I1G3
#if 1
	FGMF3::filter2DGPU(this->image->I_device, this->image->G_device_color, this->image->result_device, r, eps2, Imax, sizeInfo);
	cv::Mat result1 = UtilityForCUDA::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);
	cv::Mat result2 = FGMF::filter2DInterface(this->image->I32, this->image->G32_color, threadNum, r, eps2, Imax);
#endif
	//I3G1
#if 0
	FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device, this->image->result_device_color, r, eps2, Imax, sizeInfo);
	cv::Mat result1 = Utility::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);
	cv::Mat result2 = FGMF::filter2DInterface(this->image->I32_color, this->image->G32, threadNum, r, eps2, Imax);
#endif
	//I3G3
#if 0
	FGMF3::filter2DGPU(this->image->I_device_color, this->image->G_device_color, this->image->result_device_color, r, eps2, Imax, sizeInfo);
	cv::Mat result1 = Utility::downloadLinearArrayAsMat(this->image->result_device_color, sizeInfo);
	cv::Mat result2 = FGMF::filter2DInterface(this->image->I32_color, this->image->G32_color, threadNum, r, eps2, Imax);
#endif

	//I1G1
#if 0
	//FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);
	//GuidedFilter::filterSimplified(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, sizeInfo);
	//cv::Mat result1 = Utility::downloadLinearArrayAsMat(this->image->result_device, sizeInfo);

	cv::Mat result1 = l1Solver::filter(this->image->I32, this->image->G, r);
	cv::Mat result2 = FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
#endif



	/*
	dim3 blockSize = dim3(32, 32, 1);
	SizeInfo sizeInfo = SizeInfo(this->I.cols, this->I.rows, Imax, blockSize);
	//Mat result2 = FGMF::gpuTestCxDx(this->I32_color, this->G32, r, eps2, Imax, sizeInfo);
	Mat result1 = FGMF3::filter2DGPU(this->I32, this->G32, r, eps2, Imax, sizeInfo);

	Mat result2 = FGMF3::filter2DGPUFast(this->I32, this->G32, r, eps2, Imax, sizeInfo);
	*/

	//Mat result2 = FGMF3::filter2DGPUFast(this->I32, this->G32, r, eps2, Imax, sizeInfo);

	cv::Mat mat;
	result1.convertTo(mat, CV_8U);
	cv::imshow("Result1", mat);
	//PixelInfo::showPixelInfo("Result1", result1);
	result2.convertTo(mat, CV_8U);
	cv::imshow("Result2", mat);
	//PixelInfo::showPixelInfo("Result2", result2);

#if 1

		//差分
		cv::Mat diff = cv::abs(result1 - result2) * 1000;
		//diff.convertTo(diff, CV_8U);
		cv::imshow("Diff", diff);
		//PixelInfo::showPixelInfo("Diff", diff);


#endif

		//cv::imshow("raw", this->I);
		cv::waitKey(0);
	


}


void Method::multiTest() {
	//カラー


}

void Method::speedTest()
{
	float eps2 = 25.5f * 25.5f;
	int Imax = 256;
	int threadNum = 12;

	dim3 blockSize = dim3(4, 4, 1);

	std::vector<int> rs = { 5,15,35,55 };
	std::vector<cv::Size> sizes = { cv::Size(128 * 2, 128 * 2), cv::Size(128 * 4, 128 * 4), cv::Size(128 * 8, 128 * 4),cv::Size(128 * 4, 128 * 8),cv::Size(128 * 8, 128 * 8),cv::Size(128 * 1, 128 * 8),cv::Size(128 * 8, 128 * 1) };

	for (cv::Size sz : sizes)
	{
		SizeInfo sizeInfo = SizeInfo(sz.width, sz.height, Imax, blockSize);


		std::cout << "h*w= " << sz.height << "×" << sz.width << std::endl;
		std::vector<double> time0;
		std::vector<double> time1;
		std::vector<double> time2;
		std::vector<double> time3;
		std::vector<double> time4;
		std::cout << "Radius\t";

		for (int r : rs)
		{
			//this->raw2Image(true, sz);
			this->image->load(true, sz);

			std::cout << r << "\t";
			for (int i = 0; i < 5; i++)
			{
				if (i == 0)
				{
					std::chrono::system_clock::time_point  start, end;
					start = std::chrono::system_clock::now();

					l1Solver::filter(this->image->I32, this->image->G, r);

					end = std::chrono::system_clock::now();
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					time0.push_back(elapsed);
				}
				else if (i == 1)
				{
					std::chrono::system_clock::time_point  start, end;
					start = std::chrono::system_clock::now();

					FGMF::filter2DInterface(this->image->I32, this->image->G32, threadNum, r, eps2, Imax);
					//result.convertTo(result, CV_8U);
					//cv::imshow("Result (prop)", result);

					end = std::chrono::system_clock::now();
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					time1.push_back(elapsed);
				}
				else if (i == 2)
				{
					std::chrono::system_clock::time_point  start, end;
					start = std::chrono::system_clock::now();

					FGMF3::filter2DGPU(this->image->I_device, this->image->G_device, this->image->result_device, r, eps2, Imax, sizeInfo);

					end = std::chrono::system_clock::now();
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					time2.push_back(elapsed);
				}
				else if (i == 3)
				{
					std::chrono::system_clock::time_point  start, end;
					start = std::chrono::system_clock::now();

					//FGMF3::filter2DGPU_histogramSampling(this->image->I_device, this->image->G_device, this->image->result_device, 5, r, eps2, Imax, sizeInfo);

					end = std::chrono::system_clock::now();
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					time3.push_back(elapsed);

				}
				else if (i == 4)
				{
					std::chrono::system_clock::time_point  start, end;
					start = std::chrono::system_clock::now();

					//FGMF3::filter2DGPU_selfGuide(this->image->I_device, this->image->G_device, r, eps2, Imax, sizeInfo);

					end = std::chrono::system_clock::now();
					double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					time4.push_back(elapsed);
				}
			}
			//cv::waitKey(0);
		}
		std::cout << std::endl;

		std::cout << "conv: ";
		for (double tm : time0)
		{
			std::cout << tm << "\t";
		}
		std::cout << std::endl;

		std::cout << "cpu : ";
		for (double tm : time1)
		{
			std::cout << tm << "\t";
		}
		std::cout << std::endl;

		std::cout << "gpu: ";
		for (double tm : time2)
		{
			std::cout << tm << "\t";
		}
		std::cout << std::endl;

		std::cout << "sh1: ";
		for (double tm : time3)
		{
			std::cout << tm << "\t";
		}
		std::cout << std::endl;

		std::cout << "sh2: ";
		for (double tm : time4)
		{
			std::cout << tm << "\t";
		}
		std::cout << std::endl << std::endl;
	}


}
