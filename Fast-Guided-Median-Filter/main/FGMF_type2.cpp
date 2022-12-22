#include "FGMF_type2.h"

cv::Mat FGMF2::filter2DInterface(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax) {
	//入力画像チャンネル数
	const int IchannelNum = I.channels();
	//ガイド画像チャンネル数
	const int GchannelNum = G.channels();

	//return FGMF2::filter2DWindow<gSum, fgSumUpToIndex, fg, int, float>(I, G, threadNum, radius, eps2, Imax);
	
	
	if (IchannelNum == 1 && GchannelNum == 1)
		return FGMF2::filter2DWindow<gSum, fgSumUpToIndex, fg, int, float>(I, G, threadNum, radius, eps2, Imax);
	else if (IchannelNum == 1 && GchannelNum == 3)
		return FGMF2::filter2DWindow<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(I, G, threadNum, radius, eps2, Imax);
	else if (IchannelNum == 3 && GchannelNum == 1) {

		return FGMF2::filter2DWindowI3<gSum, fgSumUpToIndex, fg, int, float>(CV_32FC1, I, G, threadNum, radius, eps2, Imax);
	}
	else if (IchannelNum == 3 && GchannelNum == 3) {
		return FGMF2::filter2DWindowI3<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(CV_32FC3, I, G, threadNum, radius, eps2, Imax);
	}
	return cv::Mat();
}



#if 0
//GPU使用テスト
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
std::vector<cv::Mat> FGMF2::filter3DI1withGPU(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, SizeInfo sizeInfo, int radius_space, int radius_depth, float eps2, int Imax)
{
	//check validation
	assert(I[0].depth() == CV_32S && I[0].channels() == 1);
	assert(G[0].depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif



	//入力次元
	const int DIM = 3;
	//入力画像チャンネル数
	const int Ichannels = I[0].channels();
	//中央値
	const float half = 0.5f;
	//初期化
	//サイズ
	const std::vector<int> size_dim{ I[0].cols , I[0].rows, (int)I.size() };
	//半径
	const std::vector<int> r_dim{ radius_space, radius_space, radius_depth };


	//GPU計算用
	cv::Mat cx;// = cv::Mat(size_dim[1], size_dim[0], CV_32FC(G[0].channels));
	cv::Mat dx;// = cv::Mat(size_dim[1], size_dim[0], CV_32FC(1));
	//mat GMat を int* Gにコピー
	int* GDevice;
	float* cxDevice;//いずれCTYPE
	float* dxDevice;
	std::vector<int4*> sumG(size_dim[2]);//いずれr_dim[2]*2+1 or 2のサイズに縮小
	int4* sumGwindow;//window内合計格納用
	int4* temp;
	//別スレッドで実行
	Utility::allocateDeviceMemory(GDevice, G[0], sizeInfo);
	//Utility::showDevice(G, sizeInfo, "test", false, 255);
	//float* cx, dxを確保
	Utility::allocateDeviceMemory(cxDevice, sizeInfo);
	Utility::allocateDeviceMemory(dxDevice, sizeInfo);
	Utility::allocateDeviceMemory(sumG[0], sizeInfo);
	Utility::allocateDeviceMemory(sumGwindow, 0, sizeInfo);
	Utility::allocateDeviceMemory(temp, sizeInfo);
	cu_addSumG(sizeInfo, NULL, GDevice, radius_space, sumG[0], sumGwindow, temp);
	//cx,dxをcxMat, dxMatにコピー
	cx = Utility::downloadLinearArrayAsMat(cxDevice, sizeInfo);
	dx = Utility::downloadLinearArrayAsMat(dxDevice, sizeInfo);




	//マルチスレッド用
	std::vector<int> dim0Start_vec(threadNum);
	std::vector<int> dim0End_vec(threadNum);
	std::vector<int> memoryLength_vec(threadNum);
	std::vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//メモリ確保・初期化
	//結果保存
	std::vector<cv::Mat> result(I.size());
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = cv::Mat(size_dim[1], size_dim[0], CV_32SC(Ichannels));
	}

	// (gSum, sumUpToIndex, histo)
	//W0:画素なので記録の必要なし（I,G）
	//W1
	std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
	//= Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 1);
	//W2
	std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W2(threadNum);
	//Window_vector<GSum, FGSumUpToIndex, FG> W2 = Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 2);
	//W3(Wmain)
	std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		std::vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, false)));
		W2[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 2, false)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}






	//次の階層の処理位置セット
	Pos x2;
	DimStatus status2, status2GPU;
	int pixelSum2;
	setPos(x2, r_dim[2]);
	for (; x2.center < size_dim[2]; x2.center++, x2.add++, x2.rem++)
	{
		//ステータスセット
		setStatusAtOutermostLoop(x2, size_dim[2], status2);
		//1列当たりの画素数
		int pixel_sum2 = calculatePixelNumAtOutermostLoop(x2, size_dim[2], r_dim[2], status2);

		Pos vecPos;
		vecPos.center = (std::max)(0, x2.center);
		vecPos.add = (std::min)(size_dim[2] - 1, x2.add);
		vecPos.rem = (std::max)(0, x2.rem);

		//GPU用
		setNextStatus(x2, size_dim[2], 1, status2GPU);


#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			//画素数
			std::vector<int> pixel_sum(2);
			//位置
			std::vector<Pos> x(2);
			//ステータス
			std::vector<DimStatus> status(2);


			const int dim0Start = dim0Start_vec[k];
			const int dim0End = dim0End_vec[k];
			const int insideImageStart = insideImageStart_vec[k];


			//対応する画素へのポインタ（次元によって設定が異なることに注意）
			//3次元以降については、画素の座標は同じだが、vectorのどのcv::Matを取り出すのかが変わる
			GTYPE* G_center_rowStart = G[vecPos.center].ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* result_center_rowStart = result[vecPos.center].ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
			CTYPE* cx_center_rowStart = cx.ptr<CTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			float* dx_center_rowStart = dx.ptr<float>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* W0_rem_f_rowStart = I[vecPos.rem].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_rem_g_rowStart = G[vecPos.rem].ptr<GTYPE>(0) + dim0Start + r_dim[0];
			int* W0_add_f_rowStart = I[vecPos.add].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_add_g_rowStart = G[vecPos.add].ptr<GTYPE>(0) + dim0Start + r_dim[0];

			//W1メモリ位置リセット
			W1[k]->resetPos();
			//次の階層の処理位置セット
			setPos(x[1], r_dim[1]);
			for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
			{
				//ステータスセット
				setStatus(x[1], size_dim[1], status2.isInside_image, status[1]);
				//1列当たりの画素数
				pixel_sum[1] = calculatePixelNumAtDim(x[1], size_dim[1], r_dim[1], status[1], pixel_sum2);


				//初期化
				//W3
				Wmain[k]->initialize();
				//ウィンドウ内画素数
				int pixel_sum_window = 0;
				//ウィンドウ内画素数の逆数
				float pixel_sum_window_inv = 0.0f;

				//画素へのポインタ初期化
				GTYPE* G_center = G_center_rowStart;
				int* result_center = result_center_rowStart;
				CTYPE* cx_center = cx_center_rowStart;
				float* dx_center = dx_center_rowStart;
				int* W0_rem_f = W0_rem_f_rowStart;
				GTYPE* W0_rem_g = W0_rem_g_rowStart;
				int* W0_add_f = W0_add_f_rowStart;
				GTYPE* W0_add_g = W0_add_g_rowStart;


				//W2メモリ位置リセット
				W2[k]->resetPos();
				//次の階層の処理位置セット
				setPosAtDim0(x[0], r_dim[0], dim0Start);
				const int remStartPos = x[0].add;
				for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
				{
					//ステータスセット
					setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);

					//window内画素数の更新
					calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

					//W3の更新
					//W3 = W3 + W2[x[0].add] - W2[x[0].rem](W3はwindow)
					if (status[0].hasAdd)
					{
						//W2の更新
						//W2[x[0].add] = W2[x[0].add] + W1[x[1].add, x[0].add] - W1[x[1].rem, x[0].add]
						if (status[1].hasAdd)
						{
							//W1[x[0].add] の更新
							//W1[x[1].add, x[0].add] = W1[x[1].add, x[0].add] + W0[x[2].add, x[1].add, x[0].add] - W0[x[2].rem, x[1].add, x[0].add]
							if (status2.hasAdd) //画素追加	(W1[x[0].add]) + W0[x[1].add, x[0].add]
								addPixelToWindow(*W1[k], W0_add_f, W0_add_g);
							if (status2.hasRem)//画素削除	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
								removePixelFromWindow(*W1[k], W0_rem_f, W0_rem_g);
							//W1追加	(W2) + W1[x[0].add]
							updateSubWindowAndAddToWindow(Imax, *W2[k], *W1[k]);
						}
						if (status[1].hasRem)//W1削除	(W2) - W1[x[0].rem]
							removeSubWindowFromWindow(Imax, *W2[k], *W1[k]);
						//W2追加	(W3) + W1[x[0].add]
						updateSubWindowAndAddToWindow(Imax, *Wmain[k], *W2[k]);
					}
					if (status[0].hasRem)//W2削除	(W3) - W2[x[0].rem]
						removeSubWindowFromWindow(Imax, *Wmain[k], *W2[k]);

					//中央値の計算
					if (status[0].isInside_image)
					{
						/*
						CTYPE cx;
						float dx;
						calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
						*/
						findMedian(*cx_center, *dx_center, half, *Wmain[k], *result_center);
						G_center++;
						result_center++;
						cx_center++;
						dx_center++;
					}
				}
				G_center_rowStart += size_dim[0];
				result_center_rowStart += size_dim[0];
				cx_center_rowStart += size_dim[0];
				dx_center_rowStart += size_dim[0];
				W0_add_f_rowStart += size_dim[0];
				W0_add_g_rowStart += size_dim[0];
				W0_rem_f_rowStart += size_dim[0];
				W0_rem_g_rowStart += size_dim[0];
			}
			//自身の次元+1のループが終わったときにウィンドウ内容リセット
			W2[k]->setZero();

		}

		//gsum更新
		if (status2GPU.hasAddOnly)
		{
			Utility::uploadMatToDevice(G[x2.add + 1], GDevice, sizeInfo, NULL);
			Utility::allocateDeviceMemory(sumG[x2.add + 1], sizeInfo);
			cu_addSumG(sizeInfo, NULL, GDevice, radius_space, sumG[x2.add + 1], sumGwindow, temp);
		}
		else if (status2GPU.hasRemOnly)
		{
			cu_remSumG(sizeInfo, NULL, GDevice, sumG[x2.rem + 1], sumGwindow);
		}
		else
		{
			Utility::uploadMatToDevice(G[x2.add + 1], GDevice, sizeInfo, NULL);
			Utility::allocateDeviceMemory(sumG[x2.add + 1], sizeInfo);
			cu_updateSumG(sizeInfo, NULL, GDevice, radius_space, sumG[x2.add + 1], sumG[x2.rem + 1], sumGwindow, temp);
		}
		//cxdx更新
		if (status2GPU.isInside_image)
		{
			Utility::uploadMatToDevice(G[x2.center + 1], GDevice, sizeInfo, NULL);
			cu_calculateCxDx(sizeInfo, NULL, GDevice, radius_space, eps2, cxDevice, dxDevice, sumGwindow, temp);
			Utility::downloadLinearArrayAsMat(cxDevice, sizeInfo, cx);
			Utility::downloadLinearArrayAsMat(dxDevice, sizeInfo, dx);
			/*
			cv::imshow("cx", cv::abs(cx*100000));
			cv::imshow("dx", cv::abs(dx * 100));
			cv::waitKey(0);
			*/
		}

	}

	return result;
}

//GPU使用テスト
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
std::vector<cv::Mat> FGMF2::filter3DI1withGPUThread(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, SizeInfo sizeInfo, int radius_space, int radius_depth, float eps2, int Imax)
{
	//check validation
	assert(I[0].depth() == CV_32S && I[0].channels() == 1);
	assert(G[0].depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif



	//入力次元
	const int DIM = 3;
	//入力画像チャンネル数
	const int Ichannels = I[0].channels();
	//中央値
	const float half = 0.5f;
	//初期化
	//サイズ
	const std::vector<int> size_dim{ I[0].cols , I[0].rows, (int)I.size() };
	//半径
	const std::vector<int> r_dim{ radius_space, radius_space, radius_depth };


	//GPU計算用
	cv::Mat cx;// = cv::Mat(size_dim[1], size_dim[0], CV_32FC(G[0].channels));
	cv::Mat dx;// = cv::Mat(size_dim[1], size_dim[0], CV_32FC(1));
	//mat GMat を int* Gにコピー
	int* GDevice;
	float* cxDevice;//いずれCTYPE
	float* dxDevice;
	std::vector<int4*> sumG(size_dim[2]);//いずれr_dim[2]*2+1 or 2のサイズに縮小
	int4* sumGwindow;//window内合計格納用
	int4* temp;
	//別スレッドで実行
	std::thread th0([&]() {
		Utility::allocateDeviceMemory(GDevice, G[0], sizeInfo);
		//Utility::showDevice(G, sizeInfo, "test", false, 255);
		//float* cx, dxを確保
		Utility::allocateDeviceMemory(cxDevice, sizeInfo);
		Utility::allocateDeviceMemory(dxDevice, sizeInfo);
		Utility::allocateDeviceMemory(sumG[0], sizeInfo);
		Utility::allocateDeviceMemory(sumGwindow, 0, sizeInfo);
		Utility::allocateDeviceMemory(temp, sizeInfo);
		cu_addSumG(sizeInfo, NULL, GDevice, radius_space, sumG[0], sumGwindow, temp);
		//cx,dxをcxMat, dxMatにコピー
		cx = Utility::downloadLinearArrayAsMat(cxDevice, sizeInfo);
		dx = Utility::downloadLinearArrayAsMat(dxDevice, sizeInfo);
	});



	//マルチスレッド用
	std::vector<int> dim0Start_vec(threadNum);
	std::vector<int> dim0End_vec(threadNum);
	std::vector<int> memoryLength_vec(threadNum);
	std::vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//メモリ確保・初期化
	//結果保存
	std::vector<cv::Mat> result(I.size());
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = cv::Mat(size_dim[1], size_dim[0], CV_32SC(Ichannels));
	}

	// (gSum, sumUpToIndex, histo)
	//W0:画素なので記録の必要なし（I,G）
	//W1
	std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
	//= Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 1);
	//W2
	std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W2(threadNum);
	//Window_vector<GSum, FGSumUpToIndex, FG> W2 = Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 2);
	//W3(Wmain)
	std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		std::vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, false)));
		W2[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 2, false)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}



	th0.join();


	//次の階層の処理位置セット
	Pos x2;
	DimStatus status2, status2GPU;
	int pixelSum2;
	setPos(x2, r_dim[2]);
	for (; x2.center < size_dim[2]; x2.center++, x2.add++, x2.rem++)
	{
		//ステータスセット
		setStatusAtOutermostLoop(x2, size_dim[2], status2);
		//1列当たりの画素数
		int pixel_sum2 = calculatePixelNumAtOutermostLoop(x2, size_dim[2], r_dim[2], status2);

		Pos vecPos;
		vecPos.center = (std::max)(0, x2.center);
		vecPos.add = (std::min)(size_dim[2] - 1, x2.add);
		vecPos.rem = (std::max)(0, x2.rem);

		//GPU用
		setNextStatus(x2, size_dim[2], 1, status2GPU);


		std::thread th1([&]() {
			//gsum更新
			if (status2GPU.hasAddOnly)
			{
				Utility::uploadMatToDevice(G[x2.add + 1], GDevice, sizeInfo, NULL);
				Utility::allocateDeviceMemory(sumG[x2.add + 1], sizeInfo);
				cu_addSumG(sizeInfo, NULL, GDevice, radius_space, sumG[x2.add + 1], sumGwindow, temp);
			}
			else if (status2GPU.hasRemOnly)
			{
				cu_remSumG(sizeInfo, NULL, GDevice, sumG[x2.rem + 1], sumGwindow);
			}
			else
			{
				Utility::uploadMatToDevice(G[x2.add + 1], GDevice, sizeInfo, NULL);
				Utility::allocateDeviceMemory(sumG[x2.add + 1], sizeInfo);
				cu_updateSumG(sizeInfo, NULL, GDevice, radius_space, sumG[x2.add + 1], sumG[x2.rem + 1], sumGwindow, temp);
			}
			//cxdx更新
			if (status2GPU.isInside_image)
			{
				Utility::uploadMatToDevice(G[x2.center + 1], GDevice, sizeInfo, NULL);
				cu_calculateCxDx(sizeInfo, NULL, GDevice, radius_space, eps2, cxDevice, dxDevice, sumGwindow, temp);
			}
		});

		//th1.join();

#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			//画素数
			std::vector<int> pixel_sum(2);
			//位置
			std::vector<Pos> x(2);
			//ステータス
			std::vector<DimStatus> status(2);


			const int dim0Start = dim0Start_vec[k];
			const int dim0End = dim0End_vec[k];
			const int insideImageStart = insideImageStart_vec[k];


			//対応する画素へのポインタ（次元によって設定が異なることに注意）
			//3次元以降については、画素の座標は同じだが、vectorのどのcv::Matを取り出すのかが変わる
			GTYPE* G_center_rowStart = G[vecPos.center].ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* result_center_rowStart = result[vecPos.center].ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
			CTYPE* cx_center_rowStart = cx.ptr<CTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			float* dx_center_rowStart = dx.ptr<float>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* W0_rem_f_rowStart = I[vecPos.rem].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_rem_g_rowStart = G[vecPos.rem].ptr<GTYPE>(0) + dim0Start + r_dim[0];
			int* W0_add_f_rowStart = I[vecPos.add].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_add_g_rowStart = G[vecPos.add].ptr<GTYPE>(0) + dim0Start + r_dim[0];

			//W1メモリ位置リセット
			W1[k]->resetPos();
			//次の階層の処理位置セット
			setPos(x[1], r_dim[1]);
			for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
			{
				//ステータスセット
				setStatus(x[1], size_dim[1], status2.isInside_image, status[1]);
				//1列当たりの画素数
				pixel_sum[1] = calculatePixelNumAtDim(x[1], size_dim[1], r_dim[1], status[1], pixel_sum2);


				//初期化
				//W3
				Wmain[k]->initialize();
				//ウィンドウ内画素数
				int pixel_sum_window = 0;
				//ウィンドウ内画素数の逆数
				float pixel_sum_window_inv = 0.0f;

				//画素へのポインタ初期化
				GTYPE* G_center = G_center_rowStart;
				int* result_center = result_center_rowStart;
				CTYPE* cx_center = cx_center_rowStart;
				float* dx_center = dx_center_rowStart;
				int* W0_rem_f = W0_rem_f_rowStart;
				GTYPE* W0_rem_g = W0_rem_g_rowStart;
				int* W0_add_f = W0_add_f_rowStart;
				GTYPE* W0_add_g = W0_add_g_rowStart;


				//W2メモリ位置リセット
				W2[k]->resetPos();
				//次の階層の処理位置セット
				setPosAtDim0(x[0], r_dim[0], dim0Start);
				const int remStartPos = x[0].add;
				for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
				{
					//ステータスセット
					setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);

					//window内画素数の更新
					calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

					//W3の更新
					//W3 = W3 + W2[x[0].add] - W2[x[0].rem](W3はwindow)
					if (status[0].hasAdd)
					{
						//W2の更新
						//W2[x[0].add] = W2[x[0].add] + W1[x[1].add, x[0].add] - W1[x[1].rem, x[0].add]
						if (status[1].hasAdd)
						{
							//W1[x[0].add] の更新
							//W1[x[1].add, x[0].add] = W1[x[1].add, x[0].add] + W0[x[2].add, x[1].add, x[0].add] - W0[x[2].rem, x[1].add, x[0].add]
							if (status2.hasAdd) //画素追加	(W1[x[0].add]) + W0[x[1].add, x[0].add]
								addPixelToWindow(*W1[k], W0_add_f, W0_add_g);
							if (status2.hasRem)//画素削除	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
								removePixelFromWindow(*W1[k], W0_rem_f, W0_rem_g);
							//W1追加	(W2) + W1[x[0].add]
							updateSubWindowAndAddToWindow(Imax, *W2[k], *W1[k]);
						}
						if (status[1].hasRem)//W1削除	(W2) - W1[x[0].rem]
							removeSubWindowFromWindow(Imax, *W2[k], *W1[k]);
						//W2追加	(W3) + W1[x[0].add]
						updateSubWindowAndAddToWindow(Imax, *Wmain[k], *W2[k]);
					}
					if (status[0].hasRem)//W2削除	(W3) - W2[x[0].rem]
						removeSubWindowFromWindow(Imax, *Wmain[k], *W2[k]);

					//中央値の計算
					if (status[0].isInside_image)
					{
						/*
						CTYPE cx;
						float dx;
						calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
						*/
						findMedian(*cx_center, *dx_center, half, *Wmain[k], *result_center);
						G_center++;
						result_center++;
						cx_center++;
						dx_center++;
					}
				}
				G_center_rowStart += size_dim[0];
				result_center_rowStart += size_dim[0];
				cx_center_rowStart += size_dim[0];
				dx_center_rowStart += size_dim[0];
				W0_add_f_rowStart += size_dim[0];
				W0_add_g_rowStart += size_dim[0];
				W0_rem_f_rowStart += size_dim[0];
				W0_rem_g_rowStart += size_dim[0];
			}
			//自身の次元+1のループが終わったときにウィンドウ内容リセット
			W2[k]->setZero();

		}


		th1.join();
		//計算したcx,dxをダウンロード
		if (status2GPU.isInside_image)
		{
			Utility::uploadMatToDevice(G[x2.center + 1], GDevice, sizeInfo, NULL);
			cu_calculateCxDx(sizeInfo, NULL, GDevice, radius_space, eps2, cxDevice, dxDevice, sumGwindow, temp);
			Utility::downloadLinearArrayAsMat(cxDevice, sizeInfo, cx);
			Utility::downloadLinearArrayAsMat(dxDevice, sizeInfo, dx);
		}


	}

	return result;
}

//3D　入力xチャンネル
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
std::vector<cv::Mat> FGMF2::filter3DIx(std::vector<cv::Mat>& Is, std::vector<cv::Mat>& G, int threadNum, int radius_space, int radius_depth, float eps2, int Imax)
{
	//check validation
	assert(Is[0].depth() == CV_32S);
	assert(G[0].depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//入力次元
	const int DIM = 3;
	//入力画像チャンネル数
	const int Ichannels = Is[0].channels();
	//中央値
	const float half = 0.5f;
	//初期化
	//サイズ
	const std::vector<int> size_dim{ Is[0].cols , Is[0].rows, (int)Is.size() };
	//半径
	const std::vector<int> r_dim{ radius_space, radius_space, radius_depth };

	//マルチスレッド用
	std::vector<int> dim0Start_vec(threadNum);
	std::vector<int> dim0End_vec(threadNum);
	std::vector<int> memoryLength_vec(threadNum);
	std::vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//結果保存
	std::vector<cv::Mat> result(Is.size());
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = cv::Mat(size_dim[1], size_dim[0], CV_32SC(Ichannels));
	}

	//W1
	std::vector < std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>>>> W1(Ichannels, std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>>>(threadNum));
	//W2
	std::vector < std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>>>> W2(Ichannels, std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>>>(threadNum));
	//W3(Wmain)
	std::vector < std::vector< std::unique_ptr< Window_single<GSum, FGSumUpToIndex, FG>>>> Wmain(Ichannels, std::vector< std::unique_ptr< Window_single<GSum, FGSumUpToIndex, FG>>>(threadNum));

	for (int c = 0; c < Ichannels; c++)
	{
#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			std::vector<int> size_dim_memory;
			copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
			size_dim_memory[0] = memoryLength_vec[k];
			W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
			W2[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 2, true)));
			Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
		}
	}

	//次の階層の処理位置セット
	Pos x2;
	DimStatus status2;
	int pixelSum2;
	setPos(x2, r_dim[2]);
	for (; x2.center < size_dim[2]; x2.center++, x2.add++, x2.rem++)
	{
		//ステータスセット
		setStatusAtOutermostLoop(x2, size_dim[2], status2);
		//1列当たりの画素数
		int pixel_sum2 = calculatePixelNumAtOutermostLoop(x2, size_dim[2], r_dim[2], status2);

		Pos vecPos;
		vecPos.center = (std::max)(0, x2.center);
		vecPos.add = (std::min)(size_dim[2] - 1, x2.add);
		vecPos.rem = (std::max)(0, x2.rem);

		//最初のチャンネル
#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			//画素数
			std::vector<int> pixel_sum(2);
			//位置
			std::vector<Pos> x(2);
			//ステータス
			std::vector<DimStatus> status(2);


			const int dim0Start = dim0Start_vec[k];
			const int dim0End = dim0End_vec[k];
			const int insideImageStart = insideImageStart_vec[k];


			//対応する画素へのポインタ（次元によって設定が異なることに注意）
			//3次元以降については、画素の座標は同じだが、vectorのどのcv::Matを取り出すのかが変わる
			GTYPE* G_center_rowStart = G[vecPos.center].ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* result_center_rowStart = result[vecPos.center].ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* W0_rem_f_rowStart = I[vecPos.rem].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_rem_g_rowStart = G[vecPos.rem].ptr<GTYPE>(0) + dim0Start + r_dim[0];
			int* W0_add_f_rowStart = I[vecPos.add].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_add_g_rowStart = G[vecPos.add].ptr<GTYPE>(0) + dim0Start + r_dim[0];

			//W1メモリ位置リセット
			W1[k]->resetPos();
			//次の階層の処理位置セット
			setPos(x[1], r_dim[1]);
			for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
			{
				//ステータスセット
				setStatus(x[1], size_dim[1], status2.isInside_image, status[1]);
				//1列当たりの画素数
				pixel_sum[1] = calculatePixelNumAtDim(x[1], size_dim[1], r_dim[1], status[1], pixel_sum2);

				//初期化
				//W3
				Wmain[k]->initialize();
				//ウィンドウ内画素数
				int pixel_sum_window = 0;
				//ウィンドウ内画素数の逆数
				float pixel_sum_window_inv = 0.0f;

				//画素へのポインタ初期化
				GTYPE* G_center = G_center_rowStart;
				int* result_center = result_center_rowStart;
				int* W0_rem_f = W0_rem_f_rowStart;
				GTYPE* W0_rem_g = W0_rem_g_rowStart;
				int* W0_add_f = W0_add_f_rowStart;
				GTYPE* W0_add_g = W0_add_g_rowStart;

				//W2メモリ位置リセット
				W2[k]->resetPos();
				//次の階層の処理位置セット
				setPosAtDim0(x[0], r_dim[0], dim0Start);
				const int remStartPos = x[0].add;
				for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
				{
					//ステータスセット
					setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);
					//window内画素数の更新
					calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

					//W3の更新
					if (status[0].hasAdd) {
						//W2の更新
						if (status[1].hasAdd) {
							//W1[x[0].add] の更新
							if (status2.hasAdd) //画素追加
								addPixelToWindow_gSum(*W1[k], W0_add_f, W0_add_g);
							if (status2.hasRem)//画素削除
								removePixelFromWindow_gSum(*W1[k], W0_rem_f, W0_rem_g);
							updateSubWindowAndAddToWindow_gSum(Imax, *W2[k], *W1[k]);//W1追加
						}
						if (status[1].hasRem)//W1削除
							removeSubWindowFromWindow_gSum(Imax, *W2[k], *W1[k]);
						updateSubWindowAndAddToWindow_gSum(Imax, *Wmain[k], *W2[k]);//W2追加
					}
					if (status[0].hasRem)//W2削除
						removeSubWindowFromWindow_gSum(Imax, *Wmain[k], *W2[k]);

					//中央値の計算
					if (status[0].isInside_image)
					{
						CTYPE cx;
						float dx;
						calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
						findMedian(cx, dx, half, *Wmain[k], *result_center);
						G_center++;
						result_center++;
					}
				}
				G_center_rowStart += size_dim[0];
				result_center_rowStart += size_dim[0];
				W0_add_f_rowStart += size_dim[0];
				W0_add_g_rowStart += size_dim[0];
				W0_rem_f_rowStart += size_dim[0];
				W0_rem_g_rowStart += size_dim[0];
			}
			//自身の次元+1のループが終わったときにウィンドウ内容リセット
			W2[k]->setZero();

		}

		//2チャンネル目以降
		for (int c = 1; c < Ichannels; c++)
		{
		}
	}

	return result;
}






//2D　入力1チャンネル 多次元化用 マルチスレッド
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF2::filter2DWindow_Save(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	/*
	多次元の場合は、ループの順が
	・N次元～3次元
	・Iのチャンネル
	・画像平面
	となるので、各チャンネルを別の関数内で平滑化する形ではなく、関数内ですべて行う
	これはGの使いまわしで効率を上げるため。

	また、そのためにマルチスレッドの扱いも異なる。
	マルチスレッド用分割もこの関数内で行う。
	メモリについてはスレッドごとに分割する。
	画像はブロックごとにアクセス開始位置を変える。

	例えば画像のサイズがh*wであり、h*wのサイズ分メモリ確保が必要な場合（３D以上の場合)、
	dim0方向の半径をrとしたとき
	各スレッド用ブロック幅がa[i]の時、各ブロックが必要なdim0方向のメモリ長さは
	a[0]+r, a[1]+2r, ... a[j-1]+2r, a[j]+r

	//画素アクセス
	//追加は[x[n-1].add, x[n-2].add, ..., x[0].add]であり
	//削除は[x[n-1].rem, x[n-2].add, ..., x[0].add]である
	要素へのアクセスは、画素かウィンドウかで異なる。
	画素の場合、多次元になると3次元以降は要素がcv::Matの多次元vectorになる。
	3次元以降の次元が1進むたびにメモリ位置を所定の値にセットして、2次元内ループではインクリメントで対応する。
	2Dの場合、I.at[x[1].add][x[0].add], I.at[x[1].rem][x[0].add],
	3Dの場合、I[x[2].add].at[x[1].add][x[0].add], I[x[2].rem].at[x[1].add][x[0].add]
	2次元内ループの開始ポインタ位置は
	2Dの場合、I.at[0]

	インクリメントで対応する場合、forループ1回のたびに必ず1インクリメントさせ、例外を生じさせないことでプログラムしやすくする。そのためには、インクリメント開始位置を適切に設定、次の行に行くときに適切にジャンプさせる必要がある。
	forループ2次元方向については、処理開始位置は
	x[1].addが0から始まり、x[1].centerがsize_dim[1]-1になるまで繰り返される。
	1次元方向については、処理開始位置は
	dim0Startからdim0Endまで繰り返される。
	開始位置は、ウィンドウを構築するための画素追加ができる最初の場所になる。半径がrであるとき、2次元以上については、開始位置（中央画素）は-rの位置から始まる。
	1次元については、マルチスレッド用に分割されているため、多少異なる。
	一番初めのブロックは、2次元以上と同様、-rから始まる。
	二番目以降のブロックは、通常の開始位置より-r前から構築が必要であるため、処理対象画素位置に対して-2rの位置から始める必要がある。

	2Dの場合、I.ptr<int>(0)を基準として、
	x[0].add = 0 + dim0St



	ウィンドウの場合は多次元を1次元にしてあり、ある開始位置から必ず順に参照される。
	なので、add,remのポインタは0にセットしておき、追加または削除のたびにインクリメントするようにしておけばよい。
	ただし特定のタイミングで0にセットしなおす必要があり、それはウィンドウの次元＋１の要素が１インクリメントされたとき。


	画素については1行次に行くときにジャンプする。
	これについては、処理中の行の先頭のポインタを記録するようにし、次の行開始時にポインタを1行分進めて（= + size_dim[0]）初期化すればよさそう。
	1行処理中には、inside_imageがTrueである中央値計算終了時にポインタをインクリメントすればよい。
	center,add, remの初期ポインタ位置は全て共通でinside_imageの開始位置にすればよい。

	*/

	const int Ichannels = I.channels();

	//check validation
	assert(I.depth() == CV_32S);
	assert(G.depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif


	//入力次元
	const int DIM = 2;

	//中央値
	const float half = 0.5f;

	//初期化
	//サイズ
	const vector<int> size_dim{ I.cols , I.rows };
	//半径
	const vector<int> r_dim{ radius, radius };


	//マルチスレッド用
	vector<int> dim0Start_vec(threadNum);
	vector<int> dim0End_vec(threadNum);
	vector<int> memoryLength_vec(threadNum);
	vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);



	//ウィンドウアクセス用
	//size_dim = {dim0, dim1, dim3, dim4}のとき
	// = {dim0, dim0*dim1, dim0*dim1*dim3}
	//とすることで、例えば [x1,x2,x3,x4]にアクセスするとき、ウィンドウメモリ上ではこれが1次元で並んでいるので
	// x4*dim0*dim1*dim3 + x3*dim0*dim1 + x2*dim0 + x1
	//const vector<int> size_prod{ size_dim[0] };


	//メモリ確保・初期化
	cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

	//W0:画素なので記録の必要なし（I,G）
	//マルチスレッド用にベクトルで確保
	//W1
	vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);

	//W2(Wmain)
	vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}


#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		//画素数
		vector<int> pixel_sum(DIM);
		//位置
		vector<Pos> x(DIM);
		//ステータス
		vector<DimStatus> status(DIM);

		const int dim0Start = dim0Start_vec[k];
		const int dim0End = dim0End_vec[k];
		const int insideImageStart = insideImageStart_vec[k];


		/*
		//画素へのポインタ初期化
		//対応する画素へのポインタ（次元によって設定が異なることに注意）
		//処理中の画素
		GTYPE* G_center = G.ptr<GTYPE>(0) + insideImageStart_vec[k];
		int* result_center = result.ptr<int>(0) + insideImageStart_vec[k];
		//n次元空間において追加、削除の対象となる画素の位置は、注目視点座標に対して
		//追加は[x[n-1].add, x[n-2].add, ..., x[0].add]であり
		//削除は[x[n-1].rem, x[n-2].add, ..., x[0].add]である
		//2次元の時は[x[1].add, x[0].add]、[x[1].rem, x[0].add]の関係だが、n次元の場合は入力がcv::Matベクトルなので、例えば
		//I[x[n-1].add][x[n-2].add]...] のcv::Matにおける[x[1].add, x[0].add]と
		//I[x[n-1].rem][x[n-2].add]...] のcv::Matにおける[x[1].add, x[0].add]になる。
		int* W0_rem_f = I.ptr<int>(0) + insideImageStart_vec[k];
		GTYPE* W0_rem_g = G.ptr<GTYPE>(0) + insideImageStart_vec[k];
		int* W0_add_f = I.ptr<int>(0) + insideImageStart_vec[k];
		GTYPE* W0_add_g = G.ptr<GTYPE>(0) + insideImageStart_vec[k];
		*/

		//対応する画素へのポインタ（次元によって設定が異なることに注意）
		//centerの行開始位置はinsideImageStart
		//addremの行開始位置はdim0Start + r_dim[0]
		//行処理終了時にif文による
		GTYPE* G_center_rowStart = G.ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
		int* result_center_rowStart = result.ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
		int* W0_rem_f_rowStart = I.ptr<int>(0) + dim0Start + r_dim[0] - (2 * r_dim[1] + 1) * size_dim[0];
		GTYPE* W0_rem_g_rowStart = G.ptr<GTYPE>(0) + dim0Start + r_dim[0] - (2 * r_dim[1] + 1) * size_dim[0];
		int* W0_add_f_rowStart = I.ptr<int>(0) + dim0Start + r_dim[0];
		GTYPE* W0_add_g_rowStart = G.ptr<GTYPE>(0) + dim0Start + r_dim[0];


		//次の階層の処理位置セット
		setPos(x[1], r_dim[1]);
		for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
		{
			//ステータスセット
			setStatusAtOutermostLoop(x[1], size_dim[1], status[1]);
			//1列当たりの画素数
			pixel_sum[1] = calculatePixelNumAtOutermostLoop(x[1], size_dim[1], r_dim[1], status[1]);


			//初期化
			//W3
			Wmain[k]->initialize();

			//ウィンドウ内画素数
			int pixel_sum_window = 0;
			//ウィンドウ内画素数の逆数
			float pixel_sum_window_inv = 0.0f;


			//画素へのポインタ初期化
			GTYPE* G_center = G_center_rowStart;
			int* result_center = result_center_rowStart;
			int* W0_rem_f = W0_rem_f_rowStart;
			GTYPE* W0_rem_g = W0_rem_g_rowStart;
			int* W0_add_f = W0_add_f_rowStart;
			GTYPE* W0_add_g = W0_add_g_rowStart;



			/*
			各階層のウィンドウに対するポインタを持つと考えれば、画素だけでなくwin系も同じように書けるのでは

			dim0方向にスライド (W0は画素)
			W1[x[1].add, x[0].add] = W1[x[1].add, x[0].add] + W0[x[2].add, x[1].add, x[0].add] - W0[x[2].rem, x[1].add, x[0].add]
			W2[x[0].add] = W2[x[0].add] + W1[x[1].add, x[0].add] - W1[x[1].rem, x[0].add]
			W3 = W3 + W2[x[0].add] - W2[x[0].rem] (W3はwindow)
			*/


			//W1メモリ位置リセット
			W1[k]->resetPos();

			//W2

			//次の階層の処理位置セット
			//dim0
			//setPos(x[0], r_dim[0], dim0Start);
			x[0].center = dim0Start;
			x[0].add = dim0Start + r_dim[0];
			x[0].rem = dim0Start - r_dim[0] - 1;
			const int remStartPos = x[0].add;
			for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
			{
				//ステータスセット
				setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);


				//window内画素数の更新
				calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

				//W2の更新
				if (status[0].hasAdd)
				{
					/*
					* addPixelToWindow_gSum するときに、ウィンドウ側は2Dならメモリ１次元だが、n次元の時はn-1次元なので、どのようにアクセスするのか。
					* と思ったがそれを1次元状に並べているのだった
					*/
					//W1[x[0].add] の更新
					if (status[1].hasAdd) //画素追加	(W1[x[0].add]) + W0[x[1].add, x[0].add]
						addPixelToWindow_gSum(*W1[k], W0_add_f, W0_add_g);
					if (status[1].hasRem)//画素削除	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
						removePixelFromWindow_gSum(*W1[k], W0_rem_f, W0_rem_g);
					//W1追加	(W2) + W1[x[0].add]
					updateSubWindowAndAddToWindow_gSum(Imax, *Wmain[k], *W1[k]);
				}
				if (status[0].hasRem)//W1削除	(W2) - W1[x[0].rem]
					removeSubWindowFromWindow_gSum(Imax, *Wmain[k], *W1[k]);

				//中央値の計算
				if (status[0].isInside_image)
				{
					CTYPE cx;
					float dx;
					calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
					findMedian(cx, dx, half, *Wmain[k], *result_center);
					G_center++;
					result_center++;
				}
			}
			G_center_rowStart += size_dim[0];
			result_center_rowStart += size_dim[0];
			W0_add_f_rowStart += size_dim[0];
			W0_add_g_rowStart += size_dim[0];
			W0_rem_f_rowStart += size_dim[0];
			W0_rem_g_rowStart += size_dim[0];
		}
	}

	return result;
}


//2D　入力1チャンネル 多次元化用 マルチスレッド
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF2::filter2DWindow2(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	const int Ichannels = I.channels();

	//check validation
	assert(I.depth() == CV_32S);
	assert(G.depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif


	//入力次元
	const int DIM = 2;
	//中央値
	const float half = 0.5f;
	//初期化
	//サイズ
	const vector<int> size_dim{ I.cols , I.rows };
	//半径
	const vector<int> r_dim{ radius, radius };

	//マルチスレッド用
	vector<int> dim0Start_vec(threadNum);
	vector<int> dim0End_vec(threadNum);
	vector<int> memoryLength_vec(threadNum);
	vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//メモリ確保・初期化
	cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

	//W1
	vector <Window_vector<GSum, FGSumUpToIndex, FG>> W1(threadNum);
	//W2(Wmain)
	vector<Window_single<GSum, FGSumUpToIndex, FG>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true);
		Wmain[k] = Window_single<GSum, FGSumUpToIndex, FG>(Imax);
	}

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		//画素数
		vector<int> pixel_sum(DIM);
		//位置
		vector<Pos> x(DIM);
		//ステータス
		vector<DimStatus> status(DIM);

		const int dim0Start = dim0Start_vec[k];
		const int dim0End = dim0End_vec[k];
		const int insideImageStart = insideImageStart_vec[k];


		//対応する画素へのポインタ（次元によって設定が異なることに注意）
		GTYPE* G_center_rowStart = G.ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
		int* result_center_rowStart = result.ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
		int* W0_rem_f_rowStart = I.ptr<int>(0) + dim0Start + r_dim[0] - (2 * r_dim[1] + 1) * size_dim[0];
		GTYPE* W0_rem_g_rowStart = G.ptr<GTYPE>(0) + dim0Start + r_dim[0] - (2 * r_dim[1] + 1) * size_dim[0];
		int* W0_add_f_rowStart = I.ptr<int>(0) + dim0Start + r_dim[0];
		GTYPE* W0_add_g_rowStart = G.ptr<GTYPE>(0) + dim0Start + r_dim[0];


		//次の階層の処理位置セット
		setPos(x[1], r_dim[1]);
		for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
		{
			//ステータスセット
			setStatusAtOutermostLoop(x[1], size_dim[1], status[1]);
			//1列当たりの画素数
			pixel_sum[1] = calculatePixelNumAtOutermostLoop(x[1], size_dim[1], r_dim[1], status[1]);


			//初期化
			//W2
			Wmain[k].initialize();
			//ウィンドウ内画素数
			int pixel_sum_window = 0;
			//ウィンドウ内画素数の逆数
			float pixel_sum_window_inv = 0.0f;


			//画素へのポインタ初期化
			GTYPE* G_center = G_center_rowStart;
			int* result_center = result_center_rowStart;
			int* W0_rem_f = W0_rem_f_rowStart;
			GTYPE* W0_rem_g = W0_rem_g_rowStart;
			int* W0_add_f = W0_add_f_rowStart;
			GTYPE* W0_add_g = W0_add_g_rowStart;



			//W1メモリ位置リセット
			W1[k].resetPos();

			//次の階層の処理位置セット
			setPosAtDim0(x[0], r_dim[0], dim0Start);
			const int remStartPos = x[0].add;
			for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
			{
				//ステータスセット
				setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);

				//window内画素数の更新
				calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

				//W2の更新
				if (status[0].hasAdd)
				{
					//W1[x[0].add] の更新
					if (status[1].hasAdd) //画素追加	(W1[x[0].add]) + W0[x[1].add, x[0].add]
						addPixelToWindow_gSum(W1[k], W0_add_f, W0_add_g);
					if (status[1].hasRem)//画素削除	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
						removePixelFromWindow_gSum(W1[k], W0_rem_f, W0_rem_g);
					//W1追加	(W2) + W1[x[0].add]
					updateSubWindowAndAddToWindow_gSum(Imax, Wmain[k], W1[k]);
				}
				if (status[0].hasRem)//W1削除	(W2) - W1[x[0].rem]
					removeSubWindowFromWindow_gSum(Imax, Wmain[k], W1[k]);

				//中央値の計算
				if (status[0].isInside_image)
				{
					CTYPE cx;
					float dx;
					calculateCxDx(Wmain[k].gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
					findMedian(cx, dx, half, Wmain[k], *result_center);
					G_center++;
					result_center++;
				}
			}
			G_center_rowStart += size_dim[0];
			result_center_rowStart += size_dim[0];
			W0_add_f_rowStart += size_dim[0];
			W0_add_g_rowStart += size_dim[0];
			W0_rem_f_rowStart += size_dim[0];
			W0_rem_g_rowStart += size_dim[0];
		}
	}

	return result;
}


#endif