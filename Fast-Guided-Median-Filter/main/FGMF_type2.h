#pragma once
//#include "FGMF_base.h"
#include "FGMF_window.h"
#include "CxDxPrecalculation.cuh"

/*
（２）列ヒストグラム使用方式 (O(1)-sliding window)
CPUのみ CPU-O(1)
ヒストグラムとインデックス以下和をもつ全ウィンドウタイプ。メリットはウィンドウサイズに非依存な計算コスト。２Dの場合、ヒストグラム更新に２５６＊２（追加と削除）＋α（インデックス移動分）かかる。

*/


class FGMF2
{
public:
	static cv::Mat filter2DInterface(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	//多次元用
	//template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	//static cv::Mat filter2DWindow(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	//マルチスレッド用
	//template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	//static cv::Mat filter2DWindow_Save(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DWindow(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DWindow2(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DWindow_saveCD(cv::Mat& I, cv::Mat& G, cv::Mat& cx, cv::Mat& dx, int threadNum, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DWindow_useCD(cv::Mat& I, cv::Mat& G, cv::Mat& cx, cv::Mat& dx, int threadNum, int radius, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DWindowI3(int cx_cv32, cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	/*
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DWindowWithCxDx(cv::Mat& I, cv::Mat& G, cv::Mat& result, int threadNum, int radius, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DWindowI3(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	*/
	//template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	//static cv::Mat filter2DWindow2(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static std::vector<cv::Mat> filter3DI1(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, int radius_space, int radius_depth, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static std::vector<cv::Mat> filter3DIx(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, int radius_space, int radius_depth, float eps2, int Imax);

	/*
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static std::vector<cv::Mat> filter3DI1withGPU(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, SizeInfo sizeInfo, int radius_space, int radius_depth, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static std::vector<cv::Mat> filter3DI1withGPUThread(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, SizeInfo sizeInfo, int radius_space, int radius_depth, float eps2, int Imax);
*/


};


//境界拡張版
void inline calculatePosForMultithread2D_Ext(const int& width, const int& colNum, const int& r_dim0, std::vector<int>& dim0Start, std::vector<int>& dim0End, std::vector<int>& memoryLength) {
	//幅の決定
	//col
	int baseWidth = width / colNum;
	int remainder = width % colNum;
	//処理開始位置
	dim0Start[0] = 0;
	//処理終了位置
	dim0End[0] = dim0Start[0] + baseWidth + (remainder > 0) - 1;
	remainder--;
	//メモリdim0方向長さ
	memoryLength[0] = dim0End[0] - dim0Start[0] + 1 + r_dim0 * 2;
	//
	for (int i = 1; i < colNum; i++)
	{
		dim0Start[i] = dim0End[i - 1] + 1;
		dim0End[i] = dim0Start[i] + baseWidth + (remainder > 0) - 1;
		remainder--;
		//
		memoryLength[i] = dim0End[i] - dim0Start[i] + 1 + r_dim0 * 2;
	}
}



template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF2::filter2DWindowI3(int cx_cv32, cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	cv::Mat result = cv::Mat(I.size(), I.depth());
	std::vector<cv::Mat> Is;
	split(I, Is);
	std::vector<cv::Mat> results(3);
	cv::Mat cx = cv::Mat(I.size(), cx_cv32);
	cv::Mat dx = cv::Mat(I.size(), CV_32FC1);
	results[0] = FGMF2::filter2DWindow_saveCD<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(Is[0], G, cx, dx, threadNum, radius, eps2, Imax);
	for (int k = 1; k <= 2; k++)
		results[k] = FGMF2::filter2DWindow_useCD<GSum, FGSumUpToIndex, FG, GTYPE, CTYPE>(Is[k], G, cx, dx, threadNum, radius, eps2, Imax);
	merge(results, result);
	return result;
}




//2D　入力1チャンネル
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF2::filter2DWindow(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
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
	const std::vector<int> size_dim{ I.cols , I.rows };
	//半径
	const std::vector<int> r_dim{ radius, radius };

	//マルチスレッド用
	std::vector<int> dim0Start_vec(threadNum);//dim0処理開始位置
	std::vector<int> dim0End_vec(threadNum);//処理終了位置
	std::vector<int> memoryLength_vec(threadNum);//メモリdim0方向長さ
	std::vector<int> insideImageStart_vec(threadNum);//メジアン計算対象画素開始位置
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//メモリ確保・初期化
	cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

	//W1
	std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
	//W2(Wmain)
	std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		std::vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		//画素数
		std::vector<int> pixel_sum(DIM);
		//位置
		std::vector<Pos> x(DIM);
		//ステータス
		std::vector<DimStatus> status(DIM);

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



			//W1メモリ位置リセット
			W1[k]->resetPos();

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

					/*
					if (x[1].center == 0)
					{
						debugging(x[0].center, x[1].center, Wmain[k]->gsum, pixel_sum_window, Wmain[k]->histo, Wmain[k]->sumUpToIndex, *result_center, cx, dx);
					}
					*/

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




//2D　入力1チャンネル cxdxを保存しながら実行（入力カラー画像用）
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF2::filter2DWindow_saveCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, int threadNum, int radius, float eps2, int Imax)
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
	const std::vector<int> size_dim{ I.cols , I.rows };
	//半径
	const std::vector<int> r_dim{ radius, radius };

	//マルチスレッド用
	std::vector<int> dim0Start_vec(threadNum);//dim0処理開始位置
	std::vector<int> dim0End_vec(threadNum);//処理終了位置
	std::vector<int> memoryLength_vec(threadNum);//メモリdim0方向長さ
	std::vector<int> insideImageStart_vec(threadNum);//メジアン計算対象画素開始位置
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//メモリ確保・初期化
	cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

	//W1
	std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
	//W2(Wmain)
	std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		std::vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		//画素数
		std::vector<int> pixel_sum(DIM);
		//位置
		std::vector<Pos> x(DIM);
		//ステータス
		std::vector<DimStatus> status(DIM);

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
		//
		CTYPE* cx_rowStart = _cx.ptr<CTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
		float* dx_rowStart = _dx.ptr<float>(0) + insideImageStart - r_dim[1] * size_dim[0];

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
			//
			CTYPE* cx = cx_rowStart;
			float* dx = dx_rowStart;



			//W1メモリ位置リセット
			W1[k]->resetPos();

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
					//CTYPE cx;
					//float dx;
					calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
					findMedian(*cx, *dx, half, *Wmain[k], *result_center);
					G_center++;
					result_center++;
					//
					cx++;
					dx++;
				}
			}
			G_center_rowStart += size_dim[0];
			result_center_rowStart += size_dim[0];
			W0_add_f_rowStart += size_dim[0];
			W0_add_g_rowStart += size_dim[0];
			W0_rem_f_rowStart += size_dim[0];
			W0_rem_g_rowStart += size_dim[0];
			//
			cx_rowStart += size_dim[0];
			dx_rowStart += size_dim[0];
		}
	}
	return result;
}



//2D　入力1チャンネル 保存したcxdxを用いて実行（入力カラー画像用）
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF2::filter2DWindow_useCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, int threadNum, int radius, float eps2, int Imax)
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
	const std::vector<int> size_dim{ I.cols , I.rows };
	//半径
	const std::vector<int> r_dim{ radius, radius };

	//マルチスレッド用
	std::vector<int> dim0Start_vec(threadNum);//dim0処理開始位置
	std::vector<int> dim0End_vec(threadNum);//処理終了位置
	std::vector<int> memoryLength_vec(threadNum);//メモリdim0方向長さ
	std::vector<int> insideImageStart_vec(threadNum);//メジアン計算対象画素開始位置
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//メモリ確保・初期化
	cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

	//W1
	std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
	//W2(Wmain)
	std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		std::vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		//画素数
		std::vector<int> pixel_sum(DIM);
		//位置
		std::vector<Pos> x(DIM);
		//ステータス
		std::vector<DimStatus> status(DIM);

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
		//
		CTYPE* cx_rowStart = _cx.ptr<CTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
		float* dx_rowStart = _dx.ptr<float>(0) + insideImageStart - r_dim[1] * size_dim[0];

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
			//
			CTYPE* cx = cx_rowStart;
			float* dx = dx_rowStart;



			//W1メモリ位置リセット
			W1[k]->resetPos();

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
					//CTYPE cx;
					//float dx;
					//calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
					findMedian(*cx, *dx, half, *Wmain[k], *result_center);
					G_center++;
					result_center++;
					//
					cx++;
					dx++;
				}
			}
			G_center_rowStart += size_dim[0];
			result_center_rowStart += size_dim[0];
			W0_add_f_rowStart += size_dim[0];
			W0_add_g_rowStart += size_dim[0];
			W0_rem_f_rowStart += size_dim[0];
			W0_rem_g_rowStart += size_dim[0];
			//
			cx_rowStart += size_dim[0];
			dx_rowStart += size_dim[0];
		}
	}
	return result;
}






//2D　入力1チャンネル 多次元化用 マルチスレッド 画像境界扱い変更
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
	const std::vector<int> size_dim{ I.cols , I.rows };
	//半径
	const std::vector<int> r_dim{ radius, radius };

	//マルチスレッド用
	std::vector<int> dim0Start_vec(threadNum);//dim0処理開始位置
	std::vector<int> dim0End_vec(threadNum);//処理終了位置
	std::vector<int> memoryLength_vec(threadNum);//メモリdim0方向長さ
	//std::vector<int> insideImageStart_vec(threadNum);//メジアン計算対象画素開始位置
	calculatePosForMultithread2D_Ext(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec);

	//メモリ確保・初期化
	cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

	//W1
	std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
	//W2(Wmain)
	std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		std::vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		//画素数
		std::vector<int> pixel_sum(DIM);
		//位置
		std::vector<Pos> x(DIM);
		//ステータス
		std::vector<DimStatus> status(DIM);

		const int dim0Start = dim0Start_vec[k];
		const int dim0End = dim0End_vec[k];
		const int dim0MemoryLength = memoryLength_vec[k];


		//対応する画素へのポインタ（次元によって設定が異なることに注意）
		//全てブロックの左上に設定する
		GTYPE* G_center_rowStart = G.ptr<GTYPE>(0) + dim0Start;
		int* result_center_rowStart = result.ptr<int>(0) + dim0Start;
		int* W0_rem_f_rowStart = I.ptr<int>(0) + dim0Start;
		GTYPE* W0_rem_g_rowStart = G.ptr<GTYPE>(0) + dim0Start;
		int* W0_add_f_rowStart = I.ptr<int>(0) + dim0Start;
		GTYPE* W0_add_g_rowStart = G.ptr<GTYPE>(0) + dim0Start;

		
		const int dim1Start = 0;
		const int dim1End = size_dim[1];

		//W1メモリ位置リセット
		W1[k]->resetPos();
		//列ヒストグラムの初期化

		//画素へのポインタ初期化
		GTYPE* G_center = G_center_rowStart;
		int* result_center = result_center_rowStart;
		int* W0_rem_f = W0_rem_f_rowStart;
		GTYPE* W0_rem_g = W0_rem_g_rowStart;
		int* W0_add_f = W0_add_f_rowStart;
		GTYPE* W0_add_g = W0_add_g_rowStart;
		for (int i = 0; i < dim0MemoryLength; i++)
		{
			addPixelToWindow_gSum(W1[k]->histo[i], W0_add_f, W0_add_g);
		}



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



			//W1メモリ位置リセット
			W1[k]->resetPos();

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



//3D　入力1チャンネル
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
std::vector<cv::Mat> FGMF2::filter3DI1(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, int radius_space, int radius_depth, float eps2, int Imax)
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
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
		W2[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 2, true)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
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
								addPixelToWindow_gSum(*W1[k], W0_add_f, W0_add_g);
							if (status2.hasRem)//画素削除	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
								removePixelFromWindow_gSum(*W1[k], W0_rem_f, W0_rem_g);
							//W1追加	(W2) + W1[x[0].add]
							updateSubWindowAndAddToWindow_gSum(Imax, *W2[k], *W1[k]);
						}
						if (status[1].hasRem)//W1削除	(W2) - W1[x[0].rem]
							removeSubWindowFromWindow_gSum(Imax, *W2[k], *W1[k]);
						//W2追加	(W3) + W1[x[0].add]
						updateSubWindowAndAddToWindow_gSum(Imax, *Wmain[k], *W2[k]);
					}
					if (status[0].hasRem)//W2削除	(W3) - W2[x[0].rem]
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
	}

	return result;
}