#pragma once
#include "FGMF_base.h"

#include "CxDxPrecalculation.cuh"

//#include "tbb/parallel_for.h"
//#include "tbb/task_scheduler_init.h"


//#define NEARLY_ZERO 0.000000000001f

//using namespace std;
//using namespace cv;

/*
この後の予定
・余計な関数の削除、テンプレートによる統合
・マルチチャンネル化
・３D化
・４D化
・実験環境構築
・（c,d をGPU計算）

2Dcolor処理の速度以下の順
（速い）
・3チャンネル個別にcx,dx計算
・1チャンネル目でcx,dx計算し、2,3チャンネル目で流用する
・3チャンネル同時計算計算で、cx,dx1チャンネル目で計算したのを続けて使う
（遅い）
計算量だけで見れば、一番下が一番少ないが、一番遅い。
これはおそらくメモリの読み込みとかキャッシュの問題。



sumuptoindex使わないテストをしたところ、10倍くらい遅かった。かなり効いているよう。


*/


class FGMF
{
public:
	//2D汎用
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D_useCD(cv::Mat& I, cv::Mat& G, cv::Mat& cx, cv::Mat& dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D_saveCD(cv::Mat& I, cv::Mat& G, cv::Mat& cx, cv::Mat& dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);

	/*
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2DColor(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2DColor(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim0Start, int dim0End, int radius, float eps2, int Imax);
	*/
	//sumuptoindex使わないテスト
	static cv::Mat filter2DInterfaceTest(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2 = 25.5f * 25.5f, int Imax = 256);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2Dtest(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);

	//Thread数を指定
	static cv::Mat filter2DInterface(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2 = 25.5f * 25.5f, int Imax = 256);




	//Single Thread
	//I1
	static cv::Mat filter2DI1G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	static cv::Mat filter2DI1G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	//I3
	static cv::Mat filter2DI3G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	static cv::Mat filter2DI3G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_SingleThread(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	//Multi Thread
	//cols (幅分割)
	//I1
	static cv::Mat filter2DI1G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	static cv::Mat filter2DI1G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	//I3
	static cv::Mat filter2DI3G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	static cv::Mat filter2DI3G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_MultiThread(int cx_cv32, cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);





	static cv::Mat filter2DI3G3_MultiThread2(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_MultiThread2(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	static cv::Mat filter2DI3G3_MultiThread3(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_MultiThread3(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);


	//cx,dx GPU計算
	static void calculateCxDxOnGPU(cv::Mat& G, int radius, float eps2, SizeInfo& sizeInfo, cv::Mat& cx, cv::Mat& dx);

	static cv::Mat gpuTestCxDx(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax, SizeInfo& sizeInfo);



};











////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF::filter2D(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
{
	//check validation
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && ((std::is_same<GTYPE, int>::value && G.channels() == 1) || (std::is_same<GTYPE, cv::Vec3i>::value && G.channels() == 3)));
	assert(I.isContinuous() && G.isContinuous());
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//中央値
	const float half = 0.5f;

	//初期化
	const int size_dim0 = I.cols;
	const int size_dim1 = I.rows;

	//
	const int r_dim0 = radius;
	const int r_dim1 = radius;

	/*
	* 以下の範囲の〜X　はXを含む（X以下）
	* 処理対象画素範囲は [dim1Start 〜 dim1End, dim0Start 〜 dim0End]
	* ウィンドウ範囲は[max(0, dim1Start - r_dim1) 〜 (std::min)(dim1End, size_dim1) , [max(0, dim0Start) 〜 (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1方向ウィンドウ上端、下端
	const int upmostWindowDim1 = std::max(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0方向のウィンドウ左端、右端
	const int leftmostWindowDim0 = std::max(0, dim0Start - r_dim0);
	const int rightmostWindowDim0 = std::min(size_dim0 - 1, dim0End + r_dim0);
	//メモリ幅
	const int memoryLength = rightmostWindowDim0 - leftmostWindowDim0 + 1;
	//１行目用 dim1方向ウィンドウ下端
	const int bottomWindowDim1ForFirstLine = std::min(size_dim1 - 1, dim1Start + r_dim1);
	//１列目用 dim0方向ウィンドウ右端
	const int rightWindowDim0ForFirstLine = std::min(size_dim0 - 1, dim0Start + r_dim0);
	//(1,X)の時のウィンドウ幅
	int pixel_num_dim0_firstLine = rightWindowDim0ForFirstLine - leftmostWindowDim0 + 1;


	//メモリ確保・初期化

	//各列用ヒストグラム格納変数
	FG** histo_win1 = new FG *[memoryLength];
	for (int i = 0; i < memoryLength; i++)
	{
		histo_win1[i] = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(FG) * Imax);
	}
	//sum(g), sum(g*g)逐次計算用変数
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);
	//index_win1以下の列ヒストグラム和
	FGSumUpToIndex* sumUpToIndex_win1 = new FGSumUpToIndex[memoryLength];
	memset(sumUpToIndex_win1, 0, sizeof(FGSumUpToIndex) * memoryLength);

	//Window内ヒストグラム
	FG* histo_window = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);



	///////////////////////////////
	//対応する画素へのポインタ
	//処理中の画素
	GTYPE* G_center = G.ptr<GTYPE>(dim1Start) + dim0Start;
	int* result_center = result.ptr<int>(dim1Start) + dim0Start;
	//ポインタジャンプ幅
	const int stepForNextRow = size_dim0 - (dim0End - dim0Start + 1);
	const int stepForNextRow2 = stepForNextRow + 1;
	///////////////////////////////


	//1行目だけ別処理
	{
		//初期化
		//Window関係
		GSum gSum_window = { 0 };
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(dim1Start);

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(FG) * Imax);
		//1列当たりの画素数
		int pixel_sum_dim1 = bottomWindowDim1ForFirstLine - upmostWindowDim1 + 1;
		//ウィンドウ内画素数
		int pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;



		//メモリ用
		int x_add_mem = 0;
		int x_rem_mem = 0;

		//さらに１列目だけ別処理
		//(1,1)処理
		{
			// leftmostWindowDim0〜rightWindowDim1ForFirstLine のヒストグラムを構築
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				//1行目用処理　列すべて追加
				for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx], gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}
			//中央値の計算
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				G_center++;
				result_center++;
			}
		}
		//(2,1)〜
		{
			int x_add_dim0 = r_dim0 + 1 + dim0Start;// 追加される列
			int x_rem_dim0 = -r_dim0 + dim0Start;// 削除される列
			int x_dim0 = 1 + dim0Start;// 処理対象中心画素列

			for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
			{
				//追加する次の列があるか
				const int hasAdd_dim0 = x_add_dim0 < size_dim0;
				//削除する前の列があるか
				const int hasRem_dim0 = x_rem_dim0 >= 0;
				//処理位置が画像内か
				const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
				const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

				//window内画素数の更新
				if (hasAddOnly_dim0)
					addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//追加のみ
				else if (hasRemOnly_dim0)
					subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//削除のみ

				//次の列があるなら更新
				if (hasAdd_dim0) {
					//1行目用処理　列すべて追加
					for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//次の行をヒストグラムに追加
						addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0], gSum_win1[x_add_mem]);

					//subwindowを更新し、それをメインwindowのヒストグラムに追加
					updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
					x_add_mem++;
				}
				//削除列があるなら
				if (hasRem_dim0)
				{
					//メインwindowからsubwindowのヒストグラムを削除
					removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//中央値の計算
				{
					findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
					G_center++;
					result_center++;
				}
			}
			G_center += stepForNextRow;
			result_center += stepForNextRow;
		}
	}
	//初期化
	int x_add_dim1 = r_dim1 + 1 + dim1Start;//追加される列
	int x_rem_dim1 = -r_dim1 + dim1Start;//削除される行
	int x_dim1 = 1 + dim1Start;// 処理対象中心画素行
	//
	//列の削除画素
	int* I_rem = I.ptr<int>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_rem = G.ptr<GTYPE>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	//列の追加画素
	int* I_add = I.ptr<int>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_add = G.ptr<GTYPE>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;

	/*
	I_rem = I.ptr<int>(x_dim1 - r_dim1 - 1)[x_add_dim0]
	I_add = I.ptr<int>(x_dim1 + r_dim1)[x_add_dim0]

	*/

	//2行目以降処理開始
	for (; x_dim1 <= dim1End; x_dim1++, x_add_dim1++, x_rem_dim1++)
	{
		const int hasAdd_dim1 = x_add_dim1 < size_dim1;
		const int hasRem_dim1 = x_rem_dim1 >= 0;
		const int hasAddOnly_dim1 = !hasRem_dim1 && hasAdd_dim1;
		const int hasRemOnly_dim1 = hasRem_dim1 && !hasAdd_dim1;

		//初期化
		//Window関係
		GSum gSum_window = { 0 };
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(x_dim1);

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(FG) * Imax);
		//ウィンドウ内画素数
		int pixel_sum_window = 0;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 0.0f;
		//1列当たりの画素数
		int pixel_sum_dim1;

		if (hasAddOnly_dim1)
			pixel_sum_dim1 = x_dim1 + r_dim1 + 1;
		else if (hasRemOnly_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + r_dim1;
		else
			pixel_sum_dim1 = 2 * r_dim1 + 1;



		//メモリ用
		int x_add_mem = 0;
		int x_rem_mem = 0;


		//1列目処理
		{
			//ウィンドウ内画素数
			pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
			//ウィンドウ内画素数の逆数
			pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;

			// leftmostWindowDim0〜rightWindowDim1ForFirstLine のヒストグラムを構築
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				if (hasRem_dim1)//前の行をヒストグラムから削除
					removePixelFromWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx], gSum_win1[x_add_mem]);
				if (hasAdd_dim1)//次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx], gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//中央値の計算
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				G_center++;
				result_center++;
			}

		}
		//2列目以降処理
		int x_add_dim0 = r_dim0 + 1 + dim0Start;// 追加される列
		int x_rem_dim0 = -r_dim0 + dim0Start;// 削除される列
		int x_dim0 = 1 + dim0Start;// 処理対象中心画素列
		for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
		{
			const int isNotFirstDim0 = x_dim0 != dim0Start;

			//追加する次の列があるか
			const int hasAdd_dim0 = x_add_dim0 < size_dim0;
			//削除する前の列があるか
			const int hasRem_dim0 = x_rem_dim0 >= 0;
			//処理位置が画像内か
			const int isInside_image = x_dim0 >= dim0Start;
			const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
			const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

			//window内画素数の更新
			if (hasAddOnly_dim0)
				addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//追加のみ
			else if (hasRemOnly_dim0)
				subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//削除のみ


			//次の列があるなら更新
			if (hasAdd_dim0) {
				//前の行があるなら更新
				if (hasRem_dim1)//前の行をヒストグラムから削除
					removePixelFromWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_rem, *G_rem, gSum_win1[x_add_mem]);
				//次の行があるなら更新
				if (hasAdd_dim1) //次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_add, *G_add, gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//削除列があるなら
			if (hasRem_dim0)
			{
				//メインwindowからsubwindowのヒストグラムを削除
				removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//中央値の計算
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				G_center++;
				result_center++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		G_center += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//メモリ開放
	for (int i = 0; i < memoryLength; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] gSum_win1;
	delete[] sumUpToIndex_win1;

	//return result;
}


template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF::filter2D_useCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
{
	//check validation
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && ((std::is_same<GTYPE, int>::value && G.channels() == 1) || (std::is_same<GTYPE, cv::Vec3i>::value && G.channels() == 3)));
	assert(I.isContinuous() && G.isContinuous());
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//中央値
	const float half = 0.5f;

	//初期化
	const int size_dim0 = I.cols;
	const int size_dim1 = I.rows;

	//
	const int r_dim0 = radius;
	const int r_dim1 = radius;

	/*
	* 以下の範囲の〜X　はXを含む（X以下）
	* 処理対象画素範囲は [dim1Start 〜 dim1End, dim0Start 〜 dim0End]
	* ウィンドウ範囲は[max(0, dim1Start - r_dim1) 〜 (std::min)(dim1End, size_dim1) , [max(0, dim0Start) 〜 (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1方向ウィンドウ上端、下端
	const int upmostWindowDim1 = (std::max)(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0方向のウィンドウ左端、右端
	const int leftmostWindowDim0 = (std::max)(0, dim0Start - r_dim0);
	const int rightmostWindowDim0 = std::min(size_dim0 - 1, dim0End + r_dim0);
	//メモリ幅
	const int memoryLength = rightmostWindowDim0 - leftmostWindowDim0 + 1;
	//１行目用 dim1方向ウィンドウ下端
	const int bottomWindowDim1ForFirstLine = std::min(size_dim1 - 1, dim1Start + r_dim1);
	//１列目用 dim0方向ウィンドウ右端
	const int rightWindowDim0ForFirstLine = std::min(size_dim0 - 1, dim0Start + r_dim0);
	//(1,X)の時のウィンドウ幅
	int pixel_num_dim0_firstLine = rightWindowDim0ForFirstLine - leftmostWindowDim0 + 1;


	//メモリ確保・初期化

	//各列用ヒストグラム格納変数
	FG** histo_win1 = new FG *[memoryLength];
	for (int i = 0; i < memoryLength; i++)
	{
		histo_win1[i] = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(FG) * Imax);
	}
	//index_win1以下の列ヒストグラム和
	FGSumUpToIndex* sumUpToIndex_win1 = new FGSumUpToIndex[memoryLength];
	memset(sumUpToIndex_win1, 0, sizeof(FGSumUpToIndex) * memoryLength);

	//Window内ヒストグラム
	FG* histo_window = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);



	///////////////////////////////
	//対応する画素へのポインタ
	//処理中の画素
	int* result_center = result.ptr<int>(dim1Start) + dim0Start;
	CTYPE* cx = _cx.ptr<CTYPE>(dim1Start) + dim0Start;
	float* dx = _dx.ptr<float>(dim1Start) + dim0Start;
	//ポインタジャンプ幅
	const int stepForNextRow = size_dim0 - (dim0End - dim0Start + 1);
	const int stepForNextRow2 = stepForNextRow + 1;
	///////////////////////////////


	//1行目だけ別処理
	{
		//初期化
		//Window関係
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(dim1Start);

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(FG) * Imax);
		//1列当たりの画素数
		int pixel_sum_dim1 = bottomWindowDim1ForFirstLine - upmostWindowDim1 + 1;
		//ウィンドウ内画素数
		int pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;



		//メモリ用
		int x_add_mem = 0;
		int x_rem_mem = 0;

		//さらに１列目だけ別処理
		//(1,1)処理
		{
			// leftmostWindowDim0〜rightWindowDim1ForFirstLine のヒストグラムを構築
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				//1行目用処理　列すべて追加
				for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//次の行をヒストグラムに追加
					addPixelToWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem]);
				x_add_mem++;
			}
			//中央値の計算
			{
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				result_center++;
				cx++;
				dx++;
			}
		}
		//(2,1)〜
		{
			int x_add_dim0 = r_dim0 + 1 + dim0Start;// 追加される列
			int x_rem_dim0 = -r_dim0 + dim0Start;// 削除される列
			int x_dim0 = 1 + dim0Start;// 処理対象中心画素列

			for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
			{
				//追加する次の列があるか
				const int hasAdd_dim0 = x_add_dim0 < size_dim0;
				//削除する前の列があるか
				const int hasRem_dim0 = x_rem_dim0 >= 0;
				//処理位置が画像内か
				const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
				const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

				//window内画素数の更新
				if (hasAddOnly_dim0)
					addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//追加のみ
				else if (hasRemOnly_dim0)
					subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//削除のみ

				//次の列があるなら更新
				if (hasAdd_dim0) {
					//1行目用処理　列すべて追加
					for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//次の行をヒストグラムに追加
						addPixelToWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0]);

					//subwindowを更新し、それをメインwindowのヒストグラムに追加
					updateSubWindowAndAddToWindow(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem]);
					x_add_mem++;
				}
				//削除列があるなら
				if (hasRem_dim0)
				{
					//メインwindowからsubwindowのヒストグラムを削除
					removeSubWindowFromWindow(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//中央値の計算
				{
					findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
					result_center++;
					cx++;
					dx++;
				}
			}
			cx += stepForNextRow;
			dx += stepForNextRow;
			result_center += stepForNextRow;
		}
	}
	//初期化
	int x_add_dim1 = r_dim1 + 1 + dim1Start;//追加される列
	int x_rem_dim1 = -r_dim1 + dim1Start;//削除される行
	int x_dim1 = 1 + dim1Start;// 処理対象中心画素行
	//
	//列の削除画素
	int* I_rem = I.ptr<int>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_rem = G.ptr<GTYPE>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	//列の追加画素
	int* I_add = I.ptr<int>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_add = G.ptr<GTYPE>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;

	/*
	I_rem = I.ptr<int>(x_dim1 - r_dim1 - 1)[x_add_dim0]
	I_add = I.ptr<int>(x_dim1 + r_dim1)[x_add_dim0]

	*/

	//2行目以降処理開始
	for (; x_dim1 <= dim1End; x_dim1++, x_add_dim1++, x_rem_dim1++)
	{
		const int hasAdd_dim1 = x_add_dim1 < size_dim1;
		const int hasRem_dim1 = x_rem_dim1 >= 0;
		const int hasAddOnly_dim1 = !hasRem_dim1 && hasAdd_dim1;
		const int hasRemOnly_dim1 = hasRem_dim1 && !hasAdd_dim1;

		//初期化
		//Window関係
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(x_dim1);

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(FG) * Imax);
		//ウィンドウ内画素数
		int pixel_sum_window = 0;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 0.0f;
		//1列当たりの画素数
		int pixel_sum_dim1;

		if (hasAddOnly_dim1)
			pixel_sum_dim1 = x_dim1 + r_dim1 + 1;
		else if (hasRemOnly_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + r_dim1;
		else
			pixel_sum_dim1 = 2 * r_dim1 + 1;



		//メモリ用
		int x_add_mem = 0;
		int x_rem_mem = 0;


		//1列目処理
		{
			//ウィンドウ内画素数
			pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
			//ウィンドウ内画素数の逆数
			pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;

			// leftmostWindowDim0〜rightWindowDim1ForFirstLine のヒストグラムを構築
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				if (hasRem_dim1)//前の行をヒストグラムから削除
					removePixelFromWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx]);
				if (hasAdd_dim1)//次の行をヒストグラムに追加
					addPixelToWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem]);
				x_add_mem++;
			}

			//中央値の計算
			{
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				result_center++;
				cx++;
				dx++;
			}

		}
		//2列目以降処理
		int x_add_dim0 = r_dim0 + 1 + dim0Start;// 追加される列
		int x_rem_dim0 = -r_dim0 + dim0Start;// 削除される列
		int x_dim0 = 1 + dim0Start;// 処理対象中心画素列
		for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
		{
			const int isNotFirstDim0 = x_dim0 != dim0Start;

			//追加する次の列があるか
			const int hasAdd_dim0 = x_add_dim0 < size_dim0;
			//削除する前の列があるか
			const int hasRem_dim0 = x_rem_dim0 >= 0;
			//処理位置が画像内か
			const int isInside_image = x_dim0 >= dim0Start;
			const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
			const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

			//window内画素数の更新
			if (hasAddOnly_dim0)
				addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//追加のみ
			else if (hasRemOnly_dim0)
				subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//削除のみ


			//次の列があるなら更新
			if (hasAdd_dim0) {
				//前の行があるなら更新
				if (hasRem_dim1)//前の行をヒストグラムから削除
					removePixelFromWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_rem, *G_rem);
				//次の行があるなら更新
				if (hasAdd_dim1) //次の行をヒストグラムに追加
					addPixelToWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_add, *G_add);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem]);
				x_add_mem++;
			}

			//削除列があるなら
			if (hasRem_dim0)
			{
				//メインwindowからsubwindowのヒストグラムを削除
				removeSubWindowFromWindow(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//中央値の計算
			{
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				result_center++;
				cx++;
				dx++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		cx += stepForNextRow;
		dx += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//メモリ開放
	for (int i = 0; i < memoryLength; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] sumUpToIndex_win1;

	//return result;
}


template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF::filter2D_saveCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
{
	//check validation
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && ((std::is_same<GTYPE, int>::value && G.channels() == 1) || (std::is_same<GTYPE, cv::Vec3i>::value && G.channels() == 3)));
	assert(I.isContinuous() && G.isContinuous());
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//中央値
	const float half = 0.5f;

	//初期化
	const int size_dim0 = I.cols;
	const int size_dim1 = I.rows;

	//
	const int r_dim0 = radius;
	const int r_dim1 = radius;

	/*
	* 以下の範囲の〜X　はXを含む（X以下）
	* 処理対象画素範囲は [dim1Start 〜 dim1End, dim0Start 〜 dim0End]
	* ウィンドウ範囲は[max(0, dim1Start - r_dim1) 〜 (std::min)(dim1End, size_dim1) , [max(0, dim0Start) 〜 (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1方向ウィンドウ上端、下端
	const int upmostWindowDim1 = (std::max)(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0方向のウィンドウ左端、右端
	const int leftmostWindowDim0 = (std::max)(0, dim0Start - r_dim0);
	const int rightmostWindowDim0 = std::min(size_dim0 - 1, dim0End + r_dim0);
	//メモリ幅
	const int memoryLength = rightmostWindowDim0 - leftmostWindowDim0 + 1;
	//１行目用 dim1方向ウィンドウ下端
	const int bottomWindowDim1ForFirstLine = std::min(size_dim1 - 1, dim1Start + r_dim1);
	//１列目用 dim0方向ウィンドウ右端
	const int rightWindowDim0ForFirstLine = std::min(size_dim0 - 1, dim0Start + r_dim0);
	//(1,X)の時のウィンドウ幅
	int pixel_num_dim0_firstLine = rightWindowDim0ForFirstLine - leftmostWindowDim0 + 1;


	//メモリ確保・初期化

	//各列用ヒストグラム格納変数
	FG** histo_win1 = new FG *[memoryLength];
	for (int i = 0; i < memoryLength; i++)
	{
		histo_win1[i] = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(FG) * Imax);
	}
	//sum(g), sum(g*g)逐次計算用変数
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);
	//index_win1以下の列ヒストグラム和
	FGSumUpToIndex* sumUpToIndex_win1 = new FGSumUpToIndex[memoryLength];
	memset(sumUpToIndex_win1, 0, sizeof(FGSumUpToIndex) * memoryLength);

	//Window内ヒストグラム
	FG* histo_window = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);



	///////////////////////////////
	//対応する画素へのポインタ
	//処理中の画素
	GTYPE* G_center = G.ptr<GTYPE>(dim1Start) + dim0Start;
	int* result_center = result.ptr<int>(dim1Start) + dim0Start;
	CTYPE* cx = _cx.ptr<CTYPE>(dim1Start) + dim0Start;
	float* dx = _dx.ptr<float>(dim1Start) + dim0Start;
	//ポインタジャンプ幅
	const int stepForNextRow = size_dim0 - (dim0End - dim0Start + 1);
	const int stepForNextRow2 = stepForNextRow + 1;
	///////////////////////////////


	//1行目だけ別処理
	{
		//初期化
		//Window関係
		GSum gSum_window = { 0 };
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(dim1Start);

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(FG) * Imax);
		//1列当たりの画素数
		int pixel_sum_dim1 = bottomWindowDim1ForFirstLine - upmostWindowDim1 + 1;
		//ウィンドウ内画素数
		int pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;



		//メモリ用
		int x_add_mem = 0;
		int x_rem_mem = 0;

		//さらに１列目だけ別処理
		//(1,1)処理
		{
			// leftmostWindowDim0〜rightWindowDim1ForFirstLine のヒストグラムを構築
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				//1行目用処理　列すべて追加
				for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx], gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}
			//中央値の計算
			{
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}
		}
		//(2,1)〜
		{
			int x_add_dim0 = r_dim0 + 1 + dim0Start;// 追加される列
			int x_rem_dim0 = -r_dim0 + dim0Start;// 削除される列
			int x_dim0 = 1 + dim0Start;// 処理対象中心画素列

			for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
			{
				//追加する次の列があるか
				const int hasAdd_dim0 = x_add_dim0 < size_dim0;
				//削除する前の列があるか
				const int hasRem_dim0 = x_rem_dim0 >= 0;
				//処理位置が画像内か
				const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
				const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

				//window内画素数の更新
				if (hasAddOnly_dim0)
					addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//追加のみ
				else if (hasRemOnly_dim0)
					subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//削除のみ

				//次の列があるなら更新
				if (hasAdd_dim0) {
					//1行目用処理　列すべて追加
					for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//次の行をヒストグラムに追加
						addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0], gSum_win1[x_add_mem]);

					//subwindowを更新し、それをメインwindowのヒストグラムに追加
					updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
					x_add_mem++;
				}
				//削除列があるなら
				if (hasRem_dim0)
				{
					//メインwindowからsubwindowのヒストグラムを削除
					removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//中央値の計算
				{
					//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
					calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
					findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
					cx++;
					dx++;
					G_center++;
					result_center++;
				}
			}
			cx += stepForNextRow;
			dx += stepForNextRow;
			G_center += stepForNextRow;
			result_center += stepForNextRow;
		}
	}
	//初期化
	int x_add_dim1 = r_dim1 + 1 + dim1Start;//追加される列
	int x_rem_dim1 = -r_dim1 + dim1Start;//削除される行
	int x_dim1 = 1 + dim1Start;// 処理対象中心画素行
	//
	//列の削除画素
	int* I_rem = I.ptr<int>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_rem = G.ptr<GTYPE>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	//列の追加画素
	int* I_add = I.ptr<int>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_add = G.ptr<GTYPE>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;

	/*
	I_rem = I.ptr<int>(x_dim1 - r_dim1 - 1)[x_add_dim0]
	I_add = I.ptr<int>(x_dim1 + r_dim1)[x_add_dim0]

	*/

	//2行目以降処理開始
	for (; x_dim1 <= dim1End; x_dim1++, x_add_dim1++, x_rem_dim1++)
	{
		const int hasAdd_dim1 = x_add_dim1 < size_dim1;
		const int hasRem_dim1 = x_rem_dim1 >= 0;
		const int hasAddOnly_dim1 = !hasRem_dim1 && hasAdd_dim1;
		const int hasRemOnly_dim1 = hasRem_dim1 && !hasAdd_dim1;

		//初期化
		//Window関係
		GSum gSum_window = { 0 };
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(x_dim1);

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(FG) * Imax);
		//ウィンドウ内画素数
		int pixel_sum_window = 0;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 0.0f;
		//1列当たりの画素数
		int pixel_sum_dim1;

		if (hasAddOnly_dim1)
			pixel_sum_dim1 = x_dim1 + r_dim1 + 1;
		else if (hasRemOnly_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + r_dim1;
		else
			pixel_sum_dim1 = 2 * r_dim1 + 1;



		//メモリ用
		int x_add_mem = 0;
		int x_rem_mem = 0;


		//1列目処理
		{
			//ウィンドウ内画素数
			pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
			//ウィンドウ内画素数の逆数
			pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;

			// leftmostWindowDim0〜rightWindowDim1ForFirstLine のヒストグラムを構築
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				if (hasRem_dim1)//前の行をヒストグラムから削除
					removePixelFromWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx], gSum_win1[x_add_mem]);
				if (hasAdd_dim1)//次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx], gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//中央値の計算
			{
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}

		}
		//2列目以降処理
		int x_add_dim0 = r_dim0 + 1 + dim0Start;// 追加される列
		int x_rem_dim0 = -r_dim0 + dim0Start;// 削除される列
		int x_dim0 = 1 + dim0Start;// 処理対象中心画素列
		for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
		{
			const int isNotFirstDim0 = x_dim0 != dim0Start;

			//追加する次の列があるか
			const int hasAdd_dim0 = x_add_dim0 < size_dim0;
			//削除する前の列があるか
			const int hasRem_dim0 = x_rem_dim0 >= 0;
			//処理位置が画像内か
			const int isInside_image = x_dim0 >= dim0Start;
			const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
			const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

			//window内画素数の更新
			if (hasAddOnly_dim0)
				addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//追加のみ
			else if (hasRemOnly_dim0)
				subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//削除のみ


			//次の列があるなら更新
			if (hasAdd_dim0) {
				//前の行があるなら更新
				if (hasRem_dim1)//前の行をヒストグラムから削除
					removePixelFromWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_rem, *G_rem, gSum_win1[x_add_mem]);
				//次の行があるなら更新
				if (hasAdd_dim1) //次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_add, *G_add, gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//削除列があるなら
			if (hasRem_dim0)
			{
				//メインwindowからsubwindowのヒストグラムを削除
				removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//中央値の計算
			{
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		cx += stepForNextRow;
		dx += stepForNextRow;
		G_center += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//メモリ開放
	for (int i = 0; i < memoryLength; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] gSum_win1;
	delete[] sumUpToIndex_win1;

	//return result;
}







//sumuptoindex無しテスト
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF::filter2Dtest(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
{
	//check validation
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && ((std::is_same<GTYPE, int>::value && G.channels() == 1) || (std::is_same<GTYPE, cv::Vec3i>::value && G.channels() == 3)));
	assert(I.isContinuous() && G.isContinuous());
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//中央値
	const float half = 0.5f;

	//初期化
	const int size_dim0 = I.cols;
	const int size_dim1 = I.rows;

	//
	const int r_dim0 = radius;
	const int r_dim1 = radius;

	/*
	* 以下の範囲の〜X　はXを含む（X以下）
	* 処理対象画素範囲は [dim1Start 〜 dim1End, dim0Start 〜 dim0End]
	* ウィンドウ範囲は[max(0, dim1Start - r_dim1) 〜 (std::min)(dim1End, size_dim1) , [max(0, dim0Start) 〜 (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1方向ウィンドウ上端、下端
	const int upmostWindowDim1 = std::max(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0方向のウィンドウ左端、右端
	const int leftmostWindowDim0 = std::max(0, dim0Start - r_dim0);
	const int rightmostWindowDim0 = std::min(size_dim0 - 1, dim0End + r_dim0);
	//メモリ幅
	const int memoryLength = rightmostWindowDim0 - leftmostWindowDim0 + 1;
	//１行目用 dim1方向ウィンドウ下端
	const int bottomWindowDim1ForFirstLine = std::min(size_dim1 - 1, dim1Start + r_dim1);
	//１列目用 dim0方向ウィンドウ右端
	const int rightWindowDim0ForFirstLine = std::min(size_dim0 - 1, dim0Start + r_dim0);
	//(1,X)の時のウィンドウ幅
	int pixel_num_dim0_firstLine = rightWindowDim0ForFirstLine - leftmostWindowDim0 + 1;


	//メモリ確保・初期化

	//各列用ヒストグラム格納変数
	FG** histo_win1 = new FG *[memoryLength];
	for (int i = 0; i < memoryLength; i++)
	{
		histo_win1[i] = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(FG) * Imax);
	}
	//sum(g), sum(g*g)逐次計算用変数
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);

	//Window内ヒストグラム
	FG* histo_window = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);



	///////////////////////////////
	//対応する画素へのポインタ
	//処理中の画素
	GTYPE* G_center = G.ptr<GTYPE>(dim1Start) + dim0Start;
	int* result_center = result.ptr<int>(dim1Start) + dim0Start;
	//ポインタジャンプ幅
	const int stepForNextRow = size_dim0 - (dim0End - dim0Start + 1);
	const int stepForNextRow2 = stepForNextRow + 1;
	///////////////////////////////


	//1行目だけ別処理
	{
		//初期化
		//Window関係
		GSum gSum_window = { 0 };

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(FG) * Imax);
		//1列当たりの画素数
		int pixel_sum_dim1 = bottomWindowDim1ForFirstLine - upmostWindowDim1 + 1;
		//ウィンドウ内画素数
		int pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;



		//メモリ用
		int x_add_mem = 0;
		int x_rem_mem = 0;

		//さらに１列目だけ別処理
		//(1,1)処理
		{
			// leftmostWindowDim0〜rightWindowDim1ForFirstLine のヒストグラムを構築
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				//1行目用処理　列すべて追加
				for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx], gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}
			//中央値の計算
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				G_center++;
				result_center++;
			}
		}
		//(2,1)〜
		{
			int x_add_dim0 = r_dim0 + 1 + dim0Start;// 追加される列
			int x_rem_dim0 = -r_dim0 + dim0Start;// 削除される列
			int x_dim0 = 1 + dim0Start;// 処理対象中心画素列

			for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
			{
				//追加する次の列があるか
				const int hasAdd_dim0 = x_add_dim0 < size_dim0;
				//削除する前の列があるか
				const int hasRem_dim0 = x_rem_dim0 >= 0;
				//処理位置が画像内か
				const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
				const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

				//window内画素数の更新
				if (hasAddOnly_dim0)
					addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//追加のみ
				else if (hasRemOnly_dim0)
					subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//削除のみ

				//次の列があるなら更新
				if (hasAdd_dim0) {
					//1行目用処理　列すべて追加
					for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//次の行をヒストグラムに追加
						addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0], gSum_win1[x_add_mem]);

					//subwindowを更新し、それをメインwindowのヒストグラムに追加
					updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
					x_add_mem++;
				}
				//削除列があるなら
				if (hasRem_dim0)
				{
					//メインwindowからsubwindowのヒストグラムを削除
					removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//中央値の計算
				{
					findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
					G_center++;
					result_center++;
				}
			}
			G_center += stepForNextRow;
			result_center += stepForNextRow;
		}
	}
	//初期化
	int x_add_dim1 = r_dim1 + 1 + dim1Start;//追加される列
	int x_rem_dim1 = -r_dim1 + dim1Start;//削除される行
	int x_dim1 = 1 + dim1Start;// 処理対象中心画素行
	//
	//列の削除画素
	int* I_rem = I.ptr<int>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_rem = G.ptr<GTYPE>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	//列の追加画素
	int* I_add = I.ptr<int>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_add = G.ptr<GTYPE>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;

	/*
	I_rem = I.ptr<int>(x_dim1 - r_dim1 - 1)[x_add_dim0]
	I_add = I.ptr<int>(x_dim1 + r_dim1)[x_add_dim0]

	*/

	//2行目以降処理開始
	for (; x_dim1 <= dim1End; x_dim1++, x_add_dim1++, x_rem_dim1++)
	{
		const int hasAdd_dim1 = x_add_dim1 < size_dim1;
		const int hasRem_dim1 = x_rem_dim1 >= 0;
		const int hasAddOnly_dim1 = !hasRem_dim1 && hasAdd_dim1;
		const int hasRemOnly_dim1 = hasRem_dim1 && !hasAdd_dim1;

		//初期化
		//Window関係
		GSum gSum_window = { 0 };

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(FG) * Imax);
		//ウィンドウ内画素数
		int pixel_sum_window = 0;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 0.0f;
		//1列当たりの画素数
		int pixel_sum_dim1;

		if (hasAddOnly_dim1)
			pixel_sum_dim1 = x_dim1 + r_dim1 + 1;
		else if (hasRemOnly_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + r_dim1;
		else
			pixel_sum_dim1 = 2 * r_dim1 + 1;



		//メモリ用
		int x_add_mem = 0;
		int x_rem_mem = 0;


		//1列目処理
		{
			//ウィンドウ内画素数
			pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
			//ウィンドウ内画素数の逆数
			pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;

			// leftmostWindowDim0〜rightWindowDim1ForFirstLine のヒストグラムを構築
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				if (hasRem_dim1)//前の行をヒストグラムから削除
					removePixelFromWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx], gSum_win1[x_add_mem]);
				if (hasAdd_dim1)//次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx], gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//中央値の計算
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				G_center++;
				result_center++;
			}

		}
		//2列目以降処理
		int x_add_dim0 = r_dim0 + 1 + dim0Start;// 追加される列
		int x_rem_dim0 = -r_dim0 + dim0Start;// 削除される列
		int x_dim0 = 1 + dim0Start;// 処理対象中心画素列
		for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
		{
			const int isNotFirstDim0 = x_dim0 != dim0Start;

			//追加する次の列があるか
			const int hasAdd_dim0 = x_add_dim0 < size_dim0;
			//削除する前の列があるか
			const int hasRem_dim0 = x_rem_dim0 >= 0;
			//処理位置が画像内か
			const int isInside_image = x_dim0 >= dim0Start;
			const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
			const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

			//window内画素数の更新
			if (hasAddOnly_dim0)
				addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//追加のみ
			else if (hasRemOnly_dim0)
				subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//削除のみ


			//次の列があるなら更新
			if (hasAdd_dim0) {
				//前の行があるなら更新
				if (hasRem_dim1)//前の行をヒストグラムから削除
					removePixelFromWindow_gSum(histo_win1[x_add_mem], *I_rem, *G_rem, gSum_win1[x_add_mem]);
				//次の行があるなら更新
				if (hasAdd_dim1) //次の行をヒストグラムに追加
					addPixelToWindow_gSum(histo_win1[x_add_mem], *I_add, *G_add, gSum_win1[x_add_mem]);

				//subwindowを更新し、それをメインwindowのヒストグラムに追加
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//削除列があるなら
			if (hasRem_dim0)
			{
				//メインwindowからsubwindowのヒストグラムを削除
				removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//中央値の計算
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				G_center++;
				result_center++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		G_center += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//メモリ開放
	for (int i = 0; i < memoryLength; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] gSum_win1;

	//return result;
}
