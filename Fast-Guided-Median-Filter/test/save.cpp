//#include "FGMF.h"
#if 0

//ベースプログラム
Mat FGMF::filter(Mat& I, Mat& G, int r, float eps2, int Imax)
{
	//check validation
	//どちらもint型、チャンネル数1
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && G.channels() == 1);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#endif

	//中央値
	const float half = 0.5f;

	//初期化
	int width_ = I.cols;
	int height_ = I.rows;

	//結果保存
	Mat med_idx = Mat(height_, width_, CV_32S);


	//各列用ヒストグラム格納変数
	fg** histo_col = new fg *[width_];
	for (int i = 0; i < width_; i++)
	{
		histo_col[i] = (fg*)_aligned_malloc(sizeof(fg) * Imax, 32);
		memset(histo_col[i], 0, sizeof(fg) * Imax);
	}


	//mean(g), mean(gg)逐次計算用変数（和で保存）
	//under_i_indexで示される値以下の列ヒストグラム和
	struct colDataStruct {
		int g_sum;
		int gg_sum;
		int under_i_f;
		int under_i_g;

	};
	colDataStruct* colData = new colDataStruct[width_];
	memset(colData, 0, sizeof(colDataStruct) * width_);

	//under_i_indexで示される値以下 のときのindex
	int* under_i_col_index = new int[width_];
	memset(under_i_col_index, 0, sizeof(int) * width_);

	//Window内ヒストグラム
	fg* histo_window = (fg*)_aligned_malloc(sizeof(fg) * Imax, 32);


	//
	int y_add = r - 1;//削除される列
	int y_rem = -r - 2;//削除される行


	//処理開始
	for (int y = 0; y < height_; y++)
	{
		//cout << "y:" << y << endl;

		y_add++;
		y_rem++;

		bool hasPreviousRow = y_rem >= 0;
		bool hasNestRow = y_add < height_;

		//初期化

		//Window関係
		//和
		int g_sum_window = 0;
		int gg_sum_window = 0;
		int pixel_sum_window = 0;
		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(fg) * Imax);

		//index以下
		int f_sum_under_i_window = 0;
		int g_sum_under_i_window = 0;



		//1列当たりの画素数
		int pixel_sum_col = 2 * r + 1;
		if (y < r)
			pixel_sum_col = y + r + 1;
		else if ((y + r) >= height_)
			pixel_sum_col = height_ - y + r;


		//処理中のindex y = 0なら入力で初期化、y > 0なら[y - 1, 0]の値で初期化
		int index;
		if (y > 0)
			index = *med_idx.ptr<int>(y - 1);
		else
			index = *I.ptr<int>(0);

		int x_add = -1;// 追加される列
		int x_rem = -2 * r - 2;// 削除される列
		//対応する画素へのポインタ
		//列の削除画素
		int* I_rem;
		int* G_rem;
		if (hasPreviousRow) {
			I_rem = I.ptr<int>(y_rem);
			G_rem = G.ptr<int>(y_rem);
		}
		//列の追加画素
		int* I_add;
		int* G_add;
		if (hasNestRow) {
			I_add = I.ptr<int>(y_add);
			G_add = G.ptr<int>(y_add);
		}
		//処理中の画素
		int* G_center = G.ptr<int>(y);
		int* med_center = med_idx.ptr<int>(y);

		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 0.0f;

		for (int x = -r; x < width_; x++)
		{
			x_add++;
			x_rem++;


			//追加する次の列があるか
			bool hasNextColumn = x_add < width_;
			//削除する前の列があるか
			bool hasPreviousColumn = x_rem >= 0;

			//window内画素数の更新
			if (hasNextColumn && !hasPreviousColumn) {
				//追加のみ
				pixel_sum_window += pixel_sum_col;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}
			else if (!hasNextColumn && hasPreviousColumn)
			{
				//削除のみ
				pixel_sum_window -= pixel_sum_col;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}

			//列ヒストグラムの更新

			//次の列があるなら更新
			if (hasNextColumn) {
				if (y > 0) {

					//前の行があるならヒストグラムから削除
					if (hasPreviousRow)
					{
						//前の行のgとf取得
						//ヒストグラムから減算
						int fidx = I_rem[x_add];//行は削除行、列は追加行なのでこうなる
						int gval = G_rem[x_add];
						histo_col[x_add][fidx].f -= 1;
						histo_col[x_add][fidx].g -= gval;

						//SIMD化可能(効果あるか分からなかった)
						//g_sum_col, gg_sum_colから減算
						colData[x_add].g_sum -= gval;
						colData[x_add].gg_sum -= gval * gval;
						if (fidx <= under_i_col_index[x_add]) {
							colData[x_add].under_i_f -= 1;
							colData[x_add].under_i_g -= gval;
						}

						/*
						__m128i a128 = _mm_load_si128((__m128i *)&colData[x_add]);
						int m1 = fidx <= under_i_col_index[x_add];
						__m128i m128 = _mm_set_epi32(gval * m1, m1, gval * gval, gval);
						__m128i c128 = _mm_sub_epi32(a128, m128);
						_mm_store_si128((__m128i *)&colData[x_add], c128);
						*/
					}
					//次の行があるなら更新
					if (hasNestRow) {
						//次の行をヒストグラムに追加
						//次の行のgとf取得
						//ヒストグラムに加算
						int fidx = I_add[x_add];
						int gval = G_add[x_add];
						histo_col[x_add][fidx].f += 1;
						histo_col[x_add][fidx].g += gval;

						//SIMD化可能(効果あるか分からなかった)
						//g_sum_col, gg_sum_colに加算
						colData[x_add].g_sum += gval;
						colData[x_add].gg_sum += gval * gval;
						//追加要素のfがこのindex以下かどうかを判定し、以下なら追加するために足す、より大きいならsumに含まれないので何もしない
						if (fidx <= under_i_col_index[x_add]) {
							colData[x_add].under_i_f += 1;
							colData[x_add].under_i_g += gval;
						}

						/*
						__m128i a128 = _mm_load_si128((__m128i *)&colData[x_add]);
						int m1 = fidx <= under_i_col_index[x_add];
						__m128i m128 = _mm_set_epi32(gval * m1, m1, gval * gval, gval);
						__m128i c128 = _mm_add_epi32(a128, m128);
						_mm_store_si128((__m128i *)&colData[x_add], c128);
						*/
					}
				}
				else
				{
					//1行目(y=0)用処理　列すべて追加
					for (int yy = 0; yy <= r; yy++)
					{
						//gとf取得
						//ヒストグラムに加算
						int fidx = I.ptr<int>(yy)[x_add];
						int gval = G.ptr<int>(yy)[x_add];
						histo_col[x_add][fidx].f += 1;
						histo_col[x_add][fidx].g += gval;
						//g_sum_col, gg_sum_colに加算
						colData[x_add].g_sum += gval;
						colData[x_add].gg_sum += gval * gval;
					}

				}
				//window更新
				//sum_windowに追加列を加算
				g_sum_window += colData[x_add].g_sum;
				gg_sum_window += colData[x_add].gg_sum;
				//ヒストグラムに追加
				addHistogram(Imax, histo_window, histo_col[x_add]);

				int u_idx = under_i_col_index[x_add];
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//多分SIMD化可能
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//追加
					colData[x_add].under_i_f += histo_col[x_add][i].f * sign1;
					colData[x_add].under_i_g += histo_col[x_add][i].g * sign1;
				}

				//under indexの更新
				under_i_col_index[x_add] = index;

				//ウィンドウ内f_sum, g_sumの更新
				f_sum_under_i_window += colData[x_add].under_i_f;
				g_sum_under_i_window += colData[x_add].under_i_g;

			}

			//削除列があるなら
			if (hasPreviousColumn)
			{
				//window更新
				//ヒストグラムから削除
				remHistogram(Imax, histo_window, histo_col[x_rem]);


				//sum_windowから削除列を減算
				g_sum_window -= colData[x_rem].g_sum;
				gg_sum_window -= colData[x_rem].gg_sum;
				//ウィンドウ内f_sum, g_sumの更新
				f_sum_under_i_window -= colData[x_rem].under_i_f;
				g_sum_under_i_window -= colData[x_rem].under_i_g;


				int u_idx = under_i_col_index[x_rem];
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//多分SIMD化可能
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//追加
					f_sum_under_i_window -= histo_col[x_rem][i].f * sign1;
					g_sum_under_i_window -= histo_col[x_rem][i].g * sign1;
				}
			}

			//中央値の計算
			if (x >= 0)
			{
				float g_ave = g_sum_window * pixel_sum_window_inv;
				float gg_ave = gg_sum_window * pixel_sum_window_inv;
				float vx = gg_ave - g_ave * g_ave + eps2;
				float tmp = G_center[x] - g_ave;
				float cx = tmp * pixel_sum_window_inv / vx;
				float dx = pixel_sum_window_inv - g_ave * cx;
				float h = cx * g_sum_under_i_window + dx * f_sum_under_i_window;

				int flag1 = h < half;
				int flag2 = flag1 - 1;
				int sign1 = flag1 * 2 - 1;
				while (true)
				{
					index += flag1;
					int fnum = histo_window[index].f;
					if (fnum != 0)
					{
						//あるなら更新してhチェック
						f_sum_under_i_window += fnum * sign1;
						g_sum_under_i_window += histo_window[index].g * sign1;
						h = cx * g_sum_under_i_window + dx * f_sum_under_i_window;
						if ((h - half) * sign1 >= 0)
						{
							//超えたのでこのindexがmedian
							med_center[x] = index;
							index += flag2;
							break;
						}
					}
					index += flag2;
				}
			}

		}

	}



	//メモリ開放
	for (int i = 0; i < width_; i++)
	{
		_aligned_free(histo_col[i]);
	}
	delete[] histo_col;
	_aligned_free(histo_window);
	delete[] colData;


	return med_idx;
}




//性能が下がらないようにしながらリファクタリング
//ギリギリまでやった場合。これ以上やると性能低下したが、僅かなのでやる
Mat FGMF::filter4(Mat& I, Mat& G, int radius, float eps2, int Imax)
{
	//check validation
	//どちらもint型、チャンネル数1
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && G.channels() == 1);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//中央値
	const float half = 0.5f;

	//初期化
	int size_dim0 = I.cols;
	int size_dim1 = I.rows;
	//
	int r_dim0 = radius;
	int r_dim1 = radius;

	//結果保存
	Mat result = Mat(size_dim1, size_dim0, CV_32S);


	//各列用ヒストグラム格納変数
	fg** histo_win1 = new fg *[size_dim0];
	for (int i = 0; i < size_dim0; i++)
	{
		histo_win1[i] = (fg*)_aligned_malloc(sizeof(fg) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(fg) * Imax);
	}


	//mean(g), mean(gg)逐次計算用変数（和で保存）
	//under_i_indexで示される値以下の列ヒストグラム和
	struct windowDataStruct {
		int g_sum;
		int gg_sum;
		int f_sum_upto_index;
		int g_sum_upto_index;
		int index;

	};
	windowDataStruct* sum_win1 = new windowDataStruct[size_dim0];
	memset(sum_win1, 0, sizeof(windowDataStruct) * size_dim0);

	//サブウィンドウのsumUpToIndex記録時のインデックス
	//int* index_win1 = new int[size_dim0];
	//memset(index_win1, 0, sizeof(int) * size_dim0);

	//Window内ヒストグラム
	fg* histo_window = (fg*)_aligned_malloc(sizeof(fg) * Imax, MEMORY_ALIGNMENT);


	//
	int x_add_dim1 = radius - 1;//削除される列
	int x_rem_dim1 = -radius - 2;//削除される行
	int x_dim1 = 0;

	//処理開始
	for (; x_dim1 < size_dim1; x_dim1++)
	{
		x_add_dim1++;
		x_rem_dim1++;

		bool hasRem_dim1 = x_rem_dim1 >= 0;
		bool hasAdd_dim1 = x_add_dim1 < size_dim1;

		//初期化

		//Window関係
		//和
		gSumData gSum_window = { 0, 0 };
		//index以下
		int sumUpToIndex_window_f = 0;
		int sumUpToIndex_window_g = 0;

		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(fg) * Imax);
		//ウィンドウ内画素数
		int pixel_sum_window = 0;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 0.0f;
		//1列当たりの画素数
		//int pixel_sum_dim1;
		//メインウィンドウ内インデックス
		int index;


		//対応する画素へのポインタ
		//列の削除画素
		int* I_rem = NULL;
		int* G_rem = NULL;
		//列の追加画素
		int* I_add = NULL;
		int* G_add = NULL;
		//処理中の画素
		int* G_center = G.ptr<int>(x_dim1);
		int* med_center = result.ptr<int>(x_dim1);



		if (hasRem_dim1) {
			I_rem = I.ptr<int>(x_rem_dim1);
			G_rem = G.ptr<int>(x_rem_dim1);
		}
		if (hasAdd_dim1) {
			I_add = I.ptr<int>(x_add_dim1);
			G_add = G.ptr<int>(x_add_dim1);
		}


		//1列当たりの画素数
		int pixel_sum_dim1 = 2 * radius + 1;
		if (x_dim1 < radius)
			pixel_sum_dim1 = x_dim1 + radius + 1;
		else if ((x_dim1 + radius) >= size_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + radius;


		if (x_dim1 > 0)
			index = *result.ptr<int>(x_dim1 - 1);
		else
			index = *I.ptr<int>(0);



		int x_add_dim0 = -1;// 追加される列
		int x_rem_dim0 = -2 * radius - 2;// 削除される列
		int x_dim0 = -radius;

		for (; x_dim0 < size_dim0; x_dim0++)
		{
			x_add_dim0++;
			x_rem_dim0++;


			//追加する次の列があるか
			bool hasAdd_dim0 = x_add_dim0 < size_dim0;
			//削除する前の列があるか
			bool hasRem_dim0 = x_rem_dim0 >= 0;

			//window内画素数の更新
			if (hasAdd_dim0 && !hasRem_dim0) {
				//追加のみ
				pixel_sum_window += pixel_sum_dim1;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}
			else if (!hasAdd_dim0 && hasRem_dim0)
			{
				//削除のみ
				pixel_sum_window -= pixel_sum_dim1;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}

			//列ヒストグラムの更新

			//次の列があるなら更新
			if (hasAdd_dim0) {
				if (x_dim1 > 0) {

					//前の行があるならヒストグラムから削除
					if (hasRem_dim1)
					{
						//前の行のgとf取得
						//ヒストグラムから減算
						int fidx = I_rem[x_add_dim0];//行は削除行、列は追加行なのでこうなる
						int gval = G_rem[x_add_dim0];
						histo_win1[x_add_dim0][fidx].f -= 1;
						histo_win1[x_add_dim0][fidx].g -= gval;

						//SIMD化可能(効果あるか分からなかった)
						//g_sum_col, gg_sum_colから減算
						sum_win1[x_add_dim0].g_sum -= gval;
						sum_win1[x_add_dim0].gg_sum -= gval * gval;
						if (fidx <= sum_win1[x_add_dim0].index) {
							sum_win1[x_add_dim0].f_sum_upto_index -= 1;
							sum_win1[x_add_dim0].g_sum_upto_index -= gval;
						}
					}
					//次の行があるなら更新
					if (hasAdd_dim1) {
						//次の行をヒストグラムに追加
						//次の行のgとf取得
						//ヒストグラムに加算
						int fidx = I_add[x_add_dim0];
						int gval = G_add[x_add_dim0];
						histo_win1[x_add_dim0][fidx].f += 1;
						histo_win1[x_add_dim0][fidx].g += gval;

						//SIMD化可能(効果あるか分からなかった)
						//g_sum_col, gg_sum_colに加算
						sum_win1[x_add_dim0].g_sum += gval;
						sum_win1[x_add_dim0].gg_sum += gval * gval;
						//追加要素のfがこのindex以下かどうかを判定し、以下なら追加するために足す、より大きいならsumに含まれないので何もしない
						if (fidx <= sum_win1[x_add_dim0].index) {
							sum_win1[x_add_dim0].f_sum_upto_index += 1;
							sum_win1[x_add_dim0].g_sum_upto_index += gval;
						}
					}
				}
				else
				{
					//1行目(y=0)用処理　列すべて追加
					for (int yy = 0; yy <= radius; yy++)
					{
						//gとf取得
						//ヒストグラムに加算
						int fidx = I.ptr<int>(yy)[x_add_dim0];
						int gval = G.ptr<int>(yy)[x_add_dim0];
						histo_win1[x_add_dim0][fidx].f += 1;
						histo_win1[x_add_dim0][fidx].g += gval;
						//g_sum_col, gg_sum_colに加算
						sum_win1[x_add_dim0].g_sum += gval;
						sum_win1[x_add_dim0].gg_sum += gval * gval;
					}

				}
				//window更新
				//sum_windowに追加列を加算
				gSum_window.g_sum += sum_win1[x_add_dim0].g_sum;
				gSum_window.gg_sum += sum_win1[x_add_dim0].gg_sum;
				//ヒストグラムに追加
				addHistogram(Imax, histo_window, histo_win1[x_add_dim0]);

				int u_idx = sum_win1[x_add_dim0].index;
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//多分SIMD化可能
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//追加
					sum_win1[x_add_dim0].f_sum_upto_index += histo_win1[x_add_dim0][i].f * sign1;
					sum_win1[x_add_dim0].g_sum_upto_index += histo_win1[x_add_dim0][i].g * sign1;
				}

				//under indexの更新
				sum_win1[x_add_dim0].index = index;

				//ウィンドウ内f_sum, g_sumの更新
				sumUpToIndex_window_f += sum_win1[x_add_dim0].f_sum_upto_index;
				sumUpToIndex_window_g += sum_win1[x_add_dim0].g_sum_upto_index;

			}

			//削除列があるなら
			if (hasRem_dim0)
			{
				//window更新
				//ヒストグラムから削除
				remHistogram(Imax, histo_window, histo_win1[x_rem_dim0]);


				//sum_windowから削除列を減算
				gSum_window.g_sum -= sum_win1[x_rem_dim0].g_sum;
				gSum_window.gg_sum -= sum_win1[x_rem_dim0].gg_sum;
				//ウィンドウ内f_sum, g_sumの更新
				sumUpToIndex_window_f -= sum_win1[x_rem_dim0].f_sum_upto_index;
				sumUpToIndex_window_g -= sum_win1[x_rem_dim0].g_sum_upto_index;


				int u_idx = sum_win1[x_rem_dim0].index;
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//多分SIMD化可能
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//追加
					sumUpToIndex_window_f -= histo_win1[x_rem_dim0][i].f * sign1;
					sumUpToIndex_window_g -= histo_win1[x_rem_dim0][i].g * sign1;
				}
			}

			//中央値の計算
			if (x_dim0 >= 0)
			{
				float g_ave = gSum_window.g_sum * pixel_sum_window_inv;
				float gg_ave = gSum_window.gg_sum * pixel_sum_window_inv;
				float vx = gg_ave - g_ave * g_ave + eps2;
				float tmp = G_center[x_dim0] - g_ave;
				float cx = tmp * pixel_sum_window_inv / vx;
				float dx = pixel_sum_window_inv - g_ave * cx;
				float h = cx * sumUpToIndex_window_g + dx * sumUpToIndex_window_f;

				int flag1 = h < half;
				int flag2 = flag1 - 1;
				int sign1 = flag1 * 2 - 1;
				while (true)
				{
					index += flag1;
					int fnum = histo_window[index].f;
					if (fnum != 0)
					{
						//あるなら更新してhチェック
						sumUpToIndex_window_f += fnum * sign1;
						sumUpToIndex_window_g += histo_window[index].g * sign1;
						h = cx * sumUpToIndex_window_g + dx * sumUpToIndex_window_f;
						if ((h - half) * sign1 >= 0)
						{
							//超えたのでこのindexがmedian
							med_center[x_dim0] = index;
							index += flag2;
							break;
						}
					}
					index += flag2;
				}
			}

		}

	}



	//メモリ開放
	for (int i = 0; i < size_dim0; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] sum_win1;


	return result;
}




//古い
Mat FGMF::filterOld(Mat& I, Mat& G, int r, float eps2, int Imax)
{
	//check validation
	//どちらもint型、チャンネル数1
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && G.channels() == 1);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#endif

	//中央値
	const float half = 0.5f;

	//初期化
	int width_ = I.cols;
	int height_ = I.rows;

	//結果保存
	Mat med_idx = Mat(height_, width_, CV_32S);


	//各列用ヒストグラム格納変数
	fg** histo_col = new fg *[width_];
	for (int i = 0; i < width_; i++)
	{
		histo_col[i] = (fg*)_aligned_malloc(sizeof(fg) * Imax, 32);
		memset(histo_col[i], 0, sizeof(fg) * Imax);
	}


	//mean(g), mean(gg)逐次計算用変数（和で保存）
	//under_i_indexで示される値以下の列ヒストグラム和
	struct colDataStruct {
		int g_sum;
		int gg_sum;
		int under_i_f;
		int under_i_g;

	};
	colDataStruct* colData = new colDataStruct[width_];
	memset(colData, 0, sizeof(colDataStruct) * width_);

	//under_i_indexで示される値以下 のときのindex
	int* under_i_col_index = new int[width_];
	memset(under_i_col_index, 0, sizeof(int) * width_);

	//Window内ヒストグラム
	fg* histo_window = (fg*)_aligned_malloc(sizeof(fg) * Imax, 32);


	//
	int y_add = r - 1;//削除される列
	int y_rem = -r - 2;//削除される行


	//処理開始
	for (int y = 0; y < height_; y++)
	{
		//cout << "y:" << y << endl;

		y_add++;
		y_rem++;

		bool hasPreviousRow = y_rem >= 0;
		bool hasNestRow = y_add < height_;

		//初期化

		//Window関係
		//和
		int g_sum_window = 0;
		int gg_sum_window = 0;
		int pixel_sum_window = 0;
		//ウィンドウヒストグラム初期化
		memset(histo_window, 0, sizeof(fg) * Imax);

		//index以下
		int f_sum_under_i_window = 0;
		int g_sum_under_i_window = 0;



		//1列当たりの画素数
		int pixel_sum_col = 2 * r + 1;
		if (y < r)
			pixel_sum_col = y + r + 1;
		else if ((y + r) >= height_)
			pixel_sum_col = height_ - y + r;


		//処理中のindex y = 0なら入力で初期化、y > 0なら[y - 1, 0]の値で初期化
		int index;
		if (y > 0)
			index = *med_idx.ptr<int>(y - 1);
		else
			index = *I.ptr<int>(0);

		int x_add = -1;// 追加される列
		int x_rem = -2 * r - 2;// 削除される列
		//対応する画素へのポインタ
		//列の削除画素
		int* I_rem = NULL;
		int* G_rem = NULL;
		if (hasPreviousRow) {
			I_rem = I.ptr<int>(y_rem);
			G_rem = G.ptr<int>(y_rem);
		}
		//列の追加画素
		int* I_add = NULL;
		int* G_add = NULL;
		if (hasNestRow) {
			I_add = I.ptr<int>(y_add);
			G_add = G.ptr<int>(y_add);
		}
		//処理中の画素
		int* G_center = G.ptr<int>(y);
		int* med_center = med_idx.ptr<int>(y);

		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 0.0f;

		for (int x = -r; x < width_; x++)
		{
			x_add++;
			x_rem++;


			//追加する次の列があるか
			bool hasNextColumn = x_add < width_;
			//削除する前の列があるか
			bool hasPreviousColumn = x_rem >= 0;

			//window内画素数の更新
			if (hasNextColumn && !hasPreviousColumn) {
				//追加のみ
				pixel_sum_window += pixel_sum_col;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}
			else if (!hasNextColumn && hasPreviousColumn)
			{
				//削除のみ
				pixel_sum_window -= pixel_sum_col;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}

			//列ヒストグラムの更新

			//次の列があるなら更新
			if (hasNextColumn) {
				if (y > 0) {

					//前の行があるならヒストグラムから削除
					if (hasPreviousRow)
					{
						//前の行のgとf取得
						//ヒストグラムから減算
						int fidx = I_rem[x_add];//行は削除行、列は追加行なのでこうなる
						int gval = G_rem[x_add];
						histo_col[x_add][fidx].f -= 1;
						histo_col[x_add][fidx].g -= gval;

						//SIMD化可能(効果あるか分からなかった)
						//g_sum_col, gg_sum_colから減算
						colData[x_add].g_sum -= gval;
						colData[x_add].gg_sum -= gval * gval;
						if (fidx <= under_i_col_index[x_add]) {
							colData[x_add].under_i_f -= 1;
							colData[x_add].under_i_g -= gval;
						}

						/*
						__m128i a128 = _mm_load_si128((__m128i *)&colData[x_add]);
						int m1 = fidx <= under_i_col_index[x_add];
						__m128i m128 = _mm_set_epi32(gval * m1, m1, gval * gval, gval);
						__m128i c128 = _mm_sub_epi32(a128, m128);
						_mm_store_si128((__m128i *)&colData[x_add], c128);
						*/
					}
					//次の行があるなら更新
					if (hasNestRow) {
						//次の行をヒストグラムに追加
						//次の行のgとf取得
						//ヒストグラムに加算
						int fidx = I_add[x_add];
						int gval = G_add[x_add];
						histo_col[x_add][fidx].f += 1;
						histo_col[x_add][fidx].g += gval;

						//SIMD化可能(効果あるか分からなかった)
						//g_sum_col, gg_sum_colに加算
						colData[x_add].g_sum += gval;
						colData[x_add].gg_sum += gval * gval;
						//追加要素のfがこのindex以下かどうかを判定し、以下なら追加するために足す、より大きいならsumに含まれないので何もしない
						if (fidx <= under_i_col_index[x_add]) {
							colData[x_add].under_i_f += 1;
							colData[x_add].under_i_g += gval;
						}

						/*
						__m128i a128 = _mm_load_si128((__m128i *)&colData[x_add]);
						int m1 = fidx <= under_i_col_index[x_add];
						__m128i m128 = _mm_set_epi32(gval * m1, m1, gval * gval, gval);
						__m128i c128 = _mm_add_epi32(a128, m128);
						_mm_store_si128((__m128i *)&colData[x_add], c128);
						*/
					}
				}
				else
				{
					//1行目(y=0)用処理　列すべて追加
					for (int yy = 0; yy <= r; yy++)
					{
						//gとf取得
						//ヒストグラムに加算
						int fidx = I.ptr<int>(yy)[x_add];
						int gval = G.ptr<int>(yy)[x_add];
						histo_col[x_add][fidx].f += 1;
						histo_col[x_add][fidx].g += gval;
						//g_sum_col, gg_sum_colに加算
						colData[x_add].g_sum += gval;
						colData[x_add].gg_sum += gval * gval;
					}

				}
				//window更新
				//sum_windowに追加列を加算
				g_sum_window += colData[x_add].g_sum;
				gg_sum_window += colData[x_add].gg_sum;
				//ヒストグラムに追加
				addHistogram(Imax, histo_window, histo_col[x_add]);

				int u_idx = under_i_col_index[x_add];
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//多分SIMD化可能
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//追加
					colData[x_add].under_i_f += histo_col[x_add][i].f * sign1;
					colData[x_add].under_i_g += histo_col[x_add][i].g * sign1;
				}

				//under indexの更新
				under_i_col_index[x_add] = index;

				//ウィンドウ内f_sum, g_sumの更新
				f_sum_under_i_window += colData[x_add].under_i_f;
				g_sum_under_i_window += colData[x_add].under_i_g;

			}

			//削除列があるなら
			if (hasPreviousColumn)
			{
				//window更新
				//ヒストグラムから削除
				remHistogram(Imax, histo_window, histo_col[x_rem]);


				//sum_windowから削除列を減算
				g_sum_window -= colData[x_rem].g_sum;
				gg_sum_window -= colData[x_rem].gg_sum;
				//ウィンドウ内f_sum, g_sumの更新
				f_sum_under_i_window -= colData[x_rem].under_i_f;
				g_sum_under_i_window -= colData[x_rem].under_i_g;


				int u_idx = under_i_col_index[x_rem];
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//多分SIMD化可能
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//追加
					f_sum_under_i_window -= histo_col[x_rem][i].f * sign1;
					g_sum_under_i_window -= histo_col[x_rem][i].g * sign1;
				}
			}

			//中央値の計算
			if (x >= 0)
			{
				float g_ave = g_sum_window * pixel_sum_window_inv;
				float gg_ave = gg_sum_window * pixel_sum_window_inv;
				float vx = gg_ave - g_ave * g_ave + eps2;
				float tmp = G_center[x] - g_ave;
				float cx = tmp * pixel_sum_window_inv / vx;
				float dx = pixel_sum_window_inv - g_ave * cx;
				float h = cx * g_sum_under_i_window + dx * f_sum_under_i_window;

				int flag1 = h < half;
				int flag2 = flag1 - 1;
				int sign1 = flag1 * 2 - 1;
				while (true)
				{
					index += flag1;
					//if (histo_window[index].f)
					{
						//あるなら更新してhチェック
						f_sum_under_i_window += histo_window[index].f * sign1;
						g_sum_under_i_window += histo_window[index].g * sign1;
						h = cx * g_sum_under_i_window + dx * f_sum_under_i_window;
						if ((h >= half) == flag1)
						{
							//超えたのでこのindexがmedian
							med_center[x] = index;
							index += flag2;
							break;
						}
					}
					index += flag2;
				}
			}

		}

	}



	//メモリ開放
	for (int i = 0; i < width_; i++)
	{
		_aligned_free(histo_col[i]);
	}
	delete[] histo_col;
	_aligned_free(histo_window);
	delete[] colData;


	return med_idx;
}



//2D　入力1チャンネル 多次元化用
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
Mat FGMF::filter2DWindow(Mat& I, Mat& G, int radius, float eps2, int Imax)
{
	//check validation
	//どちらもint型、チャンネル数1
	assert(I.depth() == CV_32S && I.channels() == 1);
	//assert(G.depth() == CV_32S && G.channels() == 1);
	//assert(I.isContinuous() && G.isContinuous());
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
	const vector<int> r_dim{ radius,radius };

	//ウィンドウアクセス用
	//size_dim = {dim0, dim1, dim3, dim4}のとき
	// = {dim0, dim0*dim1, dim0*dim1*dim3}
	//とすることで、例えば [x1,x2,x3,x4]にアクセスするとき、ウィンドウメモリ上ではこれが1次元で並んでいるので
	// x4*dim0*dim1*dim3 + x3*dim0*dim1 + x2*dim0 + x1
	const vector<int> size_prod{ size_dim[0] };

	//メモリ確保・初期化
	//結果保存
	Mat result = Mat(I.rows, I.cols, CV_32S);


	//
	// (gSum, sumUpToIndex, histo)
	//

	//W0:画素なので記録の必要なし（I,G）

	//W1
	Window_vector<GSum, FGSumUpToIndex, FG> W1 = Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 1, true);

	//W2(Wmain)
	Window_single<GSum, FGSumUpToIndex, FG> Wmain = Window_single<GSum, FGSumUpToIndex, FG>(Imax);


	//画素数
	vector<int> pixel_sum(DIM);
	//位置
	vector<Pos> x(DIM);
	//ステータス
	vector<DimStatus> status(DIM);

	//対応する画素へのポインタ
	//処理中の画素
	int* result_center = result.ptr<int>(0);
	GTYPE* G_center = G.ptr<GTYPE>(0);
	int* W0_rem_f = I.ptr<int>(0);
	GTYPE* W0_rem_g = G.ptr<GTYPE>(0);
	int* W0_add_f = I.ptr<int>(0);
	GTYPE* W0_add_g = G.ptr<GTYPE>(0);


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
		Wmain.initialize();

		//ウィンドウ内画素数
		int pixel_sum_window = 0;
		//ウィンドウ内画素数の逆数
		float pixel_sum_window_inv = 0.0f;



		/*
		各階層のウィンドウに対するポインタを持つと考えれば、画素だけでなくwin系も同じように書けるのでは

		dim0方向にスライド (W0は画素)
		W1[x[1].add, x[0].add] = W1[x[1].add, x[0].add] + W0[x[2].add, x[1].add, x[0].add] - W0[x[2].rem, x[1].add, x[0].add]
		W2[x[0].add] = W2[x[0].add] + W1[x[1].add, x[0].add] - W1[x[1].rem, x[0].add]
		W3 = W3 + W2[x[0].add] - W2[x[0].rem] (W3はwindow)
		*/

		//W2


		//次の階層の処理位置セット
		setPos(x[0], r_dim[0]);
		for (; x[0].center < size_dim[0]; x[0].center++, x[0].add++, x[0].rem++)
		{
			//ステータスセット
			setStatus(x[0], size_dim[0], status[1].isInside_image, status[0]);


			//window内画素数の更新
			calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

			//W2の更新
			if (status[0].hasAdd)
			{
				//W1[x[0].add] の更新
				if (status[1].hasAdd)//画素追加	(W1[x[0].add]) + W0[x[1].add, x[0].add]
				{
					addPixelToWindow_gSum(W1, x[0].add, *W0_add_f, *W0_add_g);
					W0_add_f++;
					W0_add_g++;
				}
				if (status[1].hasRem)//画素削除	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
				{
					removePixelFromWindow_gSum(W1, x[0].add, *W0_rem_f, *W0_rem_g);
					W0_rem_f++;
					W0_rem_g++;
				}
				//W1追加	(W2) + W1[x[0].add]
				updateSubWindowAndAddToWindow_gSum(Imax, Wmain, W1, x[0].add);
			}
			if (status[0].hasRem)//W1削除	(W2) - W1[x[0].rem]
				removeSubWindowFromWindow_gSum(Imax, Wmain, W1, x[0].rem);


			//中央値の計算
			if (status[0].isInside_image)
			{
				CTYPE cx;
				float dx;
				calculateCxDx(Wmain.gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
				//findMedian(cx, dx, half, Wmain.sumUpToIndex.index, *result_center, Wmain);
				findMedian(cx, dx, half, Wmain, *result_center);
				G_center++;
				result_center++;
			}
		}
	}


	return result;
}



#endif