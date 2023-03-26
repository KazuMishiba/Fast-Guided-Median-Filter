#pragma once

#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include<opencv2/opencv.hpp>
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <iterator>
#include <memory>
#include <thread>
#include <list>
#include "common.h"



struct fg_node {//for doubly linked list
	int f;
	int g;
	int index;
	fg_node() :f(0), g(0), index(0) {};
	fg_node(int _f, int _g, int _index) :f(_f), g(_g), index(_index) {};
	fg_node copy() { return fg_node(f, g, index); };
	void add(fg_node& node) {
		f += node.f;
		g += node.g;
	}
	void sub(fg_node& node) {
		f -= node.f;
		g -= node.g;
	}
};
struct fg3_node {//for doubly linked list
	int f;
	cv::Vec3i g;
	int index;
	fg3_node() :f(0), index(0) { g = cv::Vec3i(0, 0, 0); };
	fg3_node(int _f, cv::Vec3i _g, int _index) :f(_f), index(_index) { g[0] = _g[0]; g[1] = _g[1];	g[2] = _g[2];};
	fg3_node copy() { return fg3_node(f, g, index); };
	void add(fg3_node& node) {
		f += node.f;
		g += node.g;
	}
	void sub(fg3_node& node) {
		f -= node.f;
		g -= node.g;
	}
};



struct fg {
	int f;
	int g;
};
struct fg3 {
	int f;
	int g1;
	int g2;
	int g3;
};
template<int ChannelNum>
struct fgx {
	int f;
	cv::Vec<int, ChannelNum> g;
};
//
struct gSum {
	int g;
	int gg;
};
struct g3Sum {
	int g1;
	int g2;
	int g3;
	int g1g1;
	int g1g2;
	int g1g3;
	int g2g2;
	int g2g3;
	int g3g3;
};
template<int ChannelNum>
struct gxSum {
	cv::Vec<int, ChannelNum> g;
	cv::Vec<int, ((ChannelNum + 1)*ChannelNum) / 2> gg;
};
//
struct fgSumUpToIndex {
	int f;
	int g;
	int index;
};
struct fg3SumUpToIndex {
	int f;
	int g1;
	int g2;
	int g3;
	int index;
};
template<int ChannelNum>
struct fgxSumUpToIndex {
	int f;
	cv::Vec<int, ChannelNum> g;
	int index;
};


inline void adjustImax(int& Imax) {
#if defined(USE_AVX2)
	//AVX/AVX2を使う場合、Imaxは4の倍数にする
	if (Imax % 4 != 0)
		Imax += (4 - (Imax % 4));
#elif defined(USE_AVX512)
	if (Imax % 8 != 0)
		Imax += (8 - (Imax % 8));
#else
#endif
}


void inline calculatePosForMultithread2D(const int& width, const int& colNum, const int& r_dim0, std::vector<int>& dim0Start, std::vector<int>& dim0End, std::vector<int>& memoryLength, std::vector<int>& insideImageStart) {
	//幅の決定
	//col
	int baseWidth = width / colNum;
	int remainder = width % colNum;
	//メジアン計算対象画素開始位置（isInside_image計算用）
	insideImageStart[0] = 0;
	//処理開始位置
	dim0Start[0] = -r_dim0;
	//処理終了位置
	dim0End[0] = insideImageStart[0] + baseWidth + (remainder > 0) - 1;
	remainder--;
	//メモリdim0方向長さ
	memoryLength[0] = dim0End[0] - insideImageStart[0] + 1 + r_dim0;
	//
	for (int i = 1; i < colNum; i++)
	{
		insideImageStart[i] = dim0End[i - 1] + 1;
		dim0Start[i] = insideImageStart[i] - 2 * r_dim0;
		dim0End[i] = insideImageStart[i] + baseWidth + (remainder > 0) - 1;
		remainder--;
		//
		memoryLength[i] = dim0End[i] - insideImageStart[i] + 1 + r_dim0 * 2;
	}
	memoryLength[colNum - 1] -= r_dim0;
}


void inline calculatePosForCols(cv::Size size, const int& colNum, std::vector<int>& dim0Start, std::vector<int>& dim0End) {
	//col
	int baseWidth = size.width / colNum;
	int remainder = size.width % colNum;
	dim0Start[0] = 0;
	dim0End[0] = dim0Start[0] + baseWidth + (remainder > 0) - 1;
	remainder--;
	for (int i = 1; i < colNum; i++)
	{
		dim0Start[i] = dim0End[i - 1] + 1;
		dim0End[i] = dim0Start[i] + baseWidth + (remainder > 0) - 1;
		remainder--;
	}
}
void inline calculatePosForRows(cv::Size size, const int& rowNum, std::vector<int>& dim1Start, std::vector<int>& dim1End) {
	//row
	int baseHeight = size.height / rowNum;
	int remainder = size.height % rowNum;
	dim1Start[0] = 0;
	dim1End[0] = dim1Start[0] + baseHeight + (remainder > 0) - 1;
	remainder--;
	for (int i = 1; i < rowNum; i++)
	{
		dim1Start[i] = dim1End[i - 1] + 1;
		dim1End[i] = dim1Start[i] + baseHeight + (remainder > 0) - 1;
		remainder--;
	}
}




//window内画素数の更新（追加）
inline void addPixelSum(const int& addedPixelNum, int& pixel_sum_window, float& pixel_sum_window_inv) {
	pixel_sum_window += addedPixelNum;
	pixel_sum_window_inv = 1.0f / pixel_sum_window;
}

//window内画素数の更新（削除）
inline void subtractPixelSum(const int& subtractedPixelNum, int& pixel_sum_window, float& pixel_sum_window_inv) {
	pixel_sum_window -= subtractedPixelNum;
	pixel_sum_window_inv = 1.0f / pixel_sum_window;
}
//window内画素数の更新
inline void updatePixelSum(const int& hasAdd, const int& hasRem, const int& pixel_sum_dim, int& pixel_sum_window, float& pixel_sum_window_inv) {
	if (hasAdd && !hasRem)
		addPixelSum(pixel_sum_dim, pixel_sum_window, pixel_sum_window_inv);//追加のみ
	else if (!hasAdd && hasRem)
		subtractPixelSum(pixel_sum_dim, pixel_sum_window, pixel_sum_window_inv);//削除のみ

}
//1列あたりの画素数計算
inline int calculatePixelSum_Column(const int& hasAdd, const int& hasRem, const int& x, const int& r, const int& size) {
	if (!hasRem && hasAdd)
		return x + r + 1;
	else if (hasRem && !hasAdd)
		return size - x + r;
	else
		return 2 * r + 1;
}




///////////////////////////
//入力1チャンネル・ガイド1チャンネル
inline void addHistogram(const int& Imax, fg* histo_window, fg* histo_subwin) {
#if defined(USE_AVX2)
	//サイズ(byte) Imax * sizeof(fg) 繰り返し回数 Imax * sizeof(fg) / 256
	// i+= 256 / sizeof(fg) = 4
	for (int i = 0; i < Imax; i += 4)
	{
		__m256i h256sub = _mm256_load_si256((__m256i*) & histo_subwin[i]);
		__m256i h256win = _mm256_load_si256((__m256i*) & histo_window[i]);
		h256win = _mm256_add_epi32(h256win, h256sub);
		_mm256_store_si256((__m256i*) & histo_window[i], h256win);
	}
#else
	for (int i = 0; i < Imax; i++)
	{
		histo_window[i].f += histo_subwin[i].f;
		histo_window[i].g += histo_subwin[i].g;
	}
#endif
}

inline void remHistogram(const int& Imax, fg* histo_window, fg* histo_subwin) {
#if defined(USE_AVX2)
	for (int i = 0; i < Imax; i += 4)
	{
		__m256i h256sub = _mm256_load_si256((__m256i*) & histo_subwin[i]);
		__m256i h256win = _mm256_load_si256((__m256i*) & histo_window[i]);
		h256win = _mm256_sub_epi32(h256win, h256sub);
		_mm256_store_si256((__m256i*) & histo_window[i], h256win);
	}
#else
	for (int i = 0; i < Imax; i++)
	{
		histo_window[i].f -= histo_subwin[i].f;
		histo_window[i].g -= histo_subwin[i].g;
	}
#endif
}
//fg3
inline void addHistogram(const int& Imax, fg3* histo_window, fg3* histo_subwin) {
#if defined(USE_AVX2)
	//サイズ(byte) Imax * sizeof(fg) 繰り返し回数 Imax * sizeof(fg) / 256
	// i+= 256 / sizeof(fg) = 4
	for (int i = 0; i < Imax; i += 2)
	{
		__m256i h256sub = _mm256_load_si256((__m256i*) & histo_subwin[i]);
		__m256i h256win = _mm256_load_si256((__m256i*) & histo_window[i]);
		h256win = _mm256_add_epi32(h256win, h256sub);
		_mm256_store_si256((__m256i*) & histo_window[i], h256win);
	}
#else
	for (int i = 0; i < Imax; i++)
	{
		histo_window[i].f += histo_subwin[i].f;
		histo_window[i].g1 += histo_subwin[i].g1;
		histo_window[i].g2 += histo_subwin[i].g2;
		histo_window[i].g3 += histo_subwin[i].g3;
	}
#endif
}
inline void remHistogram(const int& Imax, fg3* histo_window, fg3* histo_subwin) {
#if defined(USE_AVX2)
	for (int i = 0; i < Imax; i += 2)
	{
		__m256i h256sub = _mm256_load_si256((__m256i*) & histo_subwin[i]);
		__m256i h256win = _mm256_load_si256((__m256i*) & histo_window[i]);
		h256win = _mm256_sub_epi32(h256win, h256sub);
		_mm256_store_si256((__m256i*) & histo_window[i], h256win);
	}
#else
	for (int i = 0; i < Imax; i++)
	{
		histo_window[i].f -= histo_subwin[i].f;
		histo_window[i].g1 -= histo_subwin[i].g1;
		histo_window[i].g2 -= histo_subwin[i].g2;
		histo_window[i].g3 -= histo_subwin[i].g3;
	}
#endif
}
//multichannel
template<int n>
inline void addHistogram(const int& Imax, fgx<n>* histo_window, fgx<n>* histo_subwin) {
#if defined(USE_AVX2)
	//サイズ(byte) Imax * sizeof(fg) 繰り返し回数 Imax * sizeof(fg) / 256
	// i+= 256 / sizeof(fg) = 4
	//mm256ではint型(32bit)は1度に4つ処理できる
	//処理すべきint型の数はImax*(1+n)個
	//ということは繰り返し回数は Imax*(1*n)/4　回
	//Imaxは4で割れることが前提
	//iのインクリメントを決める方式は、nが可変なので無理
	//よってiは1ずつ増やすとして、
	int* window = (int*)histo_window;
	int* subwin = (int*)histo_subwin;
	for (int i = 0; i < Imax; i++)
	{
		__m256i h256sub = _mm256_load_si256((__m256i*) & histo_subwin[i]);
		__m256i h256win = _mm256_load_si256((__m256i*) & histo_window[i]);
		h256win = _mm256_add_epi32(h256win, h256sub);
		_mm256_store_si256((__m256i*) & histo_window[i], h256win);
	}
#else
	for (int i = 0; i < Imax; i++)
	{
		histo_window[i].f += histo_subwin[i].f;
		for (int j = 0; j < n; j++)
		{
			histo_window[i].g[j] += histo_subwin[i].g[j];
		}
	}
#endif
}
template<int n>
inline void remHistogram(const int& Imax, fgx<n>* histo_window, fgx<n>* histo_subwin) {
#if defined(USE_AVX2)
	for (int i = 0; i < Imax; i += 2)
	{
		__m256i h256sub = _mm256_load_si256((__m256i*) & histo_subwin[i]);
		__m256i h256win = _mm256_load_si256((__m256i*) & histo_window[i]);
		h256win = _mm256_sub_epi32(h256win, h256sub);
		_mm256_store_si256((__m256i*) & histo_window[i], h256win);
	}
#else
	for (int i = 0; i < Imax; i++)
	{
		histo_window[i].f += histo_subwin[i].f;
		for (int j = 0; j < n; j++)
		{
			histo_window[i].g[j] -= histo_subwin[i].g[j];
		}
	}
#endif
}

//list
//fg_list
template<typename FG_NODE>
inline void addHistogram(const int& Imax, std::list<FG_NODE>& histo_window, std::list<FG_NODE>& histo_subwin) {
	//subwinのヒストグラムをwindowのヒストグラムにマージ
	auto itr_window = histo_window.begin();
	auto itr_subwin = histo_subwin.begin();
	auto end_window = histo_window.end();
	auto end_subwin = histo_subwin.end();

	while (itr_subwin != end_subwin) {
		//windowの要素を全てたどり終えていたら終了する
		if (itr_window == end_window)
		{
			//subwinの要素が残っているならすべて追加
			while (itr_subwin != end_subwin) {
				histo_window.push_back((*itr_subwin).copy());
				itr_subwin++;
			}
			return;
		}

		//subwinから1つ要素を取り出し、itr_windowのindexと比較
		if ((*itr_subwin).index < (*itr_window).index)
		{
			//手前に挿入
			histo_window.insert(itr_window, (*itr_subwin).copy());
			itr_subwin++;
		}
		else if ((*itr_subwin).index == (*itr_window).index)
		{
			//同じなら統合
			(*itr_window).add((*itr_subwin));
			itr_subwin++;
		}
		else
		{
			//そうでないならwindowの次の要素に移る
			itr_window++;
		}
	}
}

template<typename FG_NODE>
inline void remHistogram(const int& Imax, std::list<FG_NODE>& histo_window, std::list<FG_NODE>& histo_subwin) {
	//windowのヒストグラムからsubwinのヒストグラムを削除
	auto itr_window = histo_window.begin();
	auto itr_subwin = histo_subwin.begin();
	auto end_window = histo_window.end();
	auto end_subwin = histo_subwin.end();

	while (itr_subwin != end_subwin) {
		//subwinから1つ要素を取り出し、itr_windowのindexと比較
		if ((*itr_subwin).index == (*itr_window).index)
		{
			//同じなら引く、引いた結果0になるようなら削除
			if ((*itr_window).f == (*itr_subwin).f)
			{
				itr_window = histo_window.erase(itr_window);
			}
			else
			{
				(*itr_window).sub(*itr_subwin);
				itr_window++;
			}
			itr_subwin++;
		}
		else
		{
			itr_window++;
		}
	}
}

/*
* 一番低い次元のウィンドウ更新は要素１の更新なので、ヒストグラムなどの情報更新時に、
* 0~Imaxの探索をしたりしなくて直接計算できる（直接計算したほうがコストが低い）
*/
///////////////////////////
//addPixel
//fg
inline void addPixelToWindow(fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel) {
	histo_window[index_pixel].f += 1;
	histo_window[index_pixel].g += g_pixel;
	//追加要素のfがこのindex以下かどうかを判定し、以下なら追加するために足す、より大きいならsumに含まれないので何もしない
	if (index_pixel <= sumUpToIndex_window.index) {
		sumUpToIndex_window.f += 1;
		sumUpToIndex_window.g += g_pixel;
	}
}
inline void addPixelToWindow_gSum(fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	addPixelToWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colに加算
	gSum_window.g += g_pixel;
	gSum_window.gg += g_pixel * g_pixel;
}
//sumuptoindex無しテスト
inline void addPixelToWindow(fg* histo_window, const int& index_pixel, const int& g_pixel) {
	histo_window[index_pixel].f += 1;
	histo_window[index_pixel].g += g_pixel;
}
inline void addPixelToWindow_gSum(fg* histo_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	addPixelToWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colに加算
	gSum_window.g += g_pixel;
	gSum_window.gg += g_pixel * g_pixel;
}
//fg3
inline void addPixelToWindow(fg3* histo_window, fg3SumUpToIndex& sumUpToIndex_window, const int& index_pixel, const cv::Vec3i& g_pixel) {
	histo_window[index_pixel].f += 1;
	histo_window[index_pixel].g1 += g_pixel[0];
	histo_window[index_pixel].g2 += g_pixel[1];
	histo_window[index_pixel].g3 += g_pixel[2];
	if (index_pixel <= sumUpToIndex_window.index) {
		sumUpToIndex_window.f += 1;
		sumUpToIndex_window.g1 += g_pixel[0];
		sumUpToIndex_window.g2 += g_pixel[1];
		sumUpToIndex_window.g3 += g_pixel[2];
	}
}
inline void addPixelToWindow_gSum(fg3* histo_window, fg3SumUpToIndex& sumUpToIndex_window, const int& index_pixel, const cv::Vec3i& g_pixel, g3Sum& gSum_window) {
	addPixelToWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	gSum_window.g1 += g_pixel[0];
	gSum_window.g2 += g_pixel[1];
	gSum_window.g3 += g_pixel[2];
	gSum_window.g1g1 += g_pixel[0] * g_pixel[0];
	gSum_window.g1g2 += g_pixel[0] * g_pixel[1];
	gSum_window.g1g3 += g_pixel[0] * g_pixel[2];
	gSum_window.g2g2 += g_pixel[1] * g_pixel[1];
	gSum_window.g2g3 += g_pixel[1] * g_pixel[2];
	gSum_window.g3g3 += g_pixel[2] * g_pixel[2];
}
//fgx
template<int n>
inline void addPixelToWindow(fgx<n>* histo_window, fgxSumUpToIndex<n>& sumUpToIndex_window, const int& index_pixel, const cv::Vec<int, n>& g_pixel) {
	histo_window[index_pixel].f += 1;
	for (int i = 0; i < n; i++)
	{
		histo_window[index_pixel].g[i] += g_pixel[i];
	}
	if (index_pixel <= sumUpToIndex_window.index) {
		sumUpToIndex_window.f += 1;
		for (int i = 0; i < n; i++)
		{
			sumUpToIndex_window.g[i] += g_pixel[i];
		}
	}
}
template<int n>
inline void addPixelToWindow_gSum(fgx<n>* histo_window, fgxSumUpToIndex<n>& sumUpToIndex_window, const int& index_pixel, const cv::Vec<int, n>& g_pixel, gxSum<n>& gSum_window) {
	addPixelToWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	for (int i = 0; i < n; i++)
	{
		gSum_window.g[i] += g_pixel[i];
	}
	int k = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = i; j < n; j++)
		{
			gSum_window.gg[k] += g_pixel[i] * g_pixel[j];
			k++;
		}
	}
}
//list
//fg_list
inline void addPixelToWindow(std::list<fg_node>& histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel) {
	/*
windowはlistを持っていて、window.histo[pos]が列posのヒストグラムリスト
これに関して、リストのindex要素を順にみていき
*pixel_fより大きい最初の要素が見つかった場合は、その一つ前にそのindex(*pixel_f), f=1、g=*pixel_g　を登録
*pixel_fと同じものが見つかった場合は、*pixel_fに1追加、またそこのgに*pixel_g追加
また、リストの最後まで到達した場合は最後に登録
*/
	for (auto itr = histo_window.begin(); itr != histo_window.end(); ++itr) {
		if ((*itr).index > index_pixel)
		{
			histo_window.insert(itr, fg_node(1, g_pixel, index_pixel));
			if (index_pixel <= sumUpToIndex_window.index) {
				sumUpToIndex_window.f += 1;
				sumUpToIndex_window.g += g_pixel;
			}
			return;
		}
		else if ((*itr).index == index_pixel)
		{
			(*itr).f += 1;
			(*itr).g += g_pixel;
			if (index_pixel <= sumUpToIndex_window.index) {
				sumUpToIndex_window.f += 1;
				sumUpToIndex_window.g += g_pixel;
			}
			return;
		}
	}
	histo_window.push_back(fg_node(1, g_pixel, index_pixel));
	if (index_pixel <= sumUpToIndex_window.index) {
		sumUpToIndex_window.f += 1;
		sumUpToIndex_window.g += g_pixel;
	}
	return;
}
inline void addPixelToWindow_gSum(std::list<fg_node>& histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	addPixelToWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colに加算
	gSum_window.g += g_pixel;
	gSum_window.gg += g_pixel * g_pixel;
}
//fgSumUpToIndex なし
template<typename FG_NODE, typename GTYPE>
inline void addPixelToWindow(std::list<FG_NODE>& histo_window, const int& index_pixel, const GTYPE& g_pixel) {
	for (auto itr = histo_window.begin(); itr != histo_window.end(); ++itr) {
		if ((*itr).index > index_pixel)
		{
			histo_window.insert(itr, FG_NODE(1, g_pixel, index_pixel));
			return;
		}
		else if ((*itr).index == index_pixel)
		{
			(*itr).f += 1;
			(*itr).g += g_pixel;
			return;
		}
	}
	histo_window.push_back(FG_NODE(1, g_pixel, index_pixel));
	return;
}
//g1
inline void addPixelToWindow_gSum(std::list<fg_node>& histo_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	addPixelToWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colに加算
	gSum_window.g += g_pixel;
	gSum_window.gg += g_pixel * g_pixel;
}
//g3
inline void addPixelToWindow_gSum(std::list<fg3_node>& histo_window, const int& index_pixel, const cv::Vec3i& g_pixel, g3Sum& gSum_window) {
	addPixelToWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colに加算
	gSum_window.g1 += g_pixel[0];
	gSum_window.g2 += g_pixel[1];
	gSum_window.g3 += g_pixel[2];
	gSum_window.g1g1 += g_pixel[0] * g_pixel[0];
	gSum_window.g1g2 += g_pixel[0] * g_pixel[1];
	gSum_window.g1g3 += g_pixel[0] * g_pixel[2];
	gSum_window.g2g2 += g_pixel[1] * g_pixel[1];
	gSum_window.g2g3 += g_pixel[1] * g_pixel[2];
	gSum_window.g3g3 += g_pixel[2] * g_pixel[2];
}

///////////////////////////
//removePixel
//fg
inline void removePixelFromWindow(fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel) {
	//前の行のgとf取得
	//ヒストグラムから減算
	histo_window[index_pixel].f -= 1;
	histo_window[index_pixel].g -= g_pixel;

	if (index_pixel <= sumUpToIndex_window.index) {
		sumUpToIndex_window.f -= 1;
		sumUpToIndex_window.g -= g_pixel;
	}
	/*
	int flag = index_pixel <= sumUpToIndex_window.index;
	sumUpToIndex_window.f -= flag;
	sumUpToIndex_window.g -= g_pixel * flag;
	*/
}
inline void removePixelFromWindow_gSum(fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	removePixelFromWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colから減算
	gSum_window.g -= g_pixel;
	gSum_window.gg -= g_pixel * g_pixel;
}
//sumuptoindex無しテスト
inline void removePixelFromWindow(fg* histo_window, const int& index_pixel, const int& g_pixel) {
	//前の行のgとf取得
	//ヒストグラムから減算
	histo_window[index_pixel].f -= 1;
	histo_window[index_pixel].g -= g_pixel;
}
inline void removePixelFromWindow_gSum(fg* histo_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	removePixelFromWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colから減算
	gSum_window.g -= g_pixel;
	gSum_window.gg -= g_pixel * g_pixel;
}

//fg3
inline void removePixelFromWindow(fg3* histo_window, fg3SumUpToIndex& sumUpToIndex_window, const int& index_pixel, const cv::Vec3i& g_pixel) {
	histo_window[index_pixel].f -= 1;
	histo_window[index_pixel].g1 -= g_pixel[0];
	histo_window[index_pixel].g2 -= g_pixel[1];
	histo_window[index_pixel].g3 -= g_pixel[2];
	if (index_pixel <= sumUpToIndex_window.index) {
		sumUpToIndex_window.f -= 1;
		sumUpToIndex_window.g1 -= g_pixel[0];
		sumUpToIndex_window.g2 -= g_pixel[1];
		sumUpToIndex_window.g3 -= g_pixel[2];
	}

}
inline void removePixelFromWindow_gSum(fg3* histo_window, fg3SumUpToIndex& sumUpToIndex_window, const int& index_pixel, const cv::Vec3i& g_pixel, g3Sum& gSum_window) {
	removePixelFromWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	gSum_window.g1 -= g_pixel[0];
	gSum_window.g2 -= g_pixel[1];
	gSum_window.g3 -= g_pixel[2];
	gSum_window.g1g1 -= g_pixel[0] * g_pixel[0];
	gSum_window.g1g2 -= g_pixel[0] * g_pixel[1];
	gSum_window.g1g3 -= g_pixel[0] * g_pixel[2];
	gSum_window.g2g2 -= g_pixel[1] * g_pixel[1];
	gSum_window.g2g3 -= g_pixel[1] * g_pixel[2];
	gSum_window.g3g3 -= g_pixel[2] * g_pixel[2];
}
//fgx
template<int n>
inline void removePixelFromWindow(fgx<n>* histo_window, fgxSumUpToIndex<n>& sumUpToIndex_window, const int& index_pixel, const cv::Vec<int, n>& g_pixel) {
	histo_window[index_pixel].f -= 1;
	for (int i = 0; i < n; i++)
	{
		histo_window[index_pixel].g[i] -= g_pixel[i];
	}
	if (index_pixel <= sumUpToIndex_window.index) {
		sumUpToIndex_window.f -= 1;
		for (int i = 0; i < n; i++)
		{
			sumUpToIndex_window.g[i] -= g_pixel[i];
		}
	}
}
template<int n>
inline void removePixelFromWindow_gSum(fgx<n>* histo_window, fgxSumUpToIndex<n>& sumUpToIndex_window, const int& index_pixel, const cv::Vec<int, n>& g_pixel, gxSum<n>& gSum_window) {
	removePixelFromWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	for (int i = 0; i < n; i++)
	{
		gSum_window.g[i] -= g_pixel[i];
	}
	int k = 0;
	for (int i = 0; i < n; i++)
	{
		for (int j = i; j < n; j++)
		{
			gSum_window.gg[k] -= g_pixel[i] * g_pixel[j];
			k++;
		}
	}
}
//list
//fg_list
inline void removePixelFromWindow(std::list<fg_node>& histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel) {
	//削除対象は必ずリスト中に存在する
	for (auto itr = histo_window.begin(); itr != histo_window.end(); ++itr) {
		if ((*itr).index == index_pixel)
		{
			if ((*itr).f == 1)
			{
				//リストから削除
				histo_window.erase(itr);
				if (index_pixel <= sumUpToIndex_window.index) {
					sumUpToIndex_window.f -= 1;
					sumUpToIndex_window.g -= g_pixel;
				}
				return;
			}
			(*itr).f -= 1;
			(*itr).g -= g_pixel;
			if (index_pixel <= sumUpToIndex_window.index) {
				sumUpToIndex_window.f -= 1;
				sumUpToIndex_window.g -= g_pixel;
			}
			return;
		}
	}
}
inline void removePixelFromWindow_gSum(std::list<fg_node>& histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	removePixelFromWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colから減算
	gSum_window.g -= g_pixel;
	gSum_window.gg -= g_pixel * g_pixel;
}
//fgSumUpToIndexなし
template<typename FG_NODE, typename GTYPE>
inline void removePixelFromWindow(std::list<FG_NODE>& histo_window, const int& index_pixel, const GTYPE& g_pixel) {
	//削除対象は必ずリスト中に存在する
	for (auto itr = histo_window.begin(); itr != histo_window.end(); ++itr) {
		if ((*itr).index == index_pixel)
		{
			if ((*itr).f == 1)
			{
				//リストから削除
				histo_window.erase(itr);
				return;
			}
			(*itr).f -= 1;
			(*itr).g -= g_pixel;
			return;
		}
	}
}
//g1
inline void removePixelFromWindow_gSum(std::list<fg_node>& histo_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	removePixelFromWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colから減算
	gSum_window.g -= g_pixel;
	gSum_window.gg -= g_pixel * g_pixel;
}
//g3
inline void removePixelFromWindow_gSum(std::list<fg3_node>& histo_window, const int& index_pixel, const cv::Vec3i& g_pixel, g3Sum& gSum_window) {
	removePixelFromWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_colに加算
	gSum_window.g1 -= g_pixel[0];
	gSum_window.g2 -= g_pixel[1];
	gSum_window.g3 -= g_pixel[2];
	gSum_window.g1g1 -= g_pixel[0] * g_pixel[0];
	gSum_window.g1g2 -= g_pixel[0] * g_pixel[1];
	gSum_window.g1g3 -= g_pixel[0] * g_pixel[2];
	gSum_window.g2g2 -= g_pixel[1] * g_pixel[1];
	gSum_window.g2g3 -= g_pixel[1] * g_pixel[2];
	gSum_window.g3g3 -= g_pixel[2] * g_pixel[2];
}

///////////////////////////
//update window
inline void updateSubWindowAndAddToWindow(const int& Imax, fg* histo_window, fg* histo_subwin, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, const int& index_window, fgSumUpToIndex& sumUpToIndex_subwin) {
	addHistogram(Imax, histo_window, histo_subwin);

	const int u_idx = sumUpToIndex_subwin.index;
	const int flag = u_idx < index_window;
	const int sign = flag * 2 - 1;
	const int startIdx = u_idx + flag;
	const int numIdx = (index_window - u_idx) * sign;
	//ここでは追加予定のウィンドウのunder_i要素を更新する
	//ここsignが負だった時に逆順にアクセスするからキャッシュ効率が悪いのでは
	//順になるように変更した方が良いかも
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		//追加
		sumUpToIndex_subwin.f += histo_subwin[i].f * sign;
		sumUpToIndex_subwin.g += histo_subwin[i].g * sign;
	}
	//indexの更新
	sumUpToIndex_subwin.index = index_window;
	//更新したウィンドウを追加することでウィンドウの更新をする
	//ウィンドウ内f_sum, g_sumの更新
	sumUpToIndex_window_f += sumUpToIndex_subwin.f;
	sumUpToIndex_window_g += sumUpToIndex_subwin.g;
}
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, const int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window_f, sumUpToIndex_window_g, index_window, sumUpToIndex_subwin);
	//g, ggの更新：sum_windowに追加列を加算
	gSum_window.g += gSum_subwin.g;
	gSum_window.gg += gSum_subwin.gg;
}
//fg
inline void updateSubWindowAndAddToWindow(const int& Imax, fg* histo_window, fg* histo_subwin, fgSumUpToIndex& sumUpToIndex_window, fgSumUpToIndex& sumUpToIndex_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window.f, sumUpToIndex_window.g, sumUpToIndex_window.index, sumUpToIndex_subwin);
}
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, fgSumUpToIndex& sumUpToIndex_window, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_subwin, sumUpToIndex_window.index, sumUpToIndex_window.f, sumUpToIndex_window.g, sumUpToIndex_subwin, gSum_window, gSum_subwin);
}
//sumuptoindex無しテスト
inline void updateSubWindowAndAddToWindow(const int& Imax, fg* histo_window, fg* histo_subwin) {
	addHistogram(Imax, histo_window, histo_subwin);
}
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin);
	//g, ggの更新：sum_windowに追加列を加算
	gSum_window.g += gSum_subwin.g;
	gSum_window.gg += gSum_subwin.gg;
}

//fg3
inline void updateSubWindowAndAddToWindow(const int& Imax, fg3* histo_window, fg3* histo_subwin, fg3SumUpToIndex& sumUpToIndex_window, fg3SumUpToIndex& sumUpToIndex_subwin) {
	addHistogram(Imax, histo_window, histo_subwin);

	const int u_idx = sumUpToIndex_subwin.index;
	const int flag = u_idx < sumUpToIndex_window.index;
	const int sign = flag * 2 - 1;
	const int startIdx = u_idx + flag;
	const int numIdx = (sumUpToIndex_window.index - u_idx) * sign;
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		sumUpToIndex_subwin.f += histo_subwin[i].f * sign;
		sumUpToIndex_subwin.g1 += histo_subwin[i].g1 * sign;
		sumUpToIndex_subwin.g2 += histo_subwin[i].g2 * sign;
		sumUpToIndex_subwin.g3 += histo_subwin[i].g3 * sign;
	}
	sumUpToIndex_subwin.index = sumUpToIndex_window.index;
	sumUpToIndex_window.f += sumUpToIndex_subwin.f;
	sumUpToIndex_window.g1 += sumUpToIndex_subwin.g1;
	sumUpToIndex_window.g2 += sumUpToIndex_subwin.g2;
	sumUpToIndex_window.g3 += sumUpToIndex_subwin.g3;
}

inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, fg3* histo_window, fg3* histo_subwin, fg3SumUpToIndex& sumUpToIndex_window, fg3SumUpToIndex& sumUpToIndex_subwin, g3Sum& gSum_window, g3Sum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window, sumUpToIndex_subwin);
	gSum_window.g1 += gSum_subwin.g1;
	gSum_window.g2 += gSum_subwin.g2;
	gSum_window.g3 += gSum_subwin.g3;
	gSum_window.g1g1 += gSum_subwin.g1g1;
	gSum_window.g1g2 += gSum_subwin.g1g2;
	gSum_window.g1g3 += gSum_subwin.g1g3;
	gSum_window.g2g2 += gSum_subwin.g2g2;
	gSum_window.g2g3 += gSum_subwin.g2g3;
	gSum_window.g3g3 += gSum_subwin.g3g3;
}
//fgx
template<int n>
inline void updateSubWindowAndAddToWindow(const int& Imax, fgx<n>* histo_window, fgx<n>* histo_subwin, fgxSumUpToIndex<n>& sumUpToIndex_window, fgxSumUpToIndex<n>& sumUpToIndex_subwin) {
	addHistogram(Imax, histo_window, histo_subwin);

	const int u_idx = sumUpToIndex_subwin.index;
	const int flag = u_idx < sumUpToIndex_window.index;
	const int sign = flag * 2 - 1;
	const int startIdx = u_idx + flag;
	const int numIdx = (sumUpToIndex_window.index - u_idx) * sign;
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		sumUpToIndex_subwin.f += histo_subwin[i].f * sign;
		for (int k = 0; k < n; k++)
		{
			sumUpToIndex_subwin.g[k] += histo_subwin[i].g[k] * sign;
		}
	}
	sumUpToIndex_subwin.index = sumUpToIndex_window.index;
	sumUpToIndex_window.f += sumUpToIndex_subwin.f;
	for (int k = 0; k < n; k++)
	{
		sumUpToIndex_window.g[k] += sumUpToIndex_subwin.g[k];
	}
}
template<int n>
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, fgx<n>* histo_window, fgx<n>* histo_subwin, fgxSumUpToIndex<n>& sumUpToIndex_window, fgxSumUpToIndex<n>& sumUpToIndex_subwin, gxSum<n>& gSum_window, gxSum<n> gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window, sumUpToIndex_subwin);
	for (int i = 0; i < n; i++)
	{
		gSum_window.g[i] += gSum_subwin.g[i];
	}
	for (int i = 0; i < ((n + 1)*n) / 2; i++)
	{
		gSum_window.gg[i] += gSum_subwin.gg[i];
	}
}

//list
inline void updateSubWindowAndAddToWindow(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, const int& index_window, fgSumUpToIndex& sumUpToIndex_subwin) {
	addHistogram(Imax, histo_window, histo_subwin);

	const int u_idx = sumUpToIndex_subwin.index;
	//ここでは追加予定のウィンドウのunder_i要素を更新する
	//u_idx < index_windowの場合、u_idx+1～index_windowまでのhisto_subwinの f,gをsumUpToIndex_subwin　に追加する
	//u_idx > index_windowの場合、index_window+1～u_idxまでのhisto_subwinの f,gをsumUpToIndex_subwin　から削除する
	if (u_idx < index_window)
	{
		auto itr_subwin = histo_subwin.begin();
		auto end_subwin = histo_subwin.end();
		while (itr_subwin != end_subwin) {
			if ((*itr_subwin).index > index_window)
			{
				break;
			}
			else if ((*itr_subwin).index > u_idx)
			{
				sumUpToIndex_subwin.f += (*itr_subwin).f;
				sumUpToIndex_subwin.g += (*itr_subwin).g;
			}
			itr_subwin++;
		}
	}
	else if(u_idx > index_window)
	{
		auto itr_subwin = histo_subwin.begin();
		auto end_subwin = histo_subwin.end();
		while (itr_subwin != end_subwin) {
			if ((*itr_subwin).index > u_idx)
			{
				break;
			}
			else if ((*itr_subwin).index > index_window)
			{
				sumUpToIndex_subwin.f -= (*itr_subwin).f;
				sumUpToIndex_subwin.g -= (*itr_subwin).g;
			}
			itr_subwin++;
		}
	}
	//indexの更新
	sumUpToIndex_subwin.index = index_window;
	//更新したウィンドウを追加することでウィンドウの更新をする
	//ウィンドウ内f_sum, g_sumの更新
	sumUpToIndex_window_f += sumUpToIndex_subwin.f;
	sumUpToIndex_window_g += sumUpToIndex_subwin.g;
}
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, const int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window_f, sumUpToIndex_window_g, index_window, sumUpToIndex_subwin);
	//g, ggの更新：sum_windowに追加列を加算
	gSum_window.g += gSum_subwin.g;
	gSum_window.gg += gSum_subwin.gg;
}
//fg_list
inline void updateSubWindowAndAddToWindow(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, fgSumUpToIndex& sumUpToIndex_window, fgSumUpToIndex& sumUpToIndex_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window.f, sumUpToIndex_window.g, sumUpToIndex_window.index, sumUpToIndex_subwin);
}
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, fgSumUpToIndex& sumUpToIndex_window, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_subwin, sumUpToIndex_window.index, sumUpToIndex_window.f, sumUpToIndex_window.g, sumUpToIndex_subwin, gSum_window, gSum_subwin);
}
//fgSumUpToIndexなし
template<typename FG_NODE>
inline void updateSubWindowAndAddToWindow(const int& Imax, std::list<FG_NODE>& histo_window, std::list<FG_NODE>& histo_subwin) {
	addHistogram(Imax, histo_window, histo_subwin);
}
//g1
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin);
	//g, ggの更新：sum_windowに追加列を加算
	gSum_window.g += gSum_subwin.g;
	gSum_window.gg += gSum_subwin.gg;
}
//g3
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, std::list<fg3_node>& histo_window, std::list<fg3_node>& histo_subwin, g3Sum& gSum_window, g3Sum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin);
	//g, ggの更新：sum_windowに追加列を加算
	gSum_window.g1 += gSum_subwin.g1;
	gSum_window.g2 += gSum_subwin.g2;
	gSum_window.g3 += gSum_subwin.g3;
	gSum_window.g1g1 += gSum_subwin.g1g1;
	gSum_window.g1g2 += gSum_subwin.g1g2;
	gSum_window.g1g3 += gSum_subwin.g1g3;
	gSum_window.g2g2 += gSum_subwin.g2g2;
	gSum_window.g2g3 += gSum_subwin.g2g3;
	gSum_window.g3g3 += gSum_subwin.g3g3;
}



///////////////////////////
//remove window
inline void removeSubWindowFromWindow(const int& Imax, const int& index_window, fg* histo_window, fg* histo_subwin, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin) {
	remHistogram(Imax, histo_window, histo_subwin);

	const int u_idx = sumUpToIndex_subwin.index;
	const int flag = u_idx < index_window;
	const int sign = flag * 2 - 1;
	const int startIdx = u_idx + flag;
	const int numIdx = (index_window - u_idx) * sign;
	//削除時はウィンドウからunder_i要素を直接削除する
	//削除するウィンドウの情報は、追加するときに更新したunder_iの情報のまま、
	//一方今のインデックスはindexなので、直接削除ウィンドウの情報を引き算すると
	//合わない。なのでまずその分の調整をして
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		//削除
		sumUpToIndex_window_f -= histo_subwin[i].f * sign;
		sumUpToIndex_window_g -= histo_subwin[i].g * sign;
	}
	//次に削除ウィンドウ分を減らす
	//ウィンドウ内f_sum, g_sumの更新
	sumUpToIndex_window_f -= sumUpToIndex_subwin.f;
	sumUpToIndex_window_g -= sumUpToIndex_subwin.g;
}

inline void removeSubWindowFromWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, const int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow(Imax, index_window, histo_window, histo_subwin, sumUpToIndex_window_f, sumUpToIndex_window_g, sumUpToIndex_subwin);
	//g, ggの更新：sum_windowから削除列を減算
	gSum_window.g -= gSum_subwin.g;
	gSum_window.gg -= gSum_subwin.gg;
}
//fg
inline void removeSubWindowFromWindow(const int& Imax, fg* histo_window, fg* histo_subwin, fgSumUpToIndex& sumUpToIndex_window, fgSumUpToIndex& sumUpToIndex_subwin) {
	removeSubWindowFromWindow(Imax, sumUpToIndex_window.index, histo_window, histo_subwin, sumUpToIndex_window.f, sumUpToIndex_window.g, sumUpToIndex_subwin);
}
inline void removeSubWindowFromWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, fgSumUpToIndex& sumUpToIndex_window, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow_gSum(Imax, histo_window, histo_subwin, sumUpToIndex_window.index, sumUpToIndex_window.f, sumUpToIndex_window.g, sumUpToIndex_subwin, gSum_window, gSum_subwin);
}
//sumuptoindex無しテスト
inline void removeSubWindowFromWindow(const int& Imax, fg* histo_window, fg* histo_subwin) {
	remHistogram(Imax, histo_window, histo_subwin);
}

inline void removeSubWindowFromWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow(Imax, histo_window, histo_subwin);
	//g, ggの更新：sum_windowから削除列を減算
	gSum_window.g -= gSum_subwin.g;
	gSum_window.gg -= gSum_subwin.gg;
}

//fg3
inline void removeSubWindowFromWindow(const int& Imax, fg3* histo_window, fg3* histo_subwin, fg3SumUpToIndex& sumUpToIndex_window, fg3SumUpToIndex& sumUpToIndex_subwin) {
	remHistogram(Imax, histo_window, histo_subwin);

	const int u_idx = sumUpToIndex_subwin.index;
	const int flag = u_idx < sumUpToIndex_window.index;
	const int sign = flag * 2 - 1;
	const int startIdx = u_idx + flag;
	const int numIdx = (sumUpToIndex_window.index - u_idx) * sign;
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		sumUpToIndex_window.f -= histo_subwin[i].f * sign;
		sumUpToIndex_window.g1 -= histo_subwin[i].g1 * sign;
		sumUpToIndex_window.g2 -= histo_subwin[i].g2 * sign;
		sumUpToIndex_window.g3 -= histo_subwin[i].g3 * sign;
	}
	sumUpToIndex_window.f -= sumUpToIndex_subwin.f;
	sumUpToIndex_window.g1 -= sumUpToIndex_subwin.g1;
	sumUpToIndex_window.g2 -= sumUpToIndex_subwin.g2;
	sumUpToIndex_window.g3 -= sumUpToIndex_subwin.g3;
}
inline void removeSubWindowFromWindow_gSum(const int& Imax, fg3* histo_window, fg3* histo_subwin, fg3SumUpToIndex& sumUpToIndex_window, fg3SumUpToIndex& sumUpToIndex_subwin, g3Sum& gSum_window, g3Sum gSum_subwin) {
	removeSubWindowFromWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window, sumUpToIndex_subwin);
	gSum_window.g1 -= gSum_subwin.g1;
	gSum_window.g2 -= gSum_subwin.g2;
	gSum_window.g3 -= gSum_subwin.g3;
	gSum_window.g1g1 -= gSum_subwin.g1g1;
	gSum_window.g1g2 -= gSum_subwin.g1g2;
	gSum_window.g1g3 -= gSum_subwin.g1g3;
	gSum_window.g2g2 -= gSum_subwin.g2g2;
	gSum_window.g2g3 -= gSum_subwin.g2g3;
	gSum_window.g3g3 -= gSum_subwin.g3g3;
}
//fgx
template<int n>
inline void removeSubWindowFromWindow(const int& Imax, fgx<n>* histo_window, fgx<n>* histo_subwin, fgxSumUpToIndex<n>& sumUpToIndex_window, fgxSumUpToIndex<n>& sumUpToIndex_subwin) {
	remHistogram(Imax, histo_window, histo_subwin);

	const int u_idx = sumUpToIndex_subwin.index;
	const int flag = u_idx < sumUpToIndex_window.index;
	const int sign = flag * 2 - 1;
	const int startIdx = u_idx + flag;
	const int numIdx = (sumUpToIndex_window.index - u_idx) * sign;
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		sumUpToIndex_window.f -= histo_subwin[i].f * sign;
		for (int k = 0; k < n; k++)
		{
			sumUpToIndex_window.g[k] -= histo_subwin[i].g[k] * sign;
		}
	}
	sumUpToIndex_window.f -= sumUpToIndex_subwin.f;
	for (int k = 0; k < n; k++)
	{
		sumUpToIndex_window.g[k] -= sumUpToIndex_subwin.g[k];
	}
}
template<int n>
inline void removeSubWindowFromWindow_gSum(const int& Imax, fgx<n>* histo_window, fgx<n>* histo_subwin, fgxSumUpToIndex<n>& sumUpToIndex_window, fgxSumUpToIndex<n>& sumUpToIndex_subwin, gxSum<n>& gSum_window, gxSum<n> gSum_subwin) {
	removeSubWindowFromWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window, sumUpToIndex_subwin);
	for (int i = 0; i < n; i++)
	{
		gSum_window.g[i] -= gSum_subwin.g[i];
	}
	for (int i = 0; i < ((n + 1) * n) / 2; i++)
	{
		gSum_window.gg[i] -= gSum_subwin.gg[i];
	}
}


//list
inline void removeSubWindowFromWindow(const int& Imax, const int& index_window, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin) {
	remHistogram(Imax, histo_window, histo_subwin);

	const int u_idx = sumUpToIndex_subwin.index;
	//削除時はウィンドウからunder_i要素を直接削除する
	//削除するウィンドウの情報は、追加するときに更新したunder_iの情報のまま、
	//一方今のインデックスはindexなので、直接削除ウィンドウの情報を引き算すると
	//合わない。なのでまずその分の調整をして
	//u_idx < index_windowの場合、u_idx+1～index_windowまでのhisto_subwinの f,gをsumUpToIndex_subwin　に追加する
	//u_idx > index_windowの場合、index_window+1～u_idxまでのhisto_subwinの f,gをsumUpToIndex_subwin　から削除する
	if (u_idx < index_window)
	{
		auto itr_subwin = histo_subwin.begin();
		auto end_subwin = histo_subwin.end();
		while (itr_subwin != end_subwin) {
			if ((*itr_subwin).index > index_window)
			{
				break;
			}
			else if ((*itr_subwin).index > u_idx)
			{
				sumUpToIndex_window_f -= (*itr_subwin).f;
				sumUpToIndex_window_g -= (*itr_subwin).g;
			}
			itr_subwin++;
		}
	}
	else if (u_idx > index_window)
	{
		auto itr_subwin = histo_subwin.begin();
		auto end_subwin = histo_subwin.end();
		while (itr_subwin != end_subwin) {
			if ((*itr_subwin).index > u_idx)
			{
				break;
			}
			else if ((*itr_subwin).index > index_window)
			{
				sumUpToIndex_window_f += (*itr_subwin).f;
				sumUpToIndex_window_g += (*itr_subwin).g;
			}
			itr_subwin++;
		}
	}
	/*
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		//削除
		sumUpToIndex_window_f -= histo_subwin[i].f * sign;
		sumUpToIndex_window_g -= histo_subwin[i].g * sign;
	}*/
	//次に削除ウィンドウ分を減らす
	//ウィンドウ内f_sum, g_sumの更新
	sumUpToIndex_window_f -= sumUpToIndex_subwin.f;
	sumUpToIndex_window_g -= sumUpToIndex_subwin.g;
}

inline void removeSubWindowFromWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, const int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow(Imax, index_window, histo_window, histo_subwin, sumUpToIndex_window_f, sumUpToIndex_window_g, sumUpToIndex_subwin);
	//g, ggの更新：sum_windowから削除列を減算
	gSum_window.g -= gSum_subwin.g;
	gSum_window.gg -= gSum_subwin.gg;
}
//fg
inline void removeSubWindowFromWindow(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, fgSumUpToIndex& sumUpToIndex_window, fgSumUpToIndex& sumUpToIndex_subwin) {
	removeSubWindowFromWindow(Imax, sumUpToIndex_window.index, histo_window, histo_subwin, sumUpToIndex_window.f, sumUpToIndex_window.g, sumUpToIndex_subwin);
}
inline void removeSubWindowFromWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, fgSumUpToIndex& sumUpToIndex_window, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow_gSum(Imax, histo_window, histo_subwin, sumUpToIndex_window.index, sumUpToIndex_window.f, sumUpToIndex_window.g, sumUpToIndex_subwin, gSum_window, gSum_subwin);
}
//fgSumUpToIndexなし
template<typename FG_NODE>
inline void removeSubWindowFromWindow(const int& Imax, std::list<FG_NODE>& histo_window, std::list<FG_NODE>& histo_subwin) {
	remHistogram(Imax, histo_window, histo_subwin);
}
//g1
inline void removeSubWindowFromWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow(Imax, histo_window, histo_subwin);
	//g, ggの更新：sum_windowから削除列を減算
	gSum_window.g -= gSum_subwin.g;
	gSum_window.gg -= gSum_subwin.gg;
}
inline void removeSubWindowFromWindow_gSum(const int& Imax, std::list<fg3_node>& histo_window, std::list<fg3_node>& histo_subwin, g3Sum& gSum_window, g3Sum gSum_subwin) {
	removeSubWindowFromWindow(Imax, histo_window, histo_subwin);
	//g, ggの更新：sum_windowから削除列を減算
	gSum_window.g1 -= gSum_subwin.g1;
	gSum_window.g2 -= gSum_subwin.g2;
	gSum_window.g3 -= gSum_subwin.g3;
	gSum_window.g1g1 -= gSum_subwin.g1g1;
	gSum_window.g1g2 -= gSum_subwin.g1g2;
	gSum_window.g1g3 -= gSum_subwin.g1g3;
	gSum_window.g2g2 -= gSum_subwin.g2g2;
	gSum_window.g2g3 -= gSum_subwin.g2g3;
	gSum_window.g3g3 -= gSum_subwin.g3g3;
}


///////////////////////////
inline void calculateCxDx(const gSum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, float& cx, float& dx) {
	float g_ave = gSum_window.g * pixel_sum_window_inv;
	float gg_ave = gSum_window.gg * pixel_sum_window_inv;
	float vx = gg_ave - g_ave * g_ave + eps2;
	float tmp = G_center - g_ave;
	cx = tmp * pixel_sum_window_inv / vx;
	dx = pixel_sum_window_inv - g_ave * cx;
}
//g3 vec
inline void calculateCxDx(const g3Sum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const cv::Vec3i& G_center, cv::Vec3f& cx, float& dx) {
	const float g_ave1 = gSum_window.g1 * pixel_sum_window_inv;
	const float g_ave2 = gSum_window.g2 * pixel_sum_window_inv;
	const float g_ave3 = gSum_window.g3 * pixel_sum_window_inv;
	const float v11 = gSum_window.g1g1 * pixel_sum_window_inv - g_ave1 * g_ave1 + eps2;
	const float v12 = gSum_window.g1g2 * pixel_sum_window_inv - g_ave1 * g_ave2;
	const float v13 = gSum_window.g1g3 * pixel_sum_window_inv - g_ave1 * g_ave3;
	const float v22 = gSum_window.g2g2 * pixel_sum_window_inv - g_ave2 * g_ave2 + eps2;
	const float v23 = gSum_window.g2g3 * pixel_sum_window_inv - g_ave2 * g_ave3;
	const float v33 = gSum_window.g3g3 * pixel_sum_window_inv - g_ave3 * g_ave3 + eps2;
	const float delta =
		v11 * v22 * v33 +
		v12 * v23 * v13 * 2 -
		v13 * v13 * v22 -
		v12 * v12 * v33 -
		v11 * v23 * v23;
	float deltaInv = 1.0f / delta;
	if (delta == 0)
	{
		printf("inf ");
		deltaInv = 0.0f;// 1000000000000.0f;
	}
	const float vinv11 = (v22 * v33 - v23 * v23);
	const float vinv12 = (v13 * v23 - v12 * v33);
	const float vinv13 = (v12 * v23 - v13 * v22);
	const float vinv22 = (v11 * v33 - v13 * v13);
	const float vinv23 = (v13 * v12 - v11 * v23);
	const float vinv33 = (v11 * v22 - v12 * v12);
	const float tmp1 = G_center[0] - g_ave1;
	const float tmp2 = G_center[1] - g_ave2;
	const float tmp3 = G_center[2] - g_ave3;
	const float mult = pixel_sum_window_inv * deltaInv;
	cx[0] = (tmp1 * vinv11 + tmp2 * vinv12 + tmp3 * vinv13) * mult;
	cx[1] = (tmp1 * vinv12 + tmp2 * vinv22 + tmp3 * vinv23) * mult;
	cx[2] = (tmp1 * vinv13 + tmp2 * vinv23 + tmp3 * vinv33) * mult;
	dx = pixel_sum_window_inv - g_ave1 * cx[0] - g_ave2 * cx[1] - g_ave3 * cx[2];
}
//g3 (もう使われていないか？確認して削除)
inline void calculateCxDx(const g3Sum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const cv::Vec3i& G_center, float* cx, float& dx) {
	const float g_ave1 = gSum_window.g1 * pixel_sum_window_inv;
	const float g_ave2 = gSum_window.g2 * pixel_sum_window_inv;
	const float g_ave3 = gSum_window.g3 * pixel_sum_window_inv;
	const float v11 = gSum_window.g1g1 * pixel_sum_window_inv - g_ave1 * g_ave1 + eps2;
	const float v12 = gSum_window.g1g2 * pixel_sum_window_inv - g_ave1 * g_ave2;
	const float v13 = gSum_window.g1g3 * pixel_sum_window_inv - g_ave1 * g_ave3;
	const float v22 = gSum_window.g2g2 * pixel_sum_window_inv - g_ave2 * g_ave2 + eps2;
	const float v23 = gSum_window.g2g3 * pixel_sum_window_inv - g_ave2 * g_ave3;
	const float v33 = gSum_window.g3g3 * pixel_sum_window_inv - g_ave3 * g_ave3 + eps2;
	const float delta =
		v11 * v22 * v33 +
		v12 * v23 * v13 * 2 -
		v13 * v13 * v22 -
		v12 * v12 * v33 -
		v11 * v23 * v23;
	float deltaInv = 1.0f / delta;
	if (delta == 0)
	{
		printf("inf ");
		deltaInv = 0.0f;// 1000000000000.0f;
	}
	const float vinv11 = (v22 * v33 - v23 * v23);
	const float vinv12 = (v13 * v23 - v12 * v33);
	const float vinv13 = (v12 * v23 - v13 * v22);
	const float vinv22 = (v11 * v33 - v13 * v13);
	const float vinv23 = (v13 * v12 - v11 * v23);
	const float vinv33 = (v11 * v22 - v12 * v12);
	const float tmp1 = G_center[0] - g_ave1;
	const float tmp2 = G_center[1] - g_ave2;
	const float tmp3 = G_center[2] - g_ave3;
	const float mult = pixel_sum_window_inv * deltaInv;
	cx[0] = (tmp1 * vinv11 + tmp2 * vinv12 + tmp3 * vinv13) * mult;
	cx[1] = (tmp1 * vinv12 + tmp2 * vinv22 + tmp3 * vinv23) * mult;
	cx[2] = (tmp1 * vinv13 + tmp2 * vinv23 + tmp3 * vinv33) * mult;
	dx = pixel_sum_window_inv - g_ave1 * cx[0] - g_ave2 * cx[1] - g_ave3 * cx[2];
}
//gx (blas使う)
template<int n>
inline void calculateCxDx(const gxSum<n>& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const cv::Vec<int, n>& G_center, cv::Vec<float, n>& cx, float& dx) {
	/*
	計算したいのは、
	cx=(g-g_ave)^T V_inv
	であり、これは
	(g-g_ave)^T = cx V
	の解、または
	(g-g_ave) = V cx^T
	の解である。
	逆行列を直接計算するのではなく、この線形方程式の解を求めることでcxを計算する。
	そのために、blasを適切に呼び出せる形に変形しながら g-g_ave, Vを計算する
	*/
	const float g_ave1 = gSum_window.g1 * pixel_sum_window_inv;
	const float g_ave2 = gSum_window.g2 * pixel_sum_window_inv;
	const float g_ave3 = gSum_window.g3 * pixel_sum_window_inv;
	const float v11 = gSum_window.g1g1 * pixel_sum_window_inv - g_ave1 * g_ave1 + eps2;
	const float v12 = gSum_window.g1g2 * pixel_sum_window_inv - g_ave1 * g_ave2;
	const float v13 = gSum_window.g1g3 * pixel_sum_window_inv - g_ave1 * g_ave3;
	const float v22 = gSum_window.g2g2 * pixel_sum_window_inv - g_ave2 * g_ave2 + eps2;
	const float v23 = gSum_window.g2g3 * pixel_sum_window_inv - g_ave2 * g_ave3;
	const float v33 = gSum_window.g3g3 * pixel_sum_window_inv - g_ave3 * g_ave3 + eps2;
	const float delta =
		v11 * v22 * v33 +
		v12 * v23 * v13 * 2 -
		v13 * v13 * v22 -
		v12 * v12 * v33 -
		v11 * v23 * v23;
	float deltaInv = 1.0f / delta;
	if (delta == 0)
	{
		printf("inf ");
		deltaInv = 0.0f;// 1000000000000.0f;
	}
	const float vinv11 = (v22 * v33 - v23 * v23);
	const float vinv12 = (v13 * v23 - v12 * v33);
	const float vinv13 = (v12 * v23 - v13 * v22);
	const float vinv22 = (v11 * v33 - v13 * v13);
	const float vinv23 = (v13 * v12 - v11 * v23);
	const float vinv33 = (v11 * v22 - v12 * v12);
	const float tmp1 = G_center[0] - g_ave1;
	const float tmp2 = G_center[1] - g_ave2;
	const float tmp3 = G_center[2] - g_ave3;
	const float mult = pixel_sum_window_inv * deltaInv;
	cx[0] = (tmp1 * vinv11 + tmp2 * vinv12 + tmp3 * vinv13) * mult;
	cx[1] = (tmp1 * vinv12 + tmp2 * vinv22 + tmp3 * vinv23) * mult;
	cx[2] = (tmp1 * vinv13 + tmp2 * vinv23 + tmp3 * vinv33) * mult;
	dx = pixel_sum_window_inv - g_ave1 * cx[0] - g_ave2 * cx[1] - g_ave3 * cx[2];
}

///////////////////////////
//中央値計算
inline void findMedian(const float& cx, const float& dx, const float& half, fg* histo_window, int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, int& result_center) {
	float h = cx * sumUpToIndex_window_g + dx * sumUpToIndex_window_f;

	const int flagA = h < half;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	while (true)
	{
		index_window += flagA;
		//if(histo_window[index_window].f)//この判定を無くした方が速いかもしれない
		{
			//あるなら更新してhチェック
			sumUpToIndex_window_f += histo_window[index_window].f * sign;
			sumUpToIndex_window_g += histo_window[index_window].g * sign;
			h = cx * sumUpToIndex_window_g + dx * sumUpToIndex_window_f;
			if ((h >= half) == flagA)
			{
				//超えたのでこのindexがmedian
				result_center = index_window;
				index_window += flag2;
				break;
			}
		}
		index_window += flag2;
	}
}
//fg
inline void findMedian(const float& cx, const float& dx, const float& half, fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, int& result_center) {
	float h = cx * sumUpToIndex_window.g + dx * sumUpToIndex_window.f;

	const int flagA = h < half;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	while (true)
	{
		sumUpToIndex_window.index += flagA;
		/*
		if (sumUpToIndex_window.index >= 256)
			printf("## 255 over : %f\n", h);
		if (sumUpToIndex_window.index <= -1)
			printf("## 0 under : %f\n", h);
			*/
		//if(histo_window[sumUpToIndex_window.index].f)//この判定を無くした方が速いかもしれない
		{
			//あるなら更新してhチェック
			sumUpToIndex_window.f += histo_window[sumUpToIndex_window.index].f * sign;
			sumUpToIndex_window.g += histo_window[sumUpToIndex_window.index].g * sign;
			h = cx * sumUpToIndex_window.g + dx * sumUpToIndex_window.f;
			if ((h >= half) == flagA)
			{
				//超えたのでこのindexがmedian
				result_center = sumUpToIndex_window.index;
				sumUpToIndex_window.index += flag2;
				break;
			}
		}
		sumUpToIndex_window.index += flag2;
		/*
		if (sumUpToIndex_window.index >= 256)
			printf("## 255 over : %f\n", h);
		if (sumUpToIndex_window.index <= -1)
			printf("## 0 under : %f\n", h);
			*/
	}
}
inline void findMedian(const gSum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, const float& half, fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, int& result_center) {
	float cx;
	float dx;
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, sumUpToIndex_window, result_center);
}
inline void findMedian(float& cx, float& dx, const gSum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, const float& half, fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, int& result_center) {
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, sumUpToIndex_window, result_center);
}

//sumuptoindex無しテスト
inline void findMedian(const float& cx, const float& dx, const float& half, fg* histo_window, int& result_center) {
	float h;
	int sumUpToIndex_f = 0;
	int sumUpToIndex_g = 0;
	int index_window = 0;


	while (true)
	{
		h = cx * sumUpToIndex_g + dx * sumUpToIndex_f;
		//if(histo_window[index_window].f)//この判定を無くした方が速いかもしれない
		{
			//あるなら更新してhチェック
			sumUpToIndex_f += histo_window[index_window].f;
			sumUpToIndex_g += histo_window[index_window].g;
			h = cx * sumUpToIndex_g + dx * sumUpToIndex_f;
			if (h >= half)
			{
				//超えたのでこのindexがmedian
				result_center = index_window;
				break;
			}
		}
		index_window++;
	}
}
inline void findMedian(const gSum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, const float& half, fg* histo_window, int& result_center) {
	float cx;
	float dx;
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, result_center);
}
inline void findMedian(float& cx, float& dx, const gSum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, const float& half, fg* histo_window, int& result_center) {
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, result_center);
}


//fg3 vec
inline void findMedian(const cv::Vec3f& cx, const float& dx, const float& half, fg3* histo_window, fg3SumUpToIndex& sumUpToIndex_window, int& result_center) {
	float h = cx[0] * sumUpToIndex_window.g1 + cx[1] * sumUpToIndex_window.g2 + cx[2] * sumUpToIndex_window.g3 + dx * sumUpToIndex_window.f;

	const int flagA = h < half;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;

	while (true)
	{
		sumUpToIndex_window.index += flagA;
		if (histo_window[sumUpToIndex_window.index].f)
		{
			//あるなら更新してhチェック
			sumUpToIndex_window.f += histo_window[sumUpToIndex_window.index].f * sign;
			sumUpToIndex_window.g1 += histo_window[sumUpToIndex_window.index].g1 * sign;
			sumUpToIndex_window.g2 += histo_window[sumUpToIndex_window.index].g2 * sign;
			sumUpToIndex_window.g3 += histo_window[sumUpToIndex_window.index].g3 * sign;
			h = cx[0] * sumUpToIndex_window.g1 + cx[1] * sumUpToIndex_window.g2 + cx[2] * sumUpToIndex_window.g3 + dx * sumUpToIndex_window.f;
			/*
			if (h < -0.001 || h > 1.001)
			{
				printf("%f ", h);
			}
			*/

			if ((h >= half) == flagA)
			{
				//超えたのでこのindexがmedian
				result_center = sumUpToIndex_window.index;
				sumUpToIndex_window.index += flag2;
				break;
			}
		}

		sumUpToIndex_window.index += flag2;
	}
}
//fgx
template<int n>
inline void findMedian(const cv::Vec<float, n>& cx, const float& dx, const float& half, fgx<n>* histo_window, fgxSumUpToIndex<n>& sumUpToIndex_window, int& result_center) {
	float h = dx * sumUpToIndex_window.f;
	for (int i = 0; i < n; i++)
	{
		h += cx[i] * sumUpToIndex_window.g[i];
	}

	const int flagA = h < half;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;

	while (true)
	{
		sumUpToIndex_window.index += flagA;
		if (histo_window[sumUpToIndex_window.index].f)
		{
			//あるなら更新してhチェック
			sumUpToIndex_window.f += histo_window[sumUpToIndex_window.index].f * sign;
			for (int i = 0; i < n; i++)
			{
				sumUpToIndex_window.g[i] += histo_window[sumUpToIndex_window.index].g[i] * sign;
			}
			h = dx * sumUpToIndex_window.f;
			for (int i = 0; i < n; i++)
			{
				h += cx[i] * sumUpToIndex_window.g[i];
			}

			if ((h >= half) == flagA)
			{
				//超えたのでこのindexがmedian
				result_center = sumUpToIndex_window.index;
				sumUpToIndex_window.index += flag2;
				break;
			}
		}

		sumUpToIndex_window.index += flag2;
	}
}

inline void findMedian(const g3Sum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const cv::Vec3i& G_center, const float& half, fg3* histo_window, fg3SumUpToIndex& sumUpToIndex_window, int& result_center) {
	cv::Vec3f cx;
	float dx;
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, sumUpToIndex_window, result_center);
}
inline void findMedian(cv::Vec3f*& cx, float*& dx, const g3Sum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, const float& half, fg3* histo_window, fg3SumUpToIndex& sumUpToIndex_window, int& result_center) {
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, *cx, *dx);
	findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, result_center);
}

//list
//fg
inline void findMedian(const float& cx, const float& dx, const float& half, std::list<fg_node>& histo_window, fgSumUpToIndex& sumUpToIndex_window, int& result_center) {
	float h = cx * sumUpToIndex_window.g + dx * sumUpToIndex_window.f;

	//この追跡版はバグあり。リストの要素数が1のときにうまく動作しない


	auto itr_window = histo_window.begin();
	//auto end_window = histo_window.end();
	while (sumUpToIndex_window.index > (*itr_window).index)
	{
		itr_window++;
		if (itr_window == histo_window.end())
		{
			itr_window--;
			break;
		}
	}

	const int flagA = h < half;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	//listはiteratorに対する数値演算はできず++,--は使えるため、以下のように分離して実装している
	if (flagA)
	{
		while (true)
		{
			//sumUpToIndex_window.index += flagA;
			itr_window++;
			sumUpToIndex_window.f += (*itr_window).f;
			sumUpToIndex_window.g += (*itr_window).g;
			h = cx * sumUpToIndex_window.g + dx * sumUpToIndex_window.f;
			if (h >= half)
			{
				//超えたのでこのindexがmedian
				sumUpToIndex_window.index = (*itr_window).index;
				result_center = sumUpToIndex_window.index;
				break;
			}
		}
	}
	else
	{
		while (true)
		{
			//降順に探索するときにこの分岐に引っかかる場合は、リストの開始indexにいるのにまだhalfを下回っていない場合。この場合はこのindexが中央値となる。これ以上リストは遡れないので別処理にする。
			if (itr_window == histo_window.begin())
			{
				sumUpToIndex_window.index = (*itr_window).index;
				result_center = (*itr_window).index;
				break;
			}

			sumUpToIndex_window.f -= (*itr_window).f;
			sumUpToIndex_window.g -= (*itr_window).g;
			itr_window--;
			h = cx * sumUpToIndex_window.g + dx * sumUpToIndex_window.f;
			if (h < half)
			{
				//下回ったので、1つ上のindexがmedian
				//sumuptoは更新しているので、下回ったものを登録
				sumUpToIndex_window.index = (*itr_window).index;
				itr_window++;
				result_center = (*itr_window).index;
				break;
			}
		}

	}
}
inline void findMedian(const gSum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, const float& half, std::list<fg_node>& histo_window, fgSumUpToIndex& sumUpToIndex_window, int& result_center) {
	float cx;
	float dx;
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, sumUpToIndex_window, result_center);
}
inline void findMedian(float& cx, float& dx, const gSum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, const float& half, std::list<fg_node>& histo_window, fgSumUpToIndex& sumUpToIndex_window, int& result_center) {
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, sumUpToIndex_window, result_center);
}
//fgSumUpToIndexなし
//g1
inline void findMedian(const float& cx, const float& dx, const float& half, std::list<fg_node>& histo_window, int& result_center) {
	//リストはじめからh計算
	auto itr_window = histo_window.begin();
	auto end_window = histo_window.end();
	float h;
	int sumUpToIndex_f = 0;
	int sumUpToIndex_g = 0;
	while (itr_window != end_window)
	{
		sumUpToIndex_f += (*itr_window).f;
		sumUpToIndex_g += (*itr_window).g;
		h = cx * sumUpToIndex_g + dx * sumUpToIndex_f;
		if (h >= half)
		{
			//超えたのでこのindexがmedian
			result_center = (*itr_window).index;
			break;
		}
		itr_window++;
	}

}
//g3
inline void findMedian(const cv::Vec3f& cx, const float& dx, const float& half, std::list<fg3_node>& histo_window, int& result_center) {
	//リストはじめからh計算
	auto itr_window = histo_window.begin();
	auto end_window = histo_window.end();
	float h;
	int sumUpToIndex_f = 0;
	cv::Vec3i sumUpToIndex_g(0,0,0);
	while (itr_window != end_window)
	{
		sumUpToIndex_f += (*itr_window).f;
		sumUpToIndex_g += (*itr_window).g;
		h = cx[0] * sumUpToIndex_g[0] + cx[1] * sumUpToIndex_g[1] + cx[2] * sumUpToIndex_g[2] + dx * sumUpToIndex_f;
		if (h >= half)
		{
			//超えたのでこのindexがmedian
			result_center = (*itr_window).index;
			break;
		}
		itr_window++;
	}

}
//g1
inline void findMedian(const gSum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const int& G_center, const float& half, std::list<fg_node>& histo_window, int& result_center) {
	float cx;
	float dx;
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, result_center);
}
//g3
inline void findMedian(const g3Sum& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const cv::Vec3i& G_center, const float& half, std::list<fg3_node>& histo_window, int& result_center) {
	cv::Vec3f cx;
	float dx;
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, result_center);
}
//cx dx記録用
template<typename FG_NODE, typename GSUM, typename GTYPE, typename CTYPE>
inline void findMedian(CTYPE& cx, float& dx, const GSUM& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const GTYPE& G_center, const float& half, std::list<FG_NODE>& histo_window, int& result_center) {
	calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, G_center, cx, dx);
	findMedian(cx, dx, half, histo_window, result_center);
}

inline void debugging(int x, int y, gSum& gSum_window, int pixel_sum_window, fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, int& result_center, float cx, float dx) {
}

inline void debugging(int x, int y, g3Sum& gSum_window, int pixel_sum_window, fg3* histo_window, fg3SumUpToIndex& sumUpToIndex_window, int& result_center, cv::Vec3f cx, float dx) {
	std::ofstream file("original.txt", std::ios::app);

	file << "(" << x << "," << y << ") : ";
	file << "W.k " << sumUpToIndex_window.index;
	file << " calc(c,d)=(" << cx << "," << dx << ") ";
	file << " g_sum=" << cv::Vec3i(gSum_window.g1, gSum_window.g2, gSum_window.g3);
	file << " gg_sum=";
	file << gSum_window.g1g1;
	file << gSum_window.g1g2;
	file << gSum_window.g1g3;
	file << gSum_window.g2g2;
	file << gSum_window.g2g3;
	file << gSum_window.g3g3;

	file << std::endl;

	file.close();

}