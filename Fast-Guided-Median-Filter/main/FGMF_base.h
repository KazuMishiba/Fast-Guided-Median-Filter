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
	//AVX/AVX2���g���ꍇ�AImax��4�̔{���ɂ���
	if (Imax % 4 != 0)
		Imax += (4 - (Imax % 4));
#elif defined(USE_AVX512)
	if (Imax % 8 != 0)
		Imax += (8 - (Imax % 8));
#else
#endif
}


void inline calculatePosForMultithread2D(const int& width, const int& colNum, const int& r_dim0, std::vector<int>& dim0Start, std::vector<int>& dim0End, std::vector<int>& memoryLength, std::vector<int>& insideImageStart) {
	//���̌���
	//col
	int baseWidth = width / colNum;
	int remainder = width % colNum;
	//���W�A���v�Z�Ώۉ�f�J�n�ʒu�iisInside_image�v�Z�p�j
	insideImageStart[0] = 0;
	//�����J�n�ʒu
	dim0Start[0] = -r_dim0;
	//�����I���ʒu
	dim0End[0] = insideImageStart[0] + baseWidth + (remainder > 0) - 1;
	remainder--;
	//������dim0��������
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




//window����f���̍X�V�i�ǉ��j
inline void addPixelSum(const int& addedPixelNum, int& pixel_sum_window, float& pixel_sum_window_inv) {
	pixel_sum_window += addedPixelNum;
	pixel_sum_window_inv = 1.0f / pixel_sum_window;
}

//window����f���̍X�V�i�폜�j
inline void subtractPixelSum(const int& subtractedPixelNum, int& pixel_sum_window, float& pixel_sum_window_inv) {
	pixel_sum_window -= subtractedPixelNum;
	pixel_sum_window_inv = 1.0f / pixel_sum_window;
}
//window����f���̍X�V
inline void updatePixelSum(const int& hasAdd, const int& hasRem, const int& pixel_sum_dim, int& pixel_sum_window, float& pixel_sum_window_inv) {
	if (hasAdd && !hasRem)
		addPixelSum(pixel_sum_dim, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
	else if (!hasAdd && hasRem)
		subtractPixelSum(pixel_sum_dim, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�

}
//1�񂠂���̉�f���v�Z
inline int calculatePixelSum_Column(const int& hasAdd, const int& hasRem, const int& x, const int& r, const int& size) {
	if (!hasRem && hasAdd)
		return x + r + 1;
	else if (hasRem && !hasAdd)
		return size - x + r;
	else
		return 2 * r + 1;
}




///////////////////////////
//����1�`�����l���E�K�C�h1�`�����l��
inline void addHistogram(const int& Imax, fg* histo_window, fg* histo_subwin) {
#if defined(USE_AVX2)
	//�T�C�Y(byte) Imax * sizeof(fg) �J��Ԃ��� Imax * sizeof(fg) / 256
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
	//�T�C�Y(byte) Imax * sizeof(fg) �J��Ԃ��� Imax * sizeof(fg) / 256
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
	//�T�C�Y(byte) Imax * sizeof(fg) �J��Ԃ��� Imax * sizeof(fg) / 256
	// i+= 256 / sizeof(fg) = 4
	//mm256�ł�int�^(32bit)��1�x��4�����ł���
	//�������ׂ�int�^�̐���Imax*(1+n)��
	//�Ƃ������Ƃ͌J��Ԃ��񐔂� Imax*(1*n)/4�@��
	//Imax��4�Ŋ���邱�Ƃ��O��
	//i�̃C���N�������g�����߂�����́An���ςȂ̂Ŗ���
	//�����i��1�����₷�Ƃ��āA
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
	//subwin�̃q�X�g�O������window�̃q�X�g�O�����Ƀ}�[�W
	auto itr_window = histo_window.begin();
	auto itr_subwin = histo_subwin.begin();
	auto end_window = histo_window.end();
	auto end_subwin = histo_subwin.end();

	while (itr_subwin != end_subwin) {
		//window�̗v�f��S�Ă��ǂ�I���Ă�����I������
		if (itr_window == end_window)
		{
			//subwin�̗v�f���c���Ă���Ȃ炷�ׂĒǉ�
			while (itr_subwin != end_subwin) {
				histo_window.push_back((*itr_subwin).copy());
				itr_subwin++;
			}
			return;
		}

		//subwin����1�v�f�����o���Aitr_window��index�Ɣ�r
		if ((*itr_subwin).index < (*itr_window).index)
		{
			//��O�ɑ}��
			histo_window.insert(itr_window, (*itr_subwin).copy());
			itr_subwin++;
		}
		else if ((*itr_subwin).index == (*itr_window).index)
		{
			//�����Ȃ瓝��
			(*itr_window).add((*itr_subwin));
			itr_subwin++;
		}
		else
		{
			//�����łȂ��Ȃ�window�̎��̗v�f�Ɉڂ�
			itr_window++;
		}
	}
}

template<typename FG_NODE>
inline void remHistogram(const int& Imax, std::list<FG_NODE>& histo_window, std::list<FG_NODE>& histo_subwin) {
	//window�̃q�X�g�O��������subwin�̃q�X�g�O�������폜
	auto itr_window = histo_window.begin();
	auto itr_subwin = histo_subwin.begin();
	auto end_window = histo_window.end();
	auto end_subwin = histo_subwin.end();

	while (itr_subwin != end_subwin) {
		//subwin����1�v�f�����o���Aitr_window��index�Ɣ�r
		if ((*itr_subwin).index == (*itr_window).index)
		{
			//�����Ȃ�����A����������0�ɂȂ�悤�Ȃ�폜
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
* ��ԒႢ�����̃E�B���h�E�X�V�͗v�f�P�̍X�V�Ȃ̂ŁA�q�X�g�O�����Ȃǂ̏��X�V���ɁA
* 0~Imax�̒T���������肵�Ȃ��Ē��ڌv�Z�ł���i���ڌv�Z�����ق����R�X�g���Ⴂ�j
*/
///////////////////////////
//addPixel
//fg
inline void addPixelToWindow(fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel) {
	histo_window[index_pixel].f += 1;
	histo_window[index_pixel].g += g_pixel;
	//�ǉ��v�f��f������index�ȉ����ǂ����𔻒肵�A�ȉ��Ȃ�ǉ����邽�߂ɑ����A���傫���Ȃ�sum�Ɋ܂܂�Ȃ��̂ŉ������Ȃ�
	if (index_pixel <= sumUpToIndex_window.index) {
		sumUpToIndex_window.f += 1;
		sumUpToIndex_window.g += g_pixel;
	}
}
inline void addPixelToWindow_gSum(fg* histo_window, fgSumUpToIndex& sumUpToIndex_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	addPixelToWindow(histo_window, sumUpToIndex_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_col�ɉ��Z
	gSum_window.g += g_pixel;
	gSum_window.gg += g_pixel * g_pixel;
}
//sumuptoindex�����e�X�g
inline void addPixelToWindow(fg* histo_window, const int& index_pixel, const int& g_pixel) {
	histo_window[index_pixel].f += 1;
	histo_window[index_pixel].g += g_pixel;
}
inline void addPixelToWindow_gSum(fg* histo_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	addPixelToWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_col�ɉ��Z
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
window��list�������Ă��āAwindow.histo[pos]����pos�̃q�X�g�O�������X�g
����Ɋւ��āA���X�g��index�v�f�����ɂ݂Ă���
*pixel_f���傫���ŏ��̗v�f�����������ꍇ�́A���̈�O�ɂ���index(*pixel_f), f=1�Ag=*pixel_g�@��o�^
*pixel_f�Ɠ������̂����������ꍇ�́A*pixel_f��1�ǉ��A�܂�������g��*pixel_g�ǉ�
�܂��A���X�g�̍Ō�܂œ��B�����ꍇ�͍Ō�ɓo�^
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
	//g_sum_col, gg_sum_col�ɉ��Z
	gSum_window.g += g_pixel;
	gSum_window.gg += g_pixel * g_pixel;
}
//fgSumUpToIndex �Ȃ�
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
	//g_sum_col, gg_sum_col�ɉ��Z
	gSum_window.g += g_pixel;
	gSum_window.gg += g_pixel * g_pixel;
}
//g3
inline void addPixelToWindow_gSum(std::list<fg3_node>& histo_window, const int& index_pixel, const cv::Vec3i& g_pixel, g3Sum& gSum_window) {
	addPixelToWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_col�ɉ��Z
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
	//�O�̍s��g��f�擾
	//�q�X�g�O�������猸�Z
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
	//g_sum_col, gg_sum_col���猸�Z
	gSum_window.g -= g_pixel;
	gSum_window.gg -= g_pixel * g_pixel;
}
//sumuptoindex�����e�X�g
inline void removePixelFromWindow(fg* histo_window, const int& index_pixel, const int& g_pixel) {
	//�O�̍s��g��f�擾
	//�q�X�g�O�������猸�Z
	histo_window[index_pixel].f -= 1;
	histo_window[index_pixel].g -= g_pixel;
}
inline void removePixelFromWindow_gSum(fg* histo_window, const int& index_pixel, const int& g_pixel, gSum& gSum_window) {
	removePixelFromWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_col���猸�Z
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
	//�폜�Ώۂ͕K�����X�g���ɑ��݂���
	for (auto itr = histo_window.begin(); itr != histo_window.end(); ++itr) {
		if ((*itr).index == index_pixel)
		{
			if ((*itr).f == 1)
			{
				//���X�g����폜
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
	//g_sum_col, gg_sum_col���猸�Z
	gSum_window.g -= g_pixel;
	gSum_window.gg -= g_pixel * g_pixel;
}
//fgSumUpToIndex�Ȃ�
template<typename FG_NODE, typename GTYPE>
inline void removePixelFromWindow(std::list<FG_NODE>& histo_window, const int& index_pixel, const GTYPE& g_pixel) {
	//�폜�Ώۂ͕K�����X�g���ɑ��݂���
	for (auto itr = histo_window.begin(); itr != histo_window.end(); ++itr) {
		if ((*itr).index == index_pixel)
		{
			if ((*itr).f == 1)
			{
				//���X�g����폜
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
	//g_sum_col, gg_sum_col���猸�Z
	gSum_window.g -= g_pixel;
	gSum_window.gg -= g_pixel * g_pixel;
}
//g3
inline void removePixelFromWindow_gSum(std::list<fg3_node>& histo_window, const int& index_pixel, const cv::Vec3i& g_pixel, g3Sum& gSum_window) {
	removePixelFromWindow(histo_window, index_pixel, g_pixel);
	//g_sum_col, gg_sum_col�ɉ��Z
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
	//�����ł͒ǉ��\��̃E�B���h�E��under_i�v�f���X�V����
	//����sign�������������ɋt���ɃA�N�Z�X���邩��L���b�V�������������̂ł�
	//���ɂȂ�悤�ɕύX���������ǂ�����
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		//�ǉ�
		sumUpToIndex_subwin.f += histo_subwin[i].f * sign;
		sumUpToIndex_subwin.g += histo_subwin[i].g * sign;
	}
	//index�̍X�V
	sumUpToIndex_subwin.index = index_window;
	//�X�V�����E�B���h�E��ǉ����邱�ƂŃE�B���h�E�̍X�V������
	//�E�B���h�E��f_sum, g_sum�̍X�V
	sumUpToIndex_window_f += sumUpToIndex_subwin.f;
	sumUpToIndex_window_g += sumUpToIndex_subwin.g;
}
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, const int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window_f, sumUpToIndex_window_g, index_window, sumUpToIndex_subwin);
	//g, gg�̍X�V�Fsum_window�ɒǉ�������Z
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
//sumuptoindex�����e�X�g
inline void updateSubWindowAndAddToWindow(const int& Imax, fg* histo_window, fg* histo_subwin) {
	addHistogram(Imax, histo_window, histo_subwin);
}
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin);
	//g, gg�̍X�V�Fsum_window�ɒǉ�������Z
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
	//�����ł͒ǉ��\��̃E�B���h�E��under_i�v�f���X�V����
	//u_idx < index_window�̏ꍇ�Au_idx+1�`index_window�܂ł�histo_subwin�� f,g��sumUpToIndex_subwin�@�ɒǉ�����
	//u_idx > index_window�̏ꍇ�Aindex_window+1�`u_idx�܂ł�histo_subwin�� f,g��sumUpToIndex_subwin�@����폜����
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
	//index�̍X�V
	sumUpToIndex_subwin.index = index_window;
	//�X�V�����E�B���h�E��ǉ����邱�ƂŃE�B���h�E�̍X�V������
	//�E�B���h�E��f_sum, g_sum�̍X�V
	sumUpToIndex_window_f += sumUpToIndex_subwin.f;
	sumUpToIndex_window_g += sumUpToIndex_subwin.g;
}
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, const int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin, sumUpToIndex_window_f, sumUpToIndex_window_g, index_window, sumUpToIndex_subwin);
	//g, gg�̍X�V�Fsum_window�ɒǉ�������Z
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
//fgSumUpToIndex�Ȃ�
template<typename FG_NODE>
inline void updateSubWindowAndAddToWindow(const int& Imax, std::list<FG_NODE>& histo_window, std::list<FG_NODE>& histo_subwin) {
	addHistogram(Imax, histo_window, histo_subwin);
}
//g1
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, gSum& gSum_window, gSum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin);
	//g, gg�̍X�V�Fsum_window�ɒǉ�������Z
	gSum_window.g += gSum_subwin.g;
	gSum_window.gg += gSum_subwin.gg;
}
//g3
inline void updateSubWindowAndAddToWindow_gSum(const int& Imax, std::list<fg3_node>& histo_window, std::list<fg3_node>& histo_subwin, g3Sum& gSum_window, g3Sum gSum_subwin) {
	updateSubWindowAndAddToWindow(Imax, histo_window, histo_subwin);
	//g, gg�̍X�V�Fsum_window�ɒǉ�������Z
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
	//�폜���̓E�B���h�E����under_i�v�f�𒼐ڍ폜����
	//�폜����E�B���h�E�̏��́A�ǉ�����Ƃ��ɍX�V����under_i�̏��̂܂܁A
	//������̃C���f�b�N�X��index�Ȃ̂ŁA���ڍ폜�E�B���h�E�̏��������Z�����
	//����Ȃ��B�Ȃ̂ł܂����̕��̒���������
	for (int i = startIdx, j = 0; j < numIdx; i += sign, j++)
	{
		//�폜
		sumUpToIndex_window_f -= histo_subwin[i].f * sign;
		sumUpToIndex_window_g -= histo_subwin[i].g * sign;
	}
	//���ɍ폜�E�B���h�E�������炷
	//�E�B���h�E��f_sum, g_sum�̍X�V
	sumUpToIndex_window_f -= sumUpToIndex_subwin.f;
	sumUpToIndex_window_g -= sumUpToIndex_subwin.g;
}

inline void removeSubWindowFromWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, const int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow(Imax, index_window, histo_window, histo_subwin, sumUpToIndex_window_f, sumUpToIndex_window_g, sumUpToIndex_subwin);
	//g, gg�̍X�V�Fsum_window����폜������Z
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
//sumuptoindex�����e�X�g
inline void removeSubWindowFromWindow(const int& Imax, fg* histo_window, fg* histo_subwin) {
	remHistogram(Imax, histo_window, histo_subwin);
}

inline void removeSubWindowFromWindow_gSum(const int& Imax, fg* histo_window, fg* histo_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow(Imax, histo_window, histo_subwin);
	//g, gg�̍X�V�Fsum_window����폜������Z
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
	//�폜���̓E�B���h�E����under_i�v�f�𒼐ڍ폜����
	//�폜����E�B���h�E�̏��́A�ǉ�����Ƃ��ɍX�V����under_i�̏��̂܂܁A
	//������̃C���f�b�N�X��index�Ȃ̂ŁA���ڍ폜�E�B���h�E�̏��������Z�����
	//����Ȃ��B�Ȃ̂ł܂����̕��̒���������
	//u_idx < index_window�̏ꍇ�Au_idx+1�`index_window�܂ł�histo_subwin�� f,g��sumUpToIndex_subwin�@�ɒǉ�����
	//u_idx > index_window�̏ꍇ�Aindex_window+1�`u_idx�܂ł�histo_subwin�� f,g��sumUpToIndex_subwin�@����폜����
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
		//�폜
		sumUpToIndex_window_f -= histo_subwin[i].f * sign;
		sumUpToIndex_window_g -= histo_subwin[i].g * sign;
	}*/
	//���ɍ폜�E�B���h�E�������炷
	//�E�B���h�E��f_sum, g_sum�̍X�V
	sumUpToIndex_window_f -= sumUpToIndex_subwin.f;
	sumUpToIndex_window_g -= sumUpToIndex_subwin.g;
}

inline void removeSubWindowFromWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, const int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, fgSumUpToIndex& sumUpToIndex_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow(Imax, index_window, histo_window, histo_subwin, sumUpToIndex_window_f, sumUpToIndex_window_g, sumUpToIndex_subwin);
	//g, gg�̍X�V�Fsum_window����폜������Z
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
//fgSumUpToIndex�Ȃ�
template<typename FG_NODE>
inline void removeSubWindowFromWindow(const int& Imax, std::list<FG_NODE>& histo_window, std::list<FG_NODE>& histo_subwin) {
	remHistogram(Imax, histo_window, histo_subwin);
}
//g1
inline void removeSubWindowFromWindow_gSum(const int& Imax, std::list<fg_node>& histo_window, std::list<fg_node>& histo_subwin, gSum& gSum_window, gSum gSum_subwin) {
	removeSubWindowFromWindow(Imax, histo_window, histo_subwin);
	//g, gg�̍X�V�Fsum_window����폜������Z
	gSum_window.g -= gSum_subwin.g;
	gSum_window.gg -= gSum_subwin.gg;
}
inline void removeSubWindowFromWindow_gSum(const int& Imax, std::list<fg3_node>& histo_window, std::list<fg3_node>& histo_subwin, g3Sum& gSum_window, g3Sum gSum_subwin) {
	removeSubWindowFromWindow(Imax, histo_window, histo_subwin);
	//g, gg�̍X�V�Fsum_window����폜������Z
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
//g3 (�����g���Ă��Ȃ����H�m�F���č폜)
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
//gx (blas�g��)
template<int n>
inline void calculateCxDx(const gxSum<n>& gSum_window, const float& pixel_sum_window_inv, const float& eps2, const cv::Vec<int, n>& G_center, cv::Vec<float, n>& cx, float& dx) {
	/*
	�v�Z�������̂́A
	cx=(g-g_ave)^T V_inv
	�ł���A�����
	(g-g_ave)^T = cx V
	�̉��A�܂���
	(g-g_ave) = V cx^T
	�̉��ł���B
	�t�s��𒼐ڌv�Z����̂ł͂Ȃ��A���̐��`�������̉������߂邱�Ƃ�cx���v�Z����B
	���̂��߂ɁAblas��K�؂ɌĂяo����`�ɕό`���Ȃ��� g-g_ave, V���v�Z����
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
//�����l�v�Z
inline void findMedian(const float& cx, const float& dx, const float& half, fg* histo_window, int& index_window, int& sumUpToIndex_window_f, int& sumUpToIndex_window_g, int& result_center) {
	float h = cx * sumUpToIndex_window_g + dx * sumUpToIndex_window_f;

	const int flagA = h < half;
	const int flag2 = flagA - 1;
	const int sign = flagA * 2 - 1;
	while (true)
	{
		index_window += flagA;
		//if(histo_window[index_window].f)//���̔���𖳂���������������������Ȃ�
		{
			//����Ȃ�X�V����h�`�F�b�N
			sumUpToIndex_window_f += histo_window[index_window].f * sign;
			sumUpToIndex_window_g += histo_window[index_window].g * sign;
			h = cx * sumUpToIndex_window_g + dx * sumUpToIndex_window_f;
			if ((h >= half) == flagA)
			{
				//�������̂ł���index��median
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
		//if(histo_window[sumUpToIndex_window.index].f)//���̔���𖳂���������������������Ȃ�
		{
			//����Ȃ�X�V����h�`�F�b�N
			sumUpToIndex_window.f += histo_window[sumUpToIndex_window.index].f * sign;
			sumUpToIndex_window.g += histo_window[sumUpToIndex_window.index].g * sign;
			h = cx * sumUpToIndex_window.g + dx * sumUpToIndex_window.f;
			if ((h >= half) == flagA)
			{
				//�������̂ł���index��median
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

//sumuptoindex�����e�X�g
inline void findMedian(const float& cx, const float& dx, const float& half, fg* histo_window, int& result_center) {
	float h;
	int sumUpToIndex_f = 0;
	int sumUpToIndex_g = 0;
	int index_window = 0;


	while (true)
	{
		h = cx * sumUpToIndex_g + dx * sumUpToIndex_f;
		//if(histo_window[index_window].f)//���̔���𖳂���������������������Ȃ�
		{
			//����Ȃ�X�V����h�`�F�b�N
			sumUpToIndex_f += histo_window[index_window].f;
			sumUpToIndex_g += histo_window[index_window].g;
			h = cx * sumUpToIndex_g + dx * sumUpToIndex_f;
			if (h >= half)
			{
				//�������̂ł���index��median
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
			//����Ȃ�X�V����h�`�F�b�N
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
				//�������̂ł���index��median
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
			//����Ȃ�X�V����h�`�F�b�N
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
				//�������̂ł���index��median
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

	//���̒ǐՔł̓o�O����B���X�g�̗v�f����1�̂Ƃ��ɂ��܂����삵�Ȃ�


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
	//list��iterator�ɑ΂��鐔�l���Z�͂ł���++,--�͎g���邽�߁A�ȉ��̂悤�ɕ������Ď������Ă���
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
				//�������̂ł���index��median
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
			//�~���ɒT������Ƃ��ɂ��̕���Ɉ���������ꍇ�́A���X�g�̊J�nindex�ɂ���̂ɂ܂�half��������Ă��Ȃ��ꍇ�B���̏ꍇ�͂���index�������l�ƂȂ�B����ȏナ�X�g�͑k��Ȃ��̂ŕʏ����ɂ���B
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
				//��������̂ŁA1���index��median
				//sumupto�͍X�V���Ă���̂ŁA����������̂�o�^
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
//fgSumUpToIndex�Ȃ�
//g1
inline void findMedian(const float& cx, const float& dx, const float& half, std::list<fg_node>& histo_window, int& result_center) {
	//���X�g�͂��߂���h�v�Z
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
			//�������̂ł���index��median
			result_center = (*itr_window).index;
			break;
		}
		itr_window++;
	}

}
//g3
inline void findMedian(const cv::Vec3f& cx, const float& dx, const float& half, std::list<fg3_node>& histo_window, int& result_center) {
	//���X�g�͂��߂���h�v�Z
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
			//�������̂ł���index��median
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
//cx dx�L�^�p
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