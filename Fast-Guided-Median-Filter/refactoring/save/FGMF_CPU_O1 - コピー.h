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
#include <vector>
#include "common.h"
#include <malloc.h>
#include <cstring>
#include <algorithm>

/*
template<class T>
using unique_ptr_aligned = std::unique_ptr<T, decltype(&aligned_free)>;

template<class T>
unique_ptr_aligned<T> aligned_uptr(size_t align, size_t size)
{
	return unique_ptr_aligned<T>(
		static_cast<T*>(aligned_malloc(align, size)),
		&aligned_free);
}
*/
/*
template<class T>
std::unique_ptr<T, void(*)(void*)> aligned_uptr(size_t align, size_t size)
{
	return std::unique_ptr<T, void(*)(void*)>(
		static_cast<T*>(aligned_malloc(align, size)),
		&aligned_free);
}
*/

namespace FGMF_CPU_O1
{
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int threadNum, int radius, float eps2, int num_level);


	template <typename G_TYPE>
	class DataForC_D
	{
	};

	template<>
	class DataForC_D<int>
	{
	public:
		DataForC_D() : g_sum_(0), gg_sum_(0), numPixels_(0) {}

		int numPixels_;
		int g_sum_;
		int gg_sum_;

		void addG(const int* g)
		{
			numPixels_++;
			g_sum_ += *g;
			gg_sum_ += (*g) * (*g);
		}
		void removeG(const int* g)
		{
			numPixels_--;
			g_sum_ -= *g;
			gg_sum_ -= (*g) * (*g);
		}
		void mergeG(DataForC_D<int>& dataForC_D)
		{
			numPixels_ += dataForC_D.numPixels_;
			g_sum_ += dataForC_D.g_sum_;
			gg_sum_ += dataForC_D.gg_sum_;
		}
		void separateG(DataForC_D<int>& dataForC_D)
		{
			numPixels_ -= dataForC_D.numPixels_;
			g_sum_ -= dataForC_D.g_sum_;
			gg_sum_ -= dataForC_D.gg_sum_;
		}
	};

	template<>
	class DataForC_D<cv::Vec3i>
	{
	public:
		DataForC_D() : g_sum_(0, 0, 0), numPixels_(0)
		{
			gg_sum_.fill(0);
		}
		int numPixels_;
		cv::Vec3i g_sum_;
		std::array<int, 6> gg_sum_;
		//int gg_sum[6];
		void addG(const cv::Vec3i* g)
		{
			numPixels_++;
			g_sum_ += *g;
			gg_sum_[0] += (*g)[0] * (*g)[0];
			gg_sum_[1] += (*g)[0] * (*g)[1];
			gg_sum_[2] += (*g)[0] * (*g)[2];
			gg_sum_[3] += (*g)[1] * (*g)[1];
			gg_sum_[4] += (*g)[1] * (*g)[2];
			gg_sum_[5] += (*g)[2] * (*g)[2];
		}
		void removeG(const cv::Vec3i* g)
		{
			numPixels_--;
			g_sum_ -= *g;
			gg_sum_[0] -= (*g)[0] * (*g)[0];
			gg_sum_[1] -= (*g)[0] * (*g)[1];
			gg_sum_[2] -= (*g)[0] * (*g)[2];
			gg_sum_[3] -= (*g)[1] * (*g)[1];
			gg_sum_[4] -= (*g)[1] * (*g)[2];
			gg_sum_[5] -= (*g)[2] * (*g)[2];
		}
		void mergeG(DataForC_D<cv::Vec3i>& dataForC_D)
		{
			numPixels_ += dataForC_D.numPixels_;
			g_sum_ += dataForC_D.g_sum_;
			for (int i = 0; i < 6; i++)
				gg_sum_[i] += dataForC_D.gg_sum_[i];
		}
		void separateG(DataForC_D<cv::Vec3i>& dataForC_D)
		{
			numPixels_ -= dataForC_D.numPixels_;
			g_sum_ -= dataForC_D.g_sum_;
			for (int i = 0; i < 6; i++)
				gg_sum_[i] -= dataForC_D.gg_sum_[i];
		}
	};

	template <typename T, std::size_t Alignment>
	std::unique_ptr<T[], decltype(&_aligned_free)> aligned_unique_ptr(std::size_t n) {
		return std::unique_ptr<T[], decltype(&_aligned_free)>{
			static_cast<T*>(_aligned_malloc(n * sizeof(T), Alignment)),
				& _aligned_free
		};
	}

	template <typename G_TYPE>
	class Window
	{
	public:
		Window(const int num_level)
			: F_{ aligned_unique_ptr<int, MEMORY_ALIGNMENT>(num_level) },
			G_{ aligned_unique_ptr<G_TYPE, MEMORY_ALIGNMENT>(num_level) },
			f_cum_{ 0 },
			g_cum_(0),
			k_{ -1 }
		{
			std::memset(F_.get(), 0, num_level * sizeof(int));
			std::memset(G_.get(), 0, num_level * sizeof(G_TYPE));
		}
		std::unique_ptr<int[], decltype(&_aligned_free)> F_;
		std::unique_ptr<G_TYPE[], decltype(&_aligned_free)> G_;
		int f_cum_;
		G_TYPE g_cum_;
		int k_;

		void addG(const G_TYPE* g) {};
		void removeG(const G_TYPE* g) {};
		void mergeG(Window<G_TYPE>& W) {};
		void separateG(Window<G_TYPE>& W) {};
	};

	template <typename G_TYPE>
	class Window_calc_cd : public Window<G_TYPE>
	{
	public:
		Window_calc_cd<G_TYPE>(const int num_level)
			: Window<G_TYPE>(num_level)
		{
		};
		DataForC_D<G_TYPE> dataForC_D_;
		void addG(const G_TYPE* g);
		void removeG(const G_TYPE* g);
		void mergeG(Window_calc_cd<G_TYPE>& W);
		void separateG(Window_calc_cd<G_TYPE>& W);
	};




	class WMF
	{
	public:
		WMF(cv::Mat& f_img, cv::Mat& g_img, int threadNum, int radius, float eps2, int Imax)
			: f_img_(f_img), g_img_(g_img), threadNum_(threadNum), radius_(radius), eps2_(eps2), Imax_(Imax) , M_(f_img.rows), N_(f_img.cols), p_(radius), m_(radius + 1), channelNum_f_(f_img_.channels()), channelNum_g_(g_img_.channels())
		{
			//check validation
			assert(f_img_.depth() == CV_32S || f_img_.depth() == CV_8U);
			assert(g_img_.depth() == CV_32S || g_img_.depth() == CV_8U);
			if (f_img_.depth() != CV_32S)
				f_img_.convertTo(f_img_, CV_32S);
			if (g_img_.depth() != CV_32S)
				g_img_.convertTo(g_img_, CV_32S);
#if defined(USE_AVX2)
			assert(Imax % 8 == 0);
#endif


			if (channelNum_g_ == 1)
				c_ = cv::Mat(M_, N_, CV_32FC1);
			else
				c_ = cv::Mat(M_, N_, CV_32FC3);

			d_ = cv::Mat(M_, N_, CV_32FC1);
		}

		cv::Mat apply_2d_filter();

		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		void filtering(const cv::Mat& f_single, const cv::Mat& g_multi, cv::Mat& result);
		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		void filteringBlock(const cv::Mat& f_single, const cv::Mat& g_multi, cv::Mat& result, const int tStart_target, const int tEnd_target);
		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		void filteringBlockHorizontal(const cv::Mat& f_single, const cv::Mat& g_multi, cv::Mat& result, const int tStart_target, const int tEnd_target);

		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		cv::Mat filtering_save(const cv::Mat& f_single, const cv::Mat& g_multi);

		static void test();

	private:
		const cv::Mat f_img_;
		const cv::Mat g_img_;
		const int threadNum_;
		const int radius_;
		const float eps2_;
		const int Imax_;
		cv::Mat c_;
		cv::Mat d_;

		const int M_; // Image height
		const int N_; // Image width
		const int channelNum_f_; // Input image channel num
		const int channelNum_g_; // Guide image channel num
		const int p_; // x^+ = x + r = x + p_
		const int m_; // x^- = x - r - 1 = x + m_


		template<typename WINDOW, typename G_TYPE>
		void addPixelToWindow(const int* f, const G_TYPE* g, WINDOW& W);

		template<typename WINDOW, typename G_TYPE>
		void removePixelFromWindow(const int* f, const G_TYPE* g, WINDOW& W);

		template<typename WINDOW, typename G_TYPE>
		void mergeWindow(WINDOW& W1, WINDOW& W2);

		template<typename WINDOW, typename G_TYPE>
		void separateWindow(WINDOW& W1, WINDOW& W2);
		
		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		int searchWeightedMedian(const G_TYPE* g, WINDOW& W, C_TYPE* cx, float* dx);

		template<typename G_TYPE, typename C_TYPE>
		float calculateHcum(int& f_cum, G_TYPE& g_cum, C_TYPE* c, float* d);
		
		template<typename WINDOW, typename C_TYPE, typename G_TYPE>
		void calculateCandD(const G_TYPE* g, WINDOW& W, C_TYPE* c, float* d);


		template<typename WINDOW, typename C_TYPE, typename G_TYPE>
		void debugging(int s, int t, WINDOW& W, C_TYPE* c, float* d, const cv::Mat& f, const cv::Mat& g);
	};


}