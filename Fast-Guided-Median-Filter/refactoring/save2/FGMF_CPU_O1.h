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


namespace FGMF_CPU_O1
{
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange, int threadNum = omp_get_num_procs());


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

		void addG(const int* g_x)
		{
			numPixels_++;
			g_sum_ += *g_x;
			gg_sum_ += (*g_x) * (*g_x);
		}
		void removeG(const int* g_x)
		{
			numPixels_--;
			g_sum_ -= *g_x;
			gg_sum_ -= (*g_x) * (*g_x);
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
		void addG(const cv::Vec3i* g_x)
		{
			numPixels_++;
			g_sum_ += *g_x;
			gg_sum_[0] += (*g_x)[0] * (*g_x)[0];
			gg_sum_[1] += (*g_x)[0] * (*g_x)[1];
			gg_sum_[2] += (*g_x)[0] * (*g_x)[2];
			gg_sum_[3] += (*g_x)[1] * (*g_x)[1];
			gg_sum_[4] += (*g_x)[1] * (*g_x)[2];
			gg_sum_[5] += (*g_x)[2] * (*g_x)[2];
		}
		void removeG(const cv::Vec3i* g_x)
		{
			numPixels_--;
			g_sum_ -= *g_x;
			gg_sum_[0] -= (*g_x)[0] * (*g_x)[0];
			gg_sum_[1] -= (*g_x)[0] * (*g_x)[1];
			gg_sum_[2] -= (*g_x)[0] * (*g_x)[2];
			gg_sum_[3] -= (*g_x)[1] * (*g_x)[1];
			gg_sum_[4] -= (*g_x)[1] * (*g_x)[2];
			gg_sum_[5] -= (*g_x)[2] * (*g_x)[2];
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
		Window(const int fRange)
			: f_cum_{ 0 },
			g_cum_(0),
			k_{ -1 }
		{
			storeFG_ = (int*)_aligned_malloc(fRange * (sizeof(int) + sizeof(G_TYPE)), MEMORY_ALIGNMENT);
			memset(storeFG_, 0, fRange * (sizeof(int) + sizeof(G_TYPE)));
			F_ = storeFG_;
			G_ = reinterpret_cast<G_TYPE*>(&storeFG_[fRange]);
		}
		int* storeFG_;
		int* F_;
		G_TYPE* G_;
		int f_cum_;
		G_TYPE g_cum_;
		int k_;

		void addG(const G_TYPE* g_x) {};
		void removeG(const G_TYPE* g_x) {};
		void mergeG(Window<G_TYPE>& W) {};
		void separateG(Window<G_TYPE>& W) {};

		~Window()
		{
			_aligned_free(storeFG_);
		}
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
		void addG(const G_TYPE* g_x);
		void removeG(const G_TYPE* g_x);
		void mergeG(Window_calc_cd<G_TYPE>& W);
		void separateG(Window_calc_cd<G_TYPE>& W);

		~Window_calc_cd()
		{
		}
	};




	class WMF
	{
	public:
		WMF(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange, int threadNum)
			: f_img_(f_img), g_img_(g_img), threadNum_(threadNum), radius_(radius), eps2_(eps2), fRange_(fRange) , M_(f_img.rows), N_(f_img.cols), p_(radius), m_(radius + 1), channelNum_f_(f_img_.channels()), channelNum_g_(g_img_.channels())
		{
			//check validation
			assert(f_img_.depth() == CV_32S || f_img_.depth() == CV_8U);
			assert(g_img_.depth() == CV_32S || g_img_.depth() == CV_8U);
			if (f_img_.depth() != CV_32S)
				f_img_.convertTo(f_img_, CV_32S);
			if (g_img_.depth() != CV_32S)
				g_img_.convertTo(g_img_, CV_32S);
#if defined(USE_AVX2)
			assert(fRange % 8 == 0);
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


		static void test();

	private:
		const cv::Mat f_img_;
		const cv::Mat g_img_;
		const int threadNum_;
		const int radius_;
		const float eps2_;
		const int fRange_;
		cv::Mat c_;
		cv::Mat d_;

		const int M_; // Image height
		const int N_; // Image width
		const int channelNum_f_; // Input image channel num
		const int channelNum_g_; // Guide image channel num
		const int p_; // x^+ = x + r = x + p_
		const int m_; // x^- = x - r - 1 = x + m_


		template<typename WINDOW, typename G_TYPE>
		void addPixelToWindow(const int* f_x, const G_TYPE* g_x, WINDOW& W);

		template<typename WINDOW, typename G_TYPE>
		void removePixelFromWindow(const int* f_x, const G_TYPE* g_x, WINDOW& W);

		template<typename WINDOW, typename G_TYPE>
		void mergeWindow(WINDOW& W1, WINDOW& W2);

		template<typename WINDOW, typename G_TYPE>
		void separateWindow(WINDOW& W1, WINDOW& W2);
		
		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		int searchWeightedMedian(const G_TYPE& g_x, WINDOW& W, C_TYPE& c_x, float& d_x);

		template<typename G_TYPE, typename C_TYPE>
		float calculateHcum(const int& f_cum, const G_TYPE& g_cum, const C_TYPE& c_x, const float& d_x);
		
		template<typename WINDOW, typename C_TYPE, typename G_TYPE>
		void calculateCandD(const G_TYPE& g_x, WINDOW& W, C_TYPE& c_x, float& d_x);


		template<typename WINDOW, typename C_TYPE, typename G_TYPE>
		void debugging(int s, int t, WINDOW& W, C_TYPE* c, float* d, const cv::Mat& f, const cv::Mat& g);
	};


}