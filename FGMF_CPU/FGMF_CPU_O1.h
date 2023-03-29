#pragma once
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <immintrin.h>
#include <cmath>
#include <malloc.h>


//#define USE_AVX512
#define USE_AVX2

#if defined(USE_AVX512)
#define MEMORY_ALIGNMENT 64
#else
#define MEMORY_ALIGNMENT 32
#endif



namespace FGMF_CPU_O1
{
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int fRange, int numThreads);


	template <typename G_TYPE>
	class DataForC_D
	{
	};
	// Class that handles c and d when the guide image has a single channel.
	template<>
	class DataForC_D<int>
	{
	public:
		DataForC_D() : g_sum_(0), gg_sum_(0), numPixels_(0) {}

		int numPixels_;  // Number of pixels inside the window
		int g_sum_;      // Sum of g values inside the window
		int gg_sum_;     // Sum of g^2 values inside the window (used for calculating vx)

		void addG(int* g_x)
		{
			numPixels_++;
			g_sum_ += *g_x;
			gg_sum_ += (*g_x) * (*g_x);
		}
		void removeG(int* g_x)
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

	// Class that handles c and d when the guide image has three channels (color image).
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
		void addG(cv::Vec3i* g_x)
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
		void removeG(cv::Vec3i* g_x)
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

	// Window when using calculated c and d (for second channel and beyond)
	template <typename G_TYPE>
	class Window
	{
	public:
		Window(int fRange)
			: f_cum_{ 0 },
			g_cum_(0),
			k_{ -1 }
		{
			storeFG_ = (int*)_aligned_malloc(fRange * (sizeof(int) + sizeof(G_TYPE)), MEMORY_ALIGNMENT);
			if (!storeFG_) {
				throw std::runtime_error("Failed to allocate memory for storeFG_");
			}
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

		void addG(G_TYPE* g_x) {};
		void removeG(G_TYPE* g_x) {};
		void mergeG(Window<G_TYPE>& W) {};
		void separateG(Window<G_TYPE>& W) {};

		~Window()
		{
			_aligned_free(storeFG_);
		}
	};

	// Window when c and d calculations are also performed (for first channel)
	template <typename G_TYPE>
	class Window_calc_cd : public Window<G_TYPE>
	{
	public:
		Window_calc_cd<G_TYPE>(int num_level)
			: Window<G_TYPE>(num_level)
		{
		};
		DataForC_D<G_TYPE> dataForC_D_;
		void addG(G_TYPE* g_x);
		void removeG(G_TYPE* g_x);
		void mergeG(Window_calc_cd<G_TYPE>& W);
		void separateG(Window_calc_cd<G_TYPE>& W);

		~Window_calc_cd()
		{
		}
	};




	class WMF
	{
	public:
		WMF(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int fRange, int numThreads)
			: f_img_(f_img), g_img_(g_img), numThreads_(numThreads), radius_(radius), epsilon_(epsilon), fRange_(fRange) , M_(f_img.rows), N_(f_img.cols), p_(radius), m_(radius + 1), channelNum_f_(f_img_.channels()), channelNum_g_(g_img_.channels())
		{
			// Check input image and guided image depth for valid values.
			assert(f_img_.depth() == CV_32S || f_img_.depth() == CV_8U);
			assert(g_img_.depth() == CV_32S || g_img_.depth() == CV_8U);
			if (f_img_.depth() != CV_32S)
				f_img_.convertTo(f_img_, CV_32S);
			if (g_img_.depth() != CV_32S)
				g_img_.convertTo(g_img_, CV_32S);
#if defined(USE_AVX2)
			assert(fRange % 8 == 0);
#endif

			// Initialize the c_ and d_ matrices based on the number of channels in the guided image.
			if (channelNum_g_ == 1)
				c_ = cv::Mat(M_, N_, CV_32FC1);
			else
				c_ = cv::Mat(M_, N_, CV_32FC3);

			d_ = cv::Mat(M_, N_, CV_32FC1);
		}


		// Applies the 2D CPU-O(1) filter to the input image.
		cv::Mat apply_2d_filter();

		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		void filtering(cv::Mat& f_single, cv::Mat& g_multi, cv::Mat& result);
		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		void filteringForDividedRegion(cv::Mat& f_single, cv::Mat& g_multi, cv::Mat& result, int tStart_target, int tEnd_target);


	private:
		cv::Mat f_img_;        // Input image
		cv::Mat g_img_;        // Guided image
		int numThreads_;       // Number of threads for parallel processing
		int radius_;           // Filter radius
		float epsilon_;        // Epsilon parameter for the filter
		int fRange_;           // Range of intensity values for the filter
		cv::Mat c_;            // Matrix to store c values
		cv::Mat d_;            // Matrix to store d values

		int M_;                // Image height
		int N_;                // Image width
		int originalDepth_;    // Original depth of input image (CV_8U or CV_32S)
		int channelNum_f_;     // Number of channels in the input image
		int channelNum_g_;     // Number of channels in the guided image
		int p_;                // x^+ = x + r = x + p_
		int m_;                // x^- = x - r - 1 = x + m_


		template<typename WINDOW, typename G_TYPE>
		void addPixelToWindow(int* f_x, G_TYPE* g_x, WINDOW& W);

		template<typename WINDOW, typename G_TYPE>
		void removePixelFromWindow(int* f_x, G_TYPE* g_x, WINDOW& W);

		template<typename WINDOW, typename G_TYPE>
		void mergeWindow(WINDOW& W1, WINDOW& W2);

		template<typename WINDOW, typename G_TYPE>
		void separateWindow(WINDOW& W1, WINDOW& W2);
		
		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		int searchWeightedMedian(G_TYPE& g_x, WINDOW& W, C_TYPE& c_x, float& d_x);

		float calculateHcum(int& f_cum, int& g_cum, float& c_x, float& d_x);
		float calculateHcum(int& f_cum, cv::Vec3i& g_cum, cv::Vec3f& c_x, float& d_x);
		
		template<typename WINDOW, typename C_TYPE, typename G_TYPE>
		void calculateCandD(G_TYPE& g_x, WINDOW& W, C_TYPE& c_x, float& d_x);
	};
}