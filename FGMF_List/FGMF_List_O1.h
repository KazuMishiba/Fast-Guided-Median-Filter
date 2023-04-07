#pragma once
#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <omp.h>



namespace FGMF_List_O1
{
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int numThreads);


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

	// Node for list
	template <typename G_TYPE>
	class Node {};

	template <>
	class Node<int>
	{
	public:
		int f;
		int g;
		int index;
		Node() :f(0), g(0), index(0) {};
		Node(int _f, int _g, int _index) :f(_f), g(_g), index(_index) {};
		Node copy() { return Node(f, g, index); };
		void add(Node& node) {
			f += node.f;
			g += node.g;
		}
		void sub(Node& node) {
			f -= node.f;
			g -= node.g;
		}
	};

	template <>
	class Node<cv::Vec3i>
	{
	public:
		int f;
		cv::Vec3i g;
		int index;
		Node() :f(0), index(0) { g = cv::Vec3i(0, 0, 0); };
		Node(int _f, cv::Vec3i _g, int _index) :f(_f), index(_index) { g[0] = _g[0]; g[1] = _g[1];	g[2] = _g[2]; };
		Node copy() { return Node(f, g, index); };
		void add(Node& node) {
			f += node.f;
			g += node.g;
		}
		void sub(Node& node) {
			f -= node.f;
			g -= node.g;
		}
	};





	// Window when using calculated c and d (for second channel and beyond)
	template <typename G_TYPE>
	class Window
	{
	public:
		Window()
		{
			nodeList_.clear();
		}
		// This implementation uses a simple list instead of a linked list.
		std::list<Node<G_TYPE>> nodeList_;

		void addG(G_TYPE* g_x) {};
		void removeG(G_TYPE* g_x) {};
		void mergeG(Window<G_TYPE>& W) {};
		void separateG(Window<G_TYPE>& W) {};

		void addPixelToWindow(int* f_x, G_TYPE* g_x);
		void removePixelFromWindow(int* f_x, G_TYPE* g_x);
		template <typename WINDOW>
		void mergeWindow(WINDOW& W);
		template <typename WINDOW>
		void separateWindow(WINDOW& W);

		~Window()
		{
		}
	};

	// Window when c and d calculations are also performed (for first channel)
	template <typename G_TYPE>
	class Window_calc_cd : public Window<G_TYPE>
	{
	public:
		Window_calc_cd<G_TYPE>()
			: Window<G_TYPE>()
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
		WMF(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int numThreads)
			: f_img_(f_img), g_img_(g_img), numThreads_(numThreads), radius_(radius), epsilon_(epsilon), M_(f_img.rows), N_(f_img.cols), p_(radius), m_(radius + 1), channelNum_f_(f_img_.channels()), channelNum_g_(g_img_.channels())
		{
			originalDepth_ = f_img_.depth();
			if (f_img_.depth() != CV_32S)
				f_img_.convertTo(f_img_, CV_32S);
			if (g_img_.depth() != CV_32S)
				g_img_.convertTo(g_img_, CV_32S);

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


	// For initialization of g_cum
	template <typename TYPE>
	TYPE initialize_zero();
}