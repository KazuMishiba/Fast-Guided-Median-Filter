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

#include "FGMF_GPU_Or.cuh"


namespace FGMF_GPU_Or
{
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int num_level);



	class WMF
	{
	public:
		WMF(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange)
			: f_img_(f_img), g_img_(g_img), radius_(radius), eps2_(eps2), fRange_(fRange) , M_(f_img.rows), N_(f_img.cols), p_(radius), m_(radius + 1), channelNum_f_(f_img_.channels()), channelNum_g_(g_img_.channels())
		{
			//check validation
			assert(f_img_.depth() == CV_32S || f_img_.depth() == CV_8U);
			assert(g_img_.depth() == CV_32S || g_img_.depth() == CV_8U);
			if (f_img_.depth() != CV_32S)
				f_img_.convertTo(f_img_, CV_32S);
			if (g_img_.depth() != CV_32S)
				g_img_.convertTo(g_img_, CV_32S);

			/*
			if (channelNum_g_ == 1)
				c_ = cv::Mat(M_, N_, CV_32FC1);
			else
				c_ = cv::Mat(M_, N_, CV_32FC3);

			d_ = cv::Mat(M_, N_, CV_32FC1);
			*/
			c_d_ = new DeviceArray<float>(channelNum_g_ + 1, sizeInfo_);
		}

		cv::Mat apply_2d_filter();

		template<typename WINDOW, typename G_TYPE, typename C_TYPE>
		void filtering(const cv::Mat& f_single, const cv::Mat& g_multi, cv::Mat& result);

		void calculateCandD();


		~WMF()
		{

		}


	private:
		const cv::Mat f_img_;
		const cv::Mat g_img_;
		const int radius_;
		const float eps2_;
		const int fRange_;
		DeviceArray<float>* c_d_;
		DeviceArray<int>* G_;
		SizeInfo sizeInfo_;
		//cv::Mat c_;
		//cv::Mat d_;

		const int M_; // Image height
		const int N_; // Image width
		const int channelNum_f_; // Input image channel num
		const int channelNum_g_; // Guide image channel num
		const int p_; // x^+ = x + r = x + p_
		const int m_; // x^- = x - r - 1 = x + m_


	};


}