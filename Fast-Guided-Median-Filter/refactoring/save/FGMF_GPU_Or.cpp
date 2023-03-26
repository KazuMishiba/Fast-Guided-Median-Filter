#include "FGMF_GPU_Or.h"

namespace FGMF_GPU_Or
{
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int Imax)
	{
		WMF wmf = WMF(f_img, g_img, radius, eps2, Imax);
		return wmf.apply_2d_filter();

	}

	enum G_CHANNEL_TYPE
	{
		SINGLE,
		COLOR,
		OTHER
	};

	cv::Mat WMF::apply_2d_filter()
	{
		/*
		* チャンネル数によらず、まずc,dを計算する。
		* 次に
		*/
		// calculateC_D
		// filtering<G_CHANNEL_TYPE>()

		calculateCandD();


		//FとGのタイプで分岐
		if (channelNum_f_ == 1 && channelNum_g_ == 1) {
			cv::Mat result(M_, N_, CV_32SC1);
			filtering<G_CHANNEL_TYPE::SINGLE>(f_img_, g_img_, result);
			return result;
		}
		else if (channelNum_f_ == 1 && channelNum_g_ == 3) {
			cv::Mat result(M_, N_, CV_32SC1);
			filtering<Window_calc_cd<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_img_, g_img_, result);
			return result;
		}
		else if (channelNum_f_ == 3 && channelNum_g_ == 1) {
			cv::Mat result;
			std::vector<cv::Mat> results = { cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1) };
			std::vector<cv::Mat> f_singles;
			cv::split(f_img_, f_singles);

			filtering<Window_calc_cd<int>, int, float>(f_singles[0], g_img_, results[0]);
			filtering<Window<int>, int, float>(f_singles[1], g_img_, results[1]);
			filtering<Window<int>, int, float>(f_singles[2], g_img_, results[2]);

			cv::merge(results, result);
			return result;
		}
		else if (channelNum_f_ == 3 && channelNum_g_ == 3) {
			cv::Mat result;
			std::vector<cv::Mat> results = { cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1) };
			std::vector<cv::Mat> f_singles;
			cv::split(f_img_, f_singles);

			filtering<Window_calc_cd<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[0], g_img_, results[0]);
			filtering<Window<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[1], g_img_, results[1]);
			filtering<Window<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[2], g_img_, results[2]);

			cv::merge(results, result);
			return result;
		}
		return cv::Mat();
	}


	template<typename WINDOW, typename G_TYPE, typename C_TYPE>
	inline void WMF::filtering(const cv::Mat& f_single, const cv::Mat& g_multi, cv::Mat& result)
	{
	}

	void WMF::calculateCandD()
	{
		DeviceArray<int>* sumG = new DeviceArray<int>(channelNum_g_, sizeInfo_);
		DeviceArray<int>* tempG = new DeviceArray<int>(channelNum_g_, sizeInfo_);
		DeviceArray<int>* sumGG = new DeviceArray<int>((channelNum_g_ + 1) * channelNum_g_ / 2, sizeInfo_);
		DeviceArray<int>* tempGG = new DeviceArray<int>((channelNum_g_ + 1) * channelNum_g_ / 2, sizeInfo_);
		int pixelNumInWindow = (radius_ * 2 + 1) * (radius_ * 2 + 1);
		//cx, dx計算
		cu_calculateCxXDxFromG(sizeInfo_, NULL, G_, radius_, pixelNumInWindow, eps2_, c_d_, sumG, sumGG, tempG, tempGG);


		//メモリ開放
		delete sumG;
		delete tempG;
		delete sumGG;
		delete tempGG;
	}

}
