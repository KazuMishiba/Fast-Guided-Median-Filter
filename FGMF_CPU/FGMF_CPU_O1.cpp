#include "FGMF_CPU_O1.h"

namespace FGMF_CPU_O1
{
	// Filter_2d function is a wrapper for WMF filtering
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float epsilon, int fRange, int numThreads)
	{
		WMF wmf = WMF(f_img, g_img, radius, epsilon, fRange, numThreads);
		return wmf.apply_2d_filter();

	}

	cv::Mat WMF::apply_2d_filter()
	{
		cv::Mat result(M_, N_, f_img_.type());

		// Branch based on the types of F and G
		if (channelNum_f_ == 1 && channelNum_g_ == 1) {
			filtering<Window_calc_cd<int>, int, float>(f_img_, g_img_, result);
		}
		else if (channelNum_f_ == 1 && channelNum_g_ == 3) {
			filtering<Window_calc_cd<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_img_, g_img_, result);
		}
		else if (channelNum_f_ == 3 && channelNum_g_ == 1) {
			std::vector<cv::Mat> results = { cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1) };
			std::vector<cv::Mat> f_singles;
			cv::split(f_img_, f_singles);

			filtering<Window_calc_cd<int>, int, float>(f_singles[0], g_img_, results[0]);
			filtering<Window<int>, int, float>(f_singles[1], g_img_, results[1]);
			filtering<Window<int>, int, float>(f_singles[2], g_img_, results[2]);

			cv::merge(results, result);
		}
		else if (channelNum_f_ == 3 && channelNum_g_ == 3) {
			std::vector<cv::Mat> results = { cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1), cv::Mat(M_, N_, CV_32SC1) };
			std::vector<cv::Mat> f_singles;
			cv::split(f_img_, f_singles);

			filtering<Window_calc_cd<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[0], g_img_, results[0]);
			filtering<Window<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[1], g_img_, results[1]);
			filtering<Window<cv::Vec3i>, cv::Vec3i, cv::Vec3f>(f_singles[2], g_img_, results[2]);

			cv::merge(results, result);
		}
		else {
			throw std::invalid_argument("The number of channels in the input image and output image is " + std::to_string(channelNum_f_) + " and " + std::to_string(channelNum_g_) + ", although only 1 or 3 are allowed.");
		}

		// Restore the data type
		if (result.depth() != originalDepth_)
		{
			result.convertTo(result, originalDepth_);
		}
		return result;
	}


	template<typename WINDOW, typename G_TYPE, typename C_TYPE>
	inline void WMF::filtering(cv::Mat& f_single, cv::Mat& g_multi, cv::Mat& result)
	{
		// Prepare a vector that stores the starting and ending column numbers of each division result
		std::vector<int> starts;
		std::vector<int> ends;
		// Calculate the width to divide equally
		int width = N_ / numThreads_;
		// Calculate and store the starting and ending column numbers for each divided result in the vector
		for (int i = 0; i < numThreads_; i++)
			starts.push_back(i * width);
		for (int i = 0; i < numThreads_ - 1; i++)
			ends.push_back(starts[i + 1] - 1);
		ends.push_back(N_ - 1);
#pragma omp parallel for
		for (int k = 0; k < numThreads_; k++)
		{
			filteringForDividedRegion<WINDOW, G_TYPE, C_TYPE>(f_single, g_multi, result, starts[k], ends[k]);
		}
	}

/*
O(1) sliding window approach

Input: f, g, c, d of size M �~ N
Output: f*

Initialize array of W(1) = {F(1),G(1), f(1)��, g(1)��, k(1)}
for t = 1 to M do
	Initialize W(2) = {F(2),G(2), f(2)��, g(2)��, k(2)}
	for s = 1 to N do
		Add pixel at (s+, t+) to W(1)[s+]
		Remove pixel at (s+, t-) from W(1)[s+]
		Merge W(1)[s+] into W(2)
		Separate W(1)[s-] from W(2)
		f*_s,t = Search weighted median(W(2), c_s,t, d_s,t)
	end for
end for


*/

	// Filtering for divided regions
	template<typename WINDOW, typename G_TYPE, typename C_TYPE>
	inline void WMF::filteringForDividedRegion(cv::Mat& f_single, cv::Mat& g_multi, cv::Mat& result, int sStart_target, int sEnd_target)
	{
		// Determine the range of pixels to use, start column, and end column
		int sStart_enable = std::max(sStart_target - radius_, 0);
		int sEnd_enable = std::min(sEnd_target + radius_, N_ - 1);

		//  Number of array elements of W(1)
		int w1Width = sEnd_enable - sStart_enable + 1;

		// Initialize array of W(1) = {F(1),G(1), f(1)��, g(1)��, k(1)}
		std::vector<std::unique_ptr<WINDOW>> pW1(w1Width);
		for (int i = 0; i < w1Width; ++i)
			pW1[i] = std::make_unique<WINDOW>(fRange_);


		for (int t = -radius_; t < M_; t++)
		{
			int tp = t + p_;
			int tm = t - m_;
			// Initialize W(2) = {F(2),G(2), f(2)��, g(2)��, k(2)}
			WINDOW W2(fRange_);
			// Initialize pixel pointers
			int* f_sp_tp = f_single.ptr<int>(std::min(tp, M_ - 1)) + sStart_enable;
			int* f_sp_tm = f_single.ptr<int>(std::max(tm, 0)) + sStart_enable;
			G_TYPE* g_x = g_multi.ptr<G_TYPE>(std::max(t, 0)) + sStart_target;
			G_TYPE* g_sp_tp = g_multi.ptr<G_TYPE>(std::min(tp, M_ - 1)) + sStart_enable;
			G_TYPE* g_sp_tm = g_multi.ptr<G_TYPE>(std::max(tm, 0)) + sStart_enable;
			int* f_star = result.ptr<int>(std::max(t, 0)) + sStart_target;

			C_TYPE* c_x = c_.ptr<C_TYPE>(std::max(t, 0)) + sStart_target;
			float* d_x = d_.ptr<float>(std::max(t, 0)) + sStart_target;

			for (int s = sStart_enable - radius_; s <= sEnd_target; s++)
			{
				int sp = s + p_;
				int sm = s - m_;
				//Add pixel at(s+, t+) to W(1)[s+]
				if (sp < N_ && tp < M_) {
					addPixelToWindow<WINDOW, G_TYPE>(f_sp_tp, g_sp_tp, *pW1[sp - sStart_enable]);
					f_sp_tp++;
					g_sp_tp++;
				}

				//Remove pixel at(s+, t-) from W(1)[s+]
				if (sp < N_ && tm >= 0) {
					removePixelFromWindow<WINDOW, G_TYPE>(f_sp_tm, g_sp_tm, *pW1[sp - sStart_enable]);
					f_sp_tm++;
					g_sp_tm++;
				}

				//Merge W(1)[s+] into W(2)
				if (sp < N_ && t >= 0)
					mergeWindow<WINDOW, G_TYPE>(*pW1[sp - sStart_enable], W2);

				//Separate W(1)[s-] from W(2)
				if (sm >= sStart_enable && t >= 0)
					separateWindow<WINDOW, G_TYPE>(*pW1[sm - sStart_enable], W2);

				//f*_{s,t} = Search weighted median(W(2), c_{s,t}, d_{s,t})
				if (s >= sStart_target && t >= 0) {
					*f_star = searchWeightedMedian<WINDOW, G_TYPE, C_TYPE>(*g_x, W2, *c_x, *d_x);

					g_x++;
					f_star++;
					c_x++;
					d_x++;
				}
			}
		}
	}



	template<typename WINDOW, typename G_TYPE>
	void WMF::addPixelToWindow(int* f_x, G_TYPE* g_x, WINDOW& W)
	{
		W.F_[*f_x] += 1;
		W.G_[*f_x] += *g_x;
		if (*f_x <= W.k_)
		{
			W.f_cum_ += 1;
			W.g_cum_ += *g_x;
		}
		W.addG(g_x);
	}
	template<typename WINDOW, typename G_TYPE>
	void WMF::removePixelFromWindow(int* f_x, G_TYPE* g_x, WINDOW& W)
	{
		W.F_[*f_x] -= 1;
		W.G_[*f_x] -= *g_x;
		if (*f_x <= W.k_)
		{
			W.f_cum_ -= 1;
			W.g_cum_ -= *g_x;
		}
		W.removeG(g_x);
	}

	template<typename WINDOW, typename G_TYPE>
	void WMF::mergeWindow(WINDOW& W1, WINDOW& W2)
	{
		if (W1.k_ < W2.k_)
		{
			while (W1.k_ < W2.k_)
			{
				W1.k_ += 1;
				W1.f_cum_ += W1.F_[W1.k_];
				W1.g_cum_ += W1.G_[W1.k_];
			}
		}
		else
		{
			while (W1.k_ > W2.k_)
			{
				W1.f_cum_ -= W1.F_[W1.k_];
				W1.g_cum_ -= W1.G_[W1.k_];
				W1.k_ -= 1;
			}
		}
		W2.f_cum_ += W1.f_cum_;
		W2.g_cum_ += W1.g_cum_;

#if defined(USE_AVX2)
		for (int i = 0; i < fRange_ * (1 + sizeof(G_TYPE) / sizeof(int)); i += 8)
		{
			__m256i w1_fg = _mm256_load_si256((__m256i*) & W1.storeFG_[i]);
			__m256i w2_fg = _mm256_load_si256((__m256i*) & W2.storeFG_[i]);
			w2_fg = _mm256_add_epi32(w2_fg, w1_fg);
			_mm256_store_si256((__m256i*) & W2.storeFG_[i], w2_fg);
		}
#else
		for (int i = 0; i < fRange_; i++)
		{
			W2.F_[i] += W1.F_[i];
			W2.G_[i] += W1.G_[i];
		}
#endif
		W2.mergeG(W1);
	}


	template<typename WINDOW, typename G_TYPE>
	void WMF::separateWindow(WINDOW& W1, WINDOW& W2)
	{
		int k = W1.k_;
		if (k < W2.k_)
		{
			while (k < W2.k_)
			{
				k += 1;
				W2.f_cum_ -= W1.F_[k];
				W2.g_cum_ -= W1.G_[k];
			}
		}
		else
		{
			while (k > W2.k_)
			{
				W2.f_cum_ += W1.F_[k];
				W2.g_cum_ += W1.G_[k];
				k -= 1;
			}
		}
		W2.f_cum_ -= W1.f_cum_;
		W2.g_cum_ -= W1.g_cum_;

#if defined(USE_AVX2)
		for (int i = 0; i < fRange_ * (1 + sizeof(G_TYPE) / sizeof(int)); i += 8)
		{
			__m256i w1_fg = _mm256_load_si256((__m256i*) & W1.storeFG_[i]);
			__m256i w2_fg = _mm256_load_si256((__m256i*) & W2.storeFG_[i]);
			w2_fg = _mm256_sub_epi32(w2_fg, w1_fg);
			_mm256_store_si256((__m256i*) & W2.storeFG_[i], w2_fg);
		}
#else
		for (int i = 0; i < fRange_; i++)
		{
			W2.F_[i] -= W1.F_[i];
			W2.G_[i] -= W1.G_[i];
		}
#endif
		W2.separateG(W1);
	}


	template<typename WINDOW, typename G_TYPE, typename C_TYPE>
	int WMF::searchWeightedMedian(G_TYPE& g, WINDOW& W, C_TYPE& c_x, float& d_x)
	{
		// Calculate c and d when WINDOW is Window_calc_cd
		calculateCandD(g, W, c_x, d_x);

		//
		//float h_cum = calculateHcum<G_TYPE, C_TYPE>(W.f_cum_, W.g_cum_, c_x, d_x);
		float h_cum = calculateHcum(W.f_cum_, W.g_cum_, c_x, d_x);
		if (h_cum < 0.5f)
		{
			while (h_cum < 0.5f)
			{
				W.k_ += 1;
				W.f_cum_ += W.F_[W.k_];
				W.g_cum_ += W.G_[W.k_];
				h_cum = calculateHcum(W.f_cum_, W.g_cum_, c_x, d_x);
			}
			return W.k_;
		}
		else
		{
			while (h_cum >= 0.5f)
			{
				W.f_cum_ -= W.F_[W.k_];
				W.g_cum_ -= W.G_[W.k_];
				h_cum = calculateHcum(W.f_cum_, W.g_cum_, c_x, d_x);
				W.k_ -= 1;
			}
			return W.k_ + 1;
		}
	}
	
	template<typename WINDOW, typename C_TYPE, typename G_TYPE>
	void WMF::calculateCandD(G_TYPE& g, WINDOW& W, C_TYPE& c_x, float& d_x)
	{
		// Noting to do.
	}
	// Calculate c and d when the guide image has single channel
	template<>
	void WMF::calculateCandD<Window_calc_cd<int>, float, int>(int& g_x, Window_calc_cd<int>& W, float& c_x, float& d_x)
	{
		float invNumPixels = 1. / W.dataForC_D_.numPixels_;
		float g_ave = W.dataForC_D_.g_sum_ * invNumPixels;
		float gg_ave = W.dataForC_D_.gg_sum_ * invNumPixels;
		float vx = gg_ave - g_ave * g_ave + epsilon_;
		float centered = g_x - g_ave;
		c_x = centered * invNumPixels / vx;
		d_x = invNumPixels - g_ave * c_x;
	}
	// Calculate c and d when the guide image has 3 channels
	template<>
	void WMF::calculateCandD<Window_calc_cd<cv::Vec3i>, cv::Vec3f, cv::Vec3i>(cv::Vec3i& g_x, Window_calc_cd<cv::Vec3i>& W, cv::Vec3f& c_x, float& d_x)
	{
		float invNumPixels = 1. / W.dataForC_D_.numPixels_;
		cv::Vec3f g_ave = { W.dataForC_D_.g_sum_[0] * invNumPixels, W.dataForC_D_.g_sum_[1] * invNumPixels, W.dataForC_D_.g_sum_[2] * invNumPixels };
		std::array<float, 6> gg_ave = {
			W.dataForC_D_.gg_sum_[0] * invNumPixels,
			W.dataForC_D_.gg_sum_[1] * invNumPixels,
			W.dataForC_D_.gg_sum_[2] * invNumPixels,
			W.dataForC_D_.gg_sum_[3] * invNumPixels,
			W.dataForC_D_.gg_sum_[4] * invNumPixels,
			W.dataForC_D_.gg_sum_[5] * invNumPixels,
		};
		float v11 = gg_ave[0] - g_ave[0] * g_ave[0] + epsilon_;
		float v12 = gg_ave[1] - g_ave[0] * g_ave[1];
		float v13 = gg_ave[2] - g_ave[0] * g_ave[2];
		float v22 = gg_ave[3] - g_ave[1] * g_ave[1] + epsilon_;
		float v23 = gg_ave[4] - g_ave[1] * g_ave[2];
		float v33 = gg_ave[5] - g_ave[2] * g_ave[2] + epsilon_;

		float det =
			  v11 * (v22 * v33 - v23 * v23)
			- v12 * (v12 * v33 - v13 * v23)
			+ v13 * (v12 * v23 - v13 * v22);
		float invDet;
		if (abs(det) < 1e-10)
			invDet = 0;
		else
			invDet = 1. / det;
		float b11 = (v22 * v33 - v23 * v23)* invDet;
		float b12 = (v13 * v23 - v12 * v33)* invDet;
		float b13 = (v12 * v23 - v13 * v22)* invDet;
		float b22 = (v11 * v33 - v13 * v13)* invDet;
		float b23 = (v12 * v13 - v11 * v23)* invDet;
		float b33 = (v11 * v22 - v12 * v12)* invDet;

		float centered1 = (g_x[0] - g_ave[0]) * invNumPixels;
		float centered2 = (g_x[1] - g_ave[1]) * invNumPixels;
		float centered3 = (g_x[2] - g_ave[2]) * invNumPixels;

		c_x[0] = b11 * centered1 + b12 * centered2 + b13 * centered3;
		c_x[1] = b12 * centered1 + b22 * centered2 + b23 * centered3;
		c_x[2] = b13 * centered1 + b23 * centered2 + b33 * centered3;

		d_x = invNumPixels - (c_x[0] * g_ave[0] + c_x[1] * g_ave[1] + c_x[2] * g_ave[2]);
	}


	float WMF::calculateHcum(int& f_cum, int& g_cum, float& c_x, float& d_x)
	{
		return c_x * g_cum + d_x * f_cum;
	}
	float WMF::calculateHcum(int& f_cum, cv::Vec3i& g_cum, cv::Vec3f& c_x, float& d_x)
	{
		return c_x[0] * g_cum[0] + c_x[1] * g_cum[1] + c_x[2] * g_cum[2] + d_x * f_cum;
	}

	template <typename G_TYPE>
	void Window_calc_cd<G_TYPE>::addG(G_TYPE* g_x)
	{
		dataForC_D_.addG(g_x);
	}
	template <typename G_TYPE>
	void Window_calc_cd<G_TYPE>::removeG(G_TYPE* g_x)
	{
		dataForC_D_.removeG(g_x);
	}
	template <typename G_TYPE>
	void Window_calc_cd<G_TYPE>::mergeG(Window_calc_cd<G_TYPE>& W)
	{
		dataForC_D_.mergeG(W.dataForC_D_);
	}
	template <typename G_TYPE>
	void Window_calc_cd<G_TYPE>::separateG(Window_calc_cd<G_TYPE>& W)
	{
		dataForC_D_.separateG(W.dataForC_D_);
	}
	
}
