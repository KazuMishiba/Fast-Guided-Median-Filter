#include "FGMF_CPU_O1.h"

namespace FGMF_CPU_O1
{
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int threadNum, int radius, float eps2, int Imax)
	{
		WMF wmf = WMF(f_img, g_img, threadNum, radius, eps2, Imax);
		return wmf.apply_2d_filter();

	}

	cv::Mat WMF::apply_2d_filter()
	{
		//IとGのタイプで分岐
		if (channelNum_f_ == 1 && channelNum_g_ == 1) {
			cv::Mat result(M_, N_, CV_32SC1);
			filtering<Window_calc_cd<int>, int, float>(f_img_, g_img_, result);
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

	void WMF::test()
	{

		// アラインメントを指定して領域を確保する
		auto p = aligned_unique_ptr<int, 32>(10);  // 32バイトアラインメントを持つint型のunique_ptrを作成
		std::cout << "aligned_uptr: ";// << p.get() << std::endl;
		std::cout << reinterpret_cast<std::uintptr_t>(p.get()) % 32 << std::endl;
		// アラインメントを指定できないmallocで領域を確保する
		auto q = static_cast<int*>(malloc(sizeof(int[10])));
		std::cout << "malloc: ";// << q << std::endl;
		std::cout << reinterpret_cast<std::uintptr_t>(q) % 32 << std::endl;
	}

	template<typename WINDOW, typename G_TYPE, typename C_TYPE>
	inline void WMF::filtering(const cv::Mat& f_single, const cv::Mat& g_multi, cv::Mat& result)
	{
#if 0
		// 各分割結果の開始の数字を格納するvectorを用意する
		std::vector<int> starts;
		std::vector<int> ends;
		// 均等に分割するための幅を計算する
		int width = M_ / threadNum_;
		// 各分割結果の開始の数字を計算してvectorに格納する
		for (int i = 0; i < threadNum_; i++)
			starts.push_back(i * width);
		for (int i = 0; i < threadNum_ - 1; i++)
			ends.push_back(starts[i + 1] - 1);
		ends.push_back(M_ - 1);
#pragma omp parallel for
		for (int k = 0; k < threadNum_; k++)
		{
			filteringBlockHorizontal<WINDOW, G_TYPE, C_TYPE>(f_single, g_multi, result, starts[k], ends[k]);
		}
#else
		// 各分割結果の開始の数字を格納するvectorを用意する
		std::vector<int> starts;
		std::vector<int> ends;
		// 均等に分割するための幅を計算する
		int width = N_ / threadNum_;
		// 各分割結果の開始の数字を計算してvectorに格納する
		for (int i = 0; i < threadNum_; i++)
			starts.push_back(i * width);
		for (int i = 0; i < threadNum_ - 1; i++)
			ends.push_back(starts[i + 1] - 1);
		ends.push_back(N_ - 1);
#pragma omp parallel for
		for (int k = 0; k < threadNum_; k++)
		{
			filteringBlock<WINDOW, G_TYPE, C_TYPE>(f_single, g_multi, result, starts[k], ends[k]);
		}
#endif
	}

/*
O(1) sliding window approach

Input: f, g, c, d of size M × N
Output: f*

Initialize array of W(1) = {F(1),G(1), f(1)↓, g(1)↓, k(1)}
for t = 1 to M do
	Initialize W(2) = {F(2),G(2), f(2)↓, g(2)↓, k(2)}
	for s = 1 to N do
		Add pixel at (s+, t+) to W(1)[s+]
		Remove pixel at (s+, t-) from W(1)[s+]
		Merge W(1)[s+] into W(2)
		Separate W(1)[s-] from W(2)
		f*_s,t = Search weighted median(W(2), c_s,t, d_s,t)
	end for
end for


*/

	template<typename WINDOW, typename G_TYPE, typename C_TYPE>
	inline void WMF::filteringBlock(const cv::Mat& f_single, const cv::Mat& g_multi, cv::Mat& result, const int sStart_target, const int sEnd_target)
	{
		// 使用画素範囲開始列、終了列(+1)
		const int sStart_enable = std::max(sStart_target - radius_, 0);
		const int sEnd_enable = std::min(sEnd_target + radius_, N_ - 1);

		// W1ウィンドウ必要幅数
		const int w1Width = sEnd_enable - sStart_enable + 1;

		// Initialize array of W(1) = {F(1),G(1), f(1)↓, g(1)↓, k(1)}
		std::vector<std::unique_ptr<WINDOW>> pW1(w1Width);
		for (int i = 0; i < w1Width; ++i)
			pW1[i] = std::make_unique<WINDOW>(Imax_);


		for (int t = -radius_; t < M_; t++)
		{
			const int tp = t + p_;
			const int tm = t - m_;
			// Initialize W(2) = {F(2),G(2), f(2)↓, g(2)↓, k(2)}
			WINDOW W2(Imax_);//毎回宣言は無駄なのでいずれループ外に出す
			//画素ポインタ初期化
			const int* f_sp_tp = f_single.ptr<int>(std::min(tp, M_ - 1)) + sStart_enable;
			const int* f_sp_tm = f_single.ptr<int>(std::max(tm, 0)) + sStart_enable;
			const G_TYPE* g_x = g_multi.ptr<G_TYPE>(std::max(t, 0)) + sStart_target;
			const G_TYPE* g_sp_tp = g_multi.ptr<G_TYPE>(std::min(tp, M_ - 1)) + sStart_enable;
			const G_TYPE* g_sp_tm = g_multi.ptr<G_TYPE>(std::max(tm, 0)) + sStart_enable;
			int* f_star = result.ptr<int>(std::max(t, 0)) + sStart_target;

			C_TYPE* c_x = c_.ptr<C_TYPE>(std::max(t, 0)) + sStart_target;
			float* d_x = d_.ptr<float>(std::max(t, 0)) + sStart_target;

			for (int s = sStart_enable - radius_; s <= sEnd_target; s++)
			{
				const int sp = s + p_;
				const int sm = s - m_;
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
					*f_star = searchWeightedMedian<WINDOW, G_TYPE, C_TYPE>(g_x, W2, c_x, d_x);

					/*
					if (t==0)
						debugging<WINDOW, C_TYPE, G_TYPE>(s, t, W2, cx, dx, f_single, g_multi);
					*/

					g_x++;
					f_star++;
					c_x++;
					d_x++;
				}
			}
		}
	}

	template<typename WINDOW, typename C_TYPE, typename G_TYPE>
	void WMF::debugging(int s, int t, WINDOW& W, C_TYPE* c, float* d, const cv::Mat& f, const cv::Mat& g)
	{
	}

	template<>
	void WMF::debugging<Window_calc_cd<cv::Vec3i>, cv::Vec3f, cv::Vec3i>(int s, int t, Window_calc_cd<cv::Vec3i>& W, cv::Vec3f* c, float* d, const cv::Mat& f, const cv::Mat& g)
	{
		std::ofstream file("refactoring.txt", std::ios::app);

		file << "(" << s << "," << t << ") : ";
		file << "W.k " << W.k_;
		file << " calc(c,d)=(" << *c << "," << *d << ") ";
		file << " g_sum=" << W.dataForC_D_.g_sum_;
		file << " gg_sum=";
		for (int i = 0; i < 6; i++)
			file << W.dataForC_D_.gg_sum_[i];
		file << std::endl;
		/*
		cv::Vec3i g_sum(0,0,0);
		std::array<int, 6> gg_sum;
		for (int i = 0; i < 6; i++)
			gg_sum[i] = 0;
		for (int rt = -radius_; rt <= radius_; rt++)
		{
			int tt = t + rt;
			if (tt >= 0 && tt < M_)
			{
				for (int rs = -radius_; rs <= radius_; rs++)
				{
					int ss = s + rs;
					if (ss >= 0 && ss < N_)
					{
						cv::Vec3i _g = g.at<cv::Vec3i>(tt, ss);
						g_sum += _g;
						gg_sum[0] += _g[0] * _g[0];
						gg_sum[1] += _g[0] * _g[1];
						gg_sum[2] += _g[0] * _g[2];
						gg_sum[3] += _g[1] * _g[1];
						gg_sum[4] += _g[1] * _g[2];
						gg_sum[5] += _g[2] * _g[2];
					}
				}
			}
		}
		file << " g_sum=" << g_sum;
		file << " gg_sum=";
		for (int i = 0; i < 6; i++)
			file << gg_sum[i];
		file << std::endl;
		*/

		file.close();
	}


	//水平に切ってしまったがそういえば垂直切りが適切だった
	template<typename WINDOW, typename G_TYPE, typename C_TYPE>
	inline void WMF::filteringBlockHorizontal(const cv::Mat& f_single, const cv::Mat& g_multi, cv::Mat& result, const int tStart_target, const int tEnd_target)
	{
		// 使用画素範囲開始行、終了行(+1)
		const int tStart_enable = std::max(tStart_target - radius_, 0);
		const int tEnd_enable = std::min(tEnd_target + radius_, M_ - 1);

		// Initialize array of W(1) = {F(1),G(1), f(1)↓, g(1)↓, k(1)}
		std::vector<std::unique_ptr<WINDOW>> pW1(N_);
		for (int i = 0; i < N_; ++i)
			pW1[i] = std::make_unique<WINDOW>(Imax_);


		for (int t = tStart_enable - radius_; t <= tEnd_target; t++)
		{
			int tp = t + p_;
			int tm = t - m_;
			// Initialize W(2) = {F(2),G(2), f(2)↓, g(2)↓, k(2)}
			WINDOW W2(Imax_);//毎回宣言は無駄なのでいずれループ外に出す
			//画素ポインタ初期化
			const int* f_sp_tp = f_single.ptr<int>(std::min(tp, M_ - 1));
			const int* f_sp_tm = f_single.ptr<int>(std::max(tm, 0));
			const G_TYPE* g = g_multi.ptr<G_TYPE>(std::max(t, 0));
			const G_TYPE* g_sp_tp = g_multi.ptr<G_TYPE>(std::min(tp, M_ - 1));
			const G_TYPE* g_sp_tm = g_multi.ptr<G_TYPE>(std::max(tm, 0));
			int* f_star = result.ptr<int>(std::max(t, 0));

			C_TYPE* cx = c_.ptr<C_TYPE>(std::max(t, 0));
			float* dx = d_.ptr<float>(std::max(t, 0));

			for (int s = -radius_; s < N_; s++)
			{
				int sp = s + p_;
				int sm = s - m_;
				//Add pixel at(s+, t+) to W(1)[s+]
				if (sp < N_ && tp <= tEnd_enable) {
					addPixelToWindow<WINDOW, G_TYPE>(f_sp_tp, g_sp_tp, *pW1[sp]);

					f_sp_tp++;
					g_sp_tp++;
				}

				//Remove pixel at(s+, t-) from W(1)[s+]
				if (sp < N_ && tm >= tStart_enable) {
					removePixelFromWindow<WINDOW, G_TYPE>(f_sp_tm, g_sp_tm, *pW1[sp]);

					f_sp_tm++;
					g_sp_tm++;
				}

				//Merge W(1)[s+] into W(2)
				if (sp < N_ && t >= tStart_target)
					mergeWindow<WINDOW, G_TYPE>(*pW1[sp], W2);

				//Separate W(1)[s-] from W(2)
				if (sm >= 0 && t >= tStart_target)
					separateWindow<WINDOW, G_TYPE>(*pW1[sm], W2);

				//f*_{s,t} = Search weighted median(W(2), c_{s,t}, d_{s,t})
				if (s >= 0 && t >= tStart_target) {
					*f_star = searchWeightedMedian<WINDOW, G_TYPE, C_TYPE>(g, W2, cx, dx);

					g++;
					f_star++;
					cx++;
					dx++;
				}
			}
		}
	}


	template<typename WINDOW, typename G_TYPE>
	void WMF::addPixelToWindow(const int* f_x, const G_TYPE* g_x, WINDOW& W)
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
	void WMF::removePixelFromWindow(const int* f_x, const G_TYPE* g_x, WINDOW& W)
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
		for (int i = 0; i < Imax_ * (1 + sizeof(G_TYPE) / sizeof(int)); i += 8)
		{
			__m256i w1_fg = _mm256_load_si256((__m256i*) & W1.storeFG_[i]);
			__m256i w2_fg = _mm256_load_si256((__m256i*) & W2.storeFG_[i]);
			w2_fg = _mm256_add_epi32(w2_fg, w1_fg);
			_mm256_store_si256((__m256i*) & W2.storeFG_[i], w2_fg);
		}
#else
		for (int i = 0; i < Imax_; i++)
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

	//
#if defined(USE_AVX2)
		for (int i = 0; i < Imax_ * (1 + sizeof(G_TYPE) / sizeof(int)); i += 8)
		{
			__m256i w1_fg = _mm256_load_si256((__m256i*) & W1.storeFG_[i]);
			__m256i w2_fg = _mm256_load_si256((__m256i*) & W2.storeFG_[i]);
			w2_fg = _mm256_sub_epi32(w2_fg, w1_fg);
			_mm256_store_si256((__m256i*) & W2.storeFG_[i], w2_fg);
		}
#else
		for (int i = 0; i < Imax_; i++)
		{
			W2.F_[i] -= W1.F_[i];
			W2.G_[i] -= W1.G_[i];
		}
#endif

		W2.separateG(W1);
	}


	template<typename WINDOW, typename G_TYPE, typename C_TYPE>
	int WMF::searchWeightedMedian(const G_TYPE* g, WINDOW& W, C_TYPE* c_x, float* d_x)
	{
		//MODE_CDによっては、ここでc,dを計算する、また保存する
		calculateCandD(g, W, c_x, d_x);

		//
		float h_cum = calculateHcum<G_TYPE, C_TYPE>(W.f_cum_, W.g_cum_, c_x, d_x);
		if (h_cum < 0.5f)
		{
			while (h_cum < 0.5f)
			{
				W.k_ += 1;
				W.f_cum_ += W.F_[W.k_];
				W.g_cum_ += W.G_[W.k_];
				h_cum = calculateHcum(W.f_cum_, W.g_cum_, c_x, d_x);
				//if (W.k_ >= 255)
				//	printf("Over\n");
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
				/*
				if (W.k_ < 0)
					printf("Under\n");
				*/
			}
			return W.k_ + 1;
		}
	}
	
	template<typename WINDOW, typename C_TYPE, typename G_TYPE>
	void WMF::calculateCandD(const G_TYPE* g, WINDOW& W, C_TYPE* c_x, float* d_x)
	{
		// Noting to do.
	}
	
	template<>
	void WMF::calculateCandD<Window_calc_cd<int>, float, int>(const int* g_x, Window_calc_cd<int>& W, float* c_x, float* d_x)
	{
		const float invNumPixels = 1. / W.dataForC_D_.numPixels_;
		const float g_ave = W.dataForC_D_.g_sum_ * invNumPixels;
		const float gg_ave = W.dataForC_D_.gg_sum_ * invNumPixels;
		const float vx = gg_ave - g_ave * g_ave + eps2_;
		const float centered = *g_x - g_ave;
		*c_x = centered * invNumPixels / vx;
		*d_x = invNumPixels - g_ave * (*c_x);
	}
	template<>
	void WMF::calculateCandD<Window_calc_cd<cv::Vec3i>, cv::Vec3f, cv::Vec3i>(const cv::Vec3i* g_x, Window_calc_cd<cv::Vec3i>& W, cv::Vec3f* c_x, float* d_x)
	{
		const float invNumPixels = 1. / W.dataForC_D_.numPixels_;
		const cv::Vec3f g_ave = { W.dataForC_D_.g_sum_[0] * invNumPixels, W.dataForC_D_.g_sum_[1] * invNumPixels, W.dataForC_D_.g_sum_[2] * invNumPixels };
		const std::array<float, 6> gg_ave = {
			W.dataForC_D_.gg_sum_[0] * invNumPixels,
			W.dataForC_D_.gg_sum_[1] * invNumPixels,
			W.dataForC_D_.gg_sum_[2] * invNumPixels,
			W.dataForC_D_.gg_sum_[3] * invNumPixels,
			W.dataForC_D_.gg_sum_[4] * invNumPixels,
			W.dataForC_D_.gg_sum_[5] * invNumPixels,
		};
		const float v11 = gg_ave[0] - g_ave[0] * g_ave[0] + eps2_;
		const float v12 = gg_ave[1] - g_ave[0] * g_ave[1];
		const float v13 = gg_ave[2] - g_ave[0] * g_ave[2];
		const float v22 = gg_ave[3] - g_ave[1] * g_ave[1] + eps2_;
		const float v23 = gg_ave[4] - g_ave[1] * g_ave[2];
		const float v33 = gg_ave[5] - g_ave[2] * g_ave[2] + eps2_;


		const float det =
			  v11 * (v22 * v33 - v23 * v23)
			- v12 * (v12 * v33 - v13 * v23)
			+ v13 * (v12 * v23 - v13 * v22);
		float invDet;
		if (abs(det) < 1e-10)
			invDet = 0;
		else
			invDet = 1. / det;
		// 逆行列を計算
		const float b11 = (v22 * v33 - v23 * v23)* invDet;
		const float b12 = (v13 * v23 - v12 * v33)* invDet;
		const float b13 = (v12 * v23 - v13 * v22)* invDet;
		const float b22 = (v11 * v33 - v13 * v13)* invDet;
		const float b23 = (v12 * v13 - v11 * v23)* invDet;
		const float b33 = (v11 * v22 - v12 * v12)* invDet;

		const float centered1 = ((*g_x)[0] - g_ave[0]) * invNumPixels;
		const float centered2 = ((*g_x)[1] - g_ave[1]) * invNumPixels;
		const float centered3 = ((*g_x)[2] - g_ave[2]) * invNumPixels;

		(*c_x)[0] = b11 * centered1 + b12 * centered2 + b13 * centered3;
		(*c_x)[1] = b12 * centered1 + b22 * centered2 + b23 * centered3;
		(*c_x)[2] = b13 * centered1 + b23 * centered2 + b33 * centered3;

		*d_x = invNumPixels - ((*c_x)[0] * g_ave[0] + (*c_x)[1] * g_ave[1] + (*c_x)[2] * g_ave[2]);
	}


	/*
	template<typename G_TYPE, typename C_TYPE>
	float WMF::calculateHcum(int& f_cum, G_TYPE& g_cum, C_TYPE* c, float* d)
	{
		return 0;
	}
	*/
	template<>
	float WMF::calculateHcum<int, float>(int& f_cum, int& g_cum, float* c, float *d)
	{
		return *c * g_cum + *d * f_cum;
	}
	template<>
	float WMF::calculateHcum<cv::Vec3i, cv::Vec3f>(int& f_cum, cv::Vec3i& g_cum, cv::Vec3f* c, float* d)
	{
		return (*c)[0] * g_cum[0] + (*c)[1] * g_cum[1] + (*c)[2] * g_cum[2]+ *d * f_cum;
	}


	template <typename G_TYPE>
	void Window_calc_cd<G_TYPE>::addG(const G_TYPE* g_x)
	{
		dataForC_D_.addG(g_x);
	}
	template <typename G_TYPE>
	void Window_calc_cd<G_TYPE>::removeG(const G_TYPE* g_x)
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
