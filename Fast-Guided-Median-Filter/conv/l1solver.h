

/***************************************************************/
/*
*   Distribution code Version 1.1 -- 09/21/2014 by Qi Zhang Copyright 2014, The Chinese University of Hong Kong.
*
*   The Code is created based on the method described in the following paper
*   [1] "100+ Times Faster Weighted Median Filter", Qi Zhang, Li Xu, Jiaya Jia, IEEE Conference on
*		Computer Vision and Pattern Recognition (CVPR), 2014
*
*   Due to the adaption for supporting mask and different types of input, this code is
*   slightly slower than the one claimed in the original paper. Please use
*   our executable on our website for performance comparison.
*
*   The code and the algorithm are for non-comercial use only.
*
/***************************************************************/


#ifndef L1SOLVER_H
#define L1SOLVER_H

//#include "common.h"

/***************************************************************/
/*
* Standard IO library is required.
* STL String library is required.
*
/***************************************************************/
#include <cstdio>
#include <string>
#include <random>
#include <iostream>
#include <fstream>
/***************************************************************/
/*
* OpenCV 2.4 is required.
* The following code is already built on OpenCV 2.4.2.
*
/***************************************************************/
#include "opencv2/core/core.hpp"
#include <time.h>


#define sign_float(a,b) (a>b)?1.0f:(a<b)?-1.0f:0.0f

//Use the namespace of CV and STD
using namespace std;
//using namespace cv;



class l1Solver {
public:
	//WMFを適用
	static cv::Mat filter(cv::Mat &I, cv::Mat &feature, int r, float sigma = 25.5, int nI = 256, int nF = 256, int iter = 1, string weightType = "exp", cv::Mat mask = cv::Mat());
	static cv::Mat filterIterative(cv::Mat &I, cv::Mat &feature, int r, float sigma = 25.5, int nI = 256, int nF = 256, int iter = 1, string weightType = "exp", cv::Mat mask = cv::Mat());
	static cv::Mat filterCore(cv::Mat &I, cv::Mat &F, float **wMap, int r = 20, int nF = 256, int nI = 256, cv::Mat mask = cv::Mat());
	static cv::Mat filterCoreParallel(cv::Mat &I, cv::Mat &F, float **wMap, int r = 20, int nF = 256, int nI = 256, cv::Mat mask = cv::Mat());





private:
	
	/***************************************************************/
	/* Function: updateBCB
	* Description: maintain the necklace table of BCB
	/***************************************************************/
	static inline void updateBCB(int &num, int *f, int *b, int i, int v) {

		static int p1, p2;
		//numはある特徴index fに関する、カットポイントの左右のビン数の差B(f)
		//f,bはnecklace tableのやつ
		//i：index fのことのはず。テーブル更新用
		//v:B(f)をどれだけ増やすか（減らすか）
		//vについては、
		//初期化時はカットポイントが一番左にあるので、全部マイナスなので、1回のアップデート(初期化時のビン1追加)につき-1(これもfloatにすると-1ではなく-cとかになるのか)
		//カットポイント移動時は、右に移動する場合は　ヒストグラムビン数*2足す
		//左に移動する場合はヒストグラムビン数*2引く
		//ウィンドウシフト時、追加する場合は　今のカットポイント（＝中央値）がある位置に対して
		//追加対象の画素indexがカットポイントより左なら１足す、右なら１引く
		//除外する場合は
		if (i) {
			if (!num) { // cell is becoming non-empty
				p2 = f[0];
				f[0] = i;
				f[i] = p2;
				b[p2] = i;
				b[i] = 0;
			}
			else if (!(num + v)) {// cell is becoming empty
				//ある特徴index f　が空ならテーブルから削除
				//と思ったが、和B(f)が0なら削除だった。
				p1 = b[i], p2 = f[i];
				f[p1] = p2;
				b[p2] = p1;
			}
		}

		// update the cell count
		num += v;
	}

	/***************************************************************/
	/* Function: float2D
	* Description: allocate a 2D float array with dimension "dim0 x dim1"
	/***************************************************************/
	static float** float2D(int dim0, int dim1) {
		float **ret = new float*[dim0];
		ret[0] = new float[dim0*dim1];
		for (int i = 1; i<dim0; i++)ret[i] = ret[i - 1] + dim1;

		return ret;
	}

	/***************************************************************/
	/* Function: float2D_release
	* Description: deallocate the 2D array created by float2D()
	/***************************************************************/
	static void float2D_release(float **p) {
		delete[]p[0];
		delete[]p;
	}

	/***************************************************************/
	/* Function: int2D
	* Description: allocate a 2D integer array with dimension "dim0 x dim1"
	/***************************************************************/
	static int** int2D(int dim0, int dim1) {
		int **ret = new int*[dim0];
		ret[0] = new int[dim0*dim1];
		for (int i = 1; i<dim0; i++)ret[i] = ret[i - 1] + dim1;

		return ret;
	}

	/***************************************************************/
	/* Function: int2D_release
	* Description: deallocate the 2D array created by int2D()
	/***************************************************************/
	static void int2D_release(int **p) {
		delete[]p[0];
		delete[]p;
	}

	/***************************************************************/
	/* Function: featureIndexing
	* Description: convert uchar feature image "F" to CV_32SC1 type.
	*				If F is 3-channel, perform k-means clustering
	*				If F is 1-channel, only perform type-casting
	* wMapはNf*Nf(Nfは特徴数)のテーブルで、特徴iと特徴jの重みをあらかじめ計算しておいて格納し、wMap[i][j]で読みだす
	/***************************************************************/
	static void featureIndexing(cv::Mat &F, float **&wMap, int &nF, float sigmaI, string weightType) {


		// Configuration and Declaration
		cv::Mat FNew;
		int cols = F.cols, rows = F.rows;
		int alls = cols * rows;
		int KmeansAttempts = 1;
		vector<string> ops;
		ops.push_back("exp");
		ops.push_back("iv1");
		ops.push_back("iv2");
		ops.push_back("cos");
		ops.push_back("jac");
		ops.push_back("off");

		// Get weight type number
		int numOfOps = (int)ops.size();
		int op = 0;
		for (; op<numOfOps; op++)if (ops[op] == weightType)break;
		if (op >= numOfOps)op = 0;

		/* For 1 channel feature image (uchar)*/
		if (F.channels() == 1) {

			nF = 256;

			// Type-casting
			F.convertTo(FNew, CV_32S);

			// Computer weight map (weight between each pair of feature index)
			{
				wMap = float2D(nF, nF);
				float nSigmaI = sigmaI;
				float divider = (1.0f / (2 * nSigmaI*nSigmaI));

#pragma omp parallel for
				for (int i = 0; i<nF; i++) {
					for (int j = i; j<nF; j++) {
						float diff = fabs((float)(i - j));
						if (op == 0)wMap[i][j] = wMap[j][i] = exp(-(diff*diff)*divider); // EXP 2
						else if (op == 2)wMap[i][j] = wMap[j][i] = 1.0f / (diff*diff + nSigmaI * nSigmaI); // IV2
						else if (op == 1)wMap[i][j] = wMap[j][i] = 1.0f / (diff + nSigmaI);// IV1
						else if (op == 3)wMap[i][j] = wMap[j][i] = 1.0f; // COS
						else if (op == 4)wMap[i][j] = wMap[j][i] = (float)(min(i, j)*1.0 / max(i, j)); // Jacard
						else if (op == 5)wMap[i][j] = wMap[j][i] = 1.0f; // Unweighted
					}
				}
			}
		}
		/* For 3 channel feature image (uchar)*/
		else if (F.channels() == 3) {

			//const int shift = 0;// 2; // 256(8-bit)->64(6-bit)
			const int shift = 2; // 256(8-bit)->64(6-bit)
			const int LOW_NUM = 256 >> shift;
			static int hash[LOW_NUM][LOW_NUM][LOW_NUM] = { 0 };

			memset(hash, 0, sizeof(hash));

			// throw pixels into a 2D histogram
			int candCnt = 0;
			{

				int lowR, lowG, lowB;
				uchar *FPtr = F.ptr<uchar>();
				for (int i = 0, i3 = 0; i<alls; i++, i3 += 3) {
					lowB = FPtr[i3] >> shift;
					lowG = FPtr[i3 + 1] >> shift;
					lowR = FPtr[i3 + 2] >> shift;

					if (hash[lowB][lowG][lowR] == 0) {
						candCnt++;
						hash[lowB][lowG][lowR] = 1;
					}
				}
			}

			nF = min(nF, candCnt);
			cv::Mat samples(candCnt, 3, CV_32F);

			//prepare for K-means
			{
				int top = 0;
				for (int i = 0; i < LOW_NUM; i++)
				{
					for (int j = 0; j < LOW_NUM; j++)
					{
						for (int k = 0; k < LOW_NUM; k++) {
							if (hash[i][j][k]) {
								samples.ptr<float>(top)[0] = (float)i;
								samples.ptr<float>(top)[1] = (float)j;
								samples.ptr<float>(top)[2] = (float)k;
								top++;
							}
						}
					}
				}
			}

			//do K-means
			cv::Mat labels;
			cv::Mat centers;
			{
				kmeans(samples, nF, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 0, 10000), KmeansAttempts, cv::KMEANS_PP_CENTERS, centers);
			}

			//make connection (i,j,k) <-> index
			{
				int top = 0;
				for (int i = 0; i<LOW_NUM; i++)for (int j = 0; j<LOW_NUM; j++)for (int k = 0; k<LOW_NUM; k++) {
					if (hash[i][j][k]) {
						hash[i][j][k] = labels.ptr<int>(top)[0];
						top++;
					}
				}
			}

			// generate index map
			{
				FNew = cv::Mat(F.size(), CV_32SC1);

				int lowR, lowG, lowB;
				uchar *FPtr = F.ptr<uchar>();
				for (int i = 0, i3 = 0; i<alls; i++, i3 += 3) {
					lowB = FPtr[i3] >> shift;
					lowG = FPtr[i3 + 1] >> shift;
					lowR = FPtr[i3 + 2] >> shift;

					FNew.ptr<int>()[i] = hash[lowB][lowG][lowR];
				}
			}

			// Computer weight map (weight between each pair of feature index)
			{
				wMap = float2D(nF, nF);
				float nSigmaI = sigmaI / 256.0f*LOW_NUM;
				float divider = (1.0f / (2 * nSigmaI*nSigmaI));

				float *length = new float[nF];
				for (int i = 0; i<nF; i++) {
					float a0 = centers.ptr<float>(i)[0];
					float a1 = centers.ptr<float>(i)[1];
					float a2 = centers.ptr<float>(i)[2];
					length[i] = sqrt(a0*a0 + a1 * a1 + a2 * a2);
				}


#pragma omp parallel for
				for (int i = 0; i<nF; i++) {
					for (int j = i; j<nF; j++) {
						float a0 = centers.ptr<float>(i)[0], b0 = centers.ptr<float>(j)[0];
						float a1 = centers.ptr<float>(i)[1], b1 = centers.ptr<float>(j)[1];
						float a2 = centers.ptr<float>(i)[2], b2 = centers.ptr<float>(j)[2];
						float diff0 = a0 - b0;
						float diff1 = a1 - b1;
						float diff2 = a2 - b2;

						if (op == 0)wMap[i][j] = wMap[j][i] = exp(-(diff0*diff0 + diff1 * diff1 + diff2 * diff2)*divider); // EXP 2
						else if (op == 2)wMap[i][j] = wMap[j][i] = 1.0f / (diff0*diff0 + diff1 * diff1 + diff2 * diff2 + nSigmaI * nSigmaI); // IV2
						else if (op == 1)wMap[i][j] = wMap[j][i] = 1.0f / (fabs(diff0) + fabs(diff1) + fabs(diff2) + nSigmaI);// IV1
						else if (op == 3)wMap[i][j] = wMap[j][i] = (a0*b0 + a1 * b1 + a2 * b2) / (length[i] * length[j]); // COS
						else if (op == 4)wMap[i][j] = wMap[j][i] = (min(a0, b0) + min(a1, b1) + min(a2, b2)) / (max(a0, b0) + max(a1, b1) + max(a2, b2)); // Jacard
						else if (op == 5)wMap[i][j] = wMap[j][i] = 1.0f; // Unweighted
					}
				}

				delete[]length;

			}

		}

		//end of the function
		F = FNew;
	}

	//提案法用　余計な処理を省いた版　入力のFはCV_32Sとする
	static void featureIndexingForProp(cv::Mat &F, float **&wMap, int &nF, float sigmaI) {
		//nF = 256;

		// Computer weight map (weight between each pair of feature index)
		{
				wMap = float2D(nF, nF);
				float nSigmaI = sigmaI;
				float divider = (1.0f / (2 * nSigmaI*nSigmaI));

				for (int i = 0; i<nF; i++) {
					for (int j = i; j<nF; j++) {
						float diff = fabs((float)(i - j));
						wMap[i][j] = wMap[j][i] = exp(-(diff*diff)*divider); // EXP 2
					}
				}
		}
	}


	/***************************************************************/
	/* Function: from32FTo32S
	* Description: adaptive quantization for changing a floating-point 1D image to integer image.
	*				The adaptive quantization strategy is based on binary search, which searches an
	*				upper bound of quantization error.
	*				The function also return a mapping between quantized value (32F) and quantized index (32S).
	*				The mapping is used to convert integer image back to floating-point image after filtering.
	/***************************************************************/
	static void from32FTo32S(cv::Mat &img, cv::Mat &outImg, int nI, float *mapping) {


		int rows = img.rows, cols = img.cols;
		int alls = rows * cols;

		float *imgPtr = img.ptr<float>();

		typedef pair<float, int> pairFI;

		pairFI *data = (pairFI *)malloc(alls * sizeof(pairFI));

		// Sort all pixels of the image by ascending order of pixel value
		{
			for (int i = 0; i<alls; i++) {
				data[i].second = i;
				data[i].first = imgPtr[i];
			}

			sort(data, data + alls);
		}

		// Find lower bound and upper bound of the pixel values
		double maxVal, minVal;
		minMaxLoc(img, &minVal, &maxVal);
		float maxRange = (float)(maxVal - minVal);
		float th = 1e-5f;

		float l = 0, r = maxRange * 2.0f / nI;

		// Perform binary search on error bound
		while (r - l > th) {
			float m = (r + l)*0.5f;
			bool suc = true;
			float base = (float)minVal;
			int cnt = 0;
			for (int i = 0; i<alls; i++) {
				if (data[i].first>base + m) {
					cnt++;
					base = data[i].first;
					if (cnt == nI) {
						suc = false;
						break;
					}
				}
			}
			if (suc)r = m;
			else l = m;
		}

		cv::Mat retImg(img.size(), CV_32SC1);
		int *retImgPtr = retImg.ptr<int>();

		// In the sorted list, divide pixel values into clusters according to the minimum error bound
		// Quantize each value to the median of its cluster
		// Also record the mapping of quantized value and quantized index.
		float base = (float)minVal;
		int baseI = 0;
		int cnt = 0;
		for (int i = 0; i <= alls; i++) {
			if (i == alls || data[i].first>base + r) {
				mapping[cnt] = data[(baseI + i - 1) >> 1].first; //median
				if (i == alls)break;
				cnt++;
				base = data[i].first;
				baseI = i;
			}
			retImgPtr[data[i].second] = cnt;
		}

		free(data);

		//end of the function
		outImg = retImg;
	}

	/***************************************************************/
	/* Function: from32STo32F
	* Description: convert the quantization index image back to the floating-point image accroding to the mapping
	/***************************************************************/
	static void from32STo32F(cv::Mat &img, cv::Mat &outImg, float *mapping) {

		cv::Mat retImg(img.size(), CV_32F);
		int rows = img.rows, cols = img.cols, alls = rows * cols;
		float *retImgPtr = retImg.ptr<float>();
		int *imgPtr = img.ptr<int>();

		// convert 32S index to 32F real value
		for (int i = 0; i<alls; i++) {
			retImgPtr[i] = mapping[imgPtr[i]];
		}

		// end of the function
		outImg = retImg;
	}


};

/***************************************************************/
/* Function: filter
*
* Description: filter implementation of joint-histogram weighted median framework
*				including clustering of feature image, adaptive quantization of input image.
*
* Input arguments:
*			I: input image (any # of channels). Accept only CV_32F and CV_8U type.
*	  feature: the feature image ("F" in the paper). Accept only CV_8UC1 and CV_8UC3 type (the # of channels should be 1 or 3).
*          r: radius of filtering kernel, should be a positive integer.
*      sigma: filter range standard deviation for the feature image.
*         nI: # of quantization level of input image. (only when the input image is CV_32F type)
*         nF: # of clusters of feature value. (only when the feature image is 3-channel)
*       iter: # of filtering times/iterations. (without changing the feature map)
* weightType: the type of weight definition, including:
*					exp: exp(-|I1-I2|^2/(2*sigma^2))
*					iv1: (|I1-I2|+sigma)^-1
*					iv2: (|I1-I2|^2+sigma^2)^-1
*					cos: dot(I1,I2)/(|I1|*|I2|)
*					jac: (min(r1,r2)+min(g1,g2)+min(b1,b2))/(max(r1,r2)+max(g1,g2)+max(b1,b2))
*					off: unweighted
*		 mask: a 0-1 mask that has the same size with I. This mask is used to ignore the effect of some pixels. If the pixel value on mask is 0,
*			   the pixel will be ignored when maintaining the joint-histogram. This is useful for applications like optical flow occlusion handling.
*
* Note:
*		1. When feature image clustering (when F is 3-channel) OR adaptive quantization (when I is floating point image) is
*         performed, the result is an approximation. To increase the accuracy, using a larger "nI" or "nF" will help.
*
*/
/***************************************************************/

/***************************************************************/
/* Function: filterCore
*
* Description: filter core implementation only containing joint-histogram weighted median framework
*
* input arguments:
*			I: input image. Only accept CV_32S type.
*          F: feature image. Only accept CV_32S type.
*       wMap: a 2D array that defines the distance between each pair of feature values. wMap[i][j] is the weight between feature value "i" and "j".
*          r: radius of filtering kernel, should be a positive integer.
*         nI: # of possible values in I, i.e., all values of I should in range [0, nI)
*         nF: # of possible values in F, i.e., all values of F should in range [0, nF)
*		 mask: a 0-1 mask that has the same size with I, for ignoring the effect of some pixels, as introduced in function "filter"
*/
/***************************************************************/

/***************************************************************/
/* Function: solveWeightedMedian
*
* Description: K(xi) = Σ_j qj wij |xi - yj| + (mu / pi - qi wij)|xi - yi| + lambda |xi - zi|	を最小化するxiを求める
*
* input arguments:
*			I: input image. Only accept CV_32S type.	yi
*          F: feature image. Only accept CV_32S type.
*       wMap: a 2D array that defines the distance between each pair of feature values. wMap[i][j] is the weight between feature value "i" and "j".
*          r: radius of filtering kernel, should be a positive integer.
*         nI: # of possible values in I, i.e., all values of I should in range [0, nI)
*         nF: # of possible values in F, i.e., all values of F should in range [0, nF)
*		p,q:
*		lambda,mu:
*		Z:初期推定値
*		countZ:解がZになることが確定している残り回数
*		countY:解がY(I)になるまでの残り回数
*		処理順は countZ > 0 ? (countY > 0 ? 中央値計算 : Z) : x = Y
*		でカウンターをデクリメントする。またY,Z確定時にもヒストグラムの更新などは必要
*/
/***************************************************************/
//ヒストグラムへの追加削除を重み付きに
//BCBへの追加削除も重み付きに　変更する
//とりあえず重みqの導入完了
//iterationがtrueなら　うんたら　あとで初回用とそれ以外プログラム分離するか検討
#endif
