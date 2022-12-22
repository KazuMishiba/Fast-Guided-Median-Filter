//#include "stdafx.h"
#include "l1solver.h"
//#include "opencv2/core/core.hpp"

#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
//using namespace cv;

//Iは初期推定値（Medianされる画像）Fは重み計算用画像（各視点3チャンネル画像）
cv::Mat l1Solver::filter(cv::Mat &I, cv::Mat &feature, int r, float sigma, int nI, int nF, int iter, string weightType, cv::Mat mask) {

	cv::Mat F = feature.clone();


	//check validation
	assert(I.depth() == CV_32S);
	assert(F.depth() == CV_8U && (F.channels() == 1 || F.channels() == 3));

	cv::Mat result;

	//Preprocess I
	//OUTPUT OF THIS STEP: Is, iMap 
	//If I is floating point image, "adaptive quantization" is done in from32FTo32S.
	//The mapping of floating value to integer value is stored in iMap (for each channel).
	//"Is" stores each channel of "I". The channels are converted to CV_32S type after this step.
	vector<cv::Mat> Is;
	split(I, Is);


	//Preprocess F
	//OUTPUT OF THIS STEP: F(new), wMap 
	//If "F" is 3-channel image, "clustering feature image" is done in featureIndexing.
	//If "F" is 1-channel image, featureIndexing only does a type-casting on "F".
	//The output "F" is CV_32S type, containing indexes of feature values.
	//"wMap" is a 2D array that defines the distance between each pair of feature indexes. 
	// wMap[i][j] is the weight between feature index "i" and "j".
	float **wMap;
	{
		featureIndexing(F, wMap, nF, sigma, weightType);
	}

	//Filtering - Joint-Histogram Framework
	{
		for (int i = 0; i<(int)Is.size(); i++) {
			{//Do filtering
				//Is[i] = filterCore(Is[i], F, wMap, r, nF, nI, mask);
				Is[i] = filterCoreParallel(Is[i], F, wMap, r, nF, nI, mask);
			}
		}
	}
	float2D_release(wMap);


	//merge the channels
	merge(Is, result);

	//end of the function
	return result;
}

//フィルタ連続してかける用　入力の適応的量子化はしない
cv::Mat l1Solver::filterIterative(cv::Mat &I, cv::Mat &feature, int r, float sigma, int nI, int nF, int iter, string weightType, cv::Mat mask) {

	cv::Mat F = feature.clone();
	cv::Mat result;
	I.convertTo(result, CV_32S);

	//check validation
	assert(I.depth() == CV_32F || I.depth() == CV_8U);
	assert(F.depth() == CV_8U && (F.channels() == 1 || F.channels() == 3));


	//Preprocess F
	//OUTPUT OF THIS STEP: F(new), wMap 
	//If "F" is 3-channel image, "clustering feature image" is done in featureIndexing.
	//If "F" is 1-channel image, featureIndexing only does a type-casting on "F".
	//The output "F" is CV_32S type, containing indexes of feature values.
	//"wMap" is a 2D array that defines the distance between each pair of feature indexes. 
	// wMap[i][j] is the weight between feature index "i" and "j".
	float **wMap;
	{
		featureIndexing(F, wMap, nF, sigma, weightType);
	}

	//Filtering - Joint-Histogram Framework
	{
		for (int k = 0; k < iter; k++)
		{//Do filtering
		 //result = filterCore(result, F, wMap, r, nF, nI, mask);
			result = filterCoreParallel(result, F, wMap, r, nF, nI, mask);
		}

	}
	float2D_release(wMap);

	result.convertTo(result, CV_8U);

	//end of the function
	return result;
}

cv::Mat l1Solver::filterCore(cv::Mat &I, cv::Mat &F, float **wMap, int r, int nF, int nI, cv::Mat mask) {

	 // Check validation
	 assert(I.depth() == CV_32S && I.channels() == 1);//input image: 32SC1
	 assert(F.depth() == CV_32S && F.channels() == 1);//feature image: 32SC1

													  // Configuration and declaration
	 int rows = I.rows, cols = I.cols;
	 int alls = rows * cols;
	 int winSize = (2 * r + 1)*(2 * r + 1);
	 cv::Mat outImg = I.clone();

	 // Handle Mask
	 if (mask.empty()) {
		 mask = cv::Mat(I.size(), CV_8U);
		 mask = cv::Scalar(1);
	 }

	 // Allocate memory for joint-histogram and BCB
	 int **H = int2D(nI, nF);
	 int *BCB = new int[nF];
	 // Hは画素index分×特徴index分、BCBは特徴index分の長さが確保されているが、そのindex順に並んでいるわけではない（necklace tableの構造なので）

	 // Allocate links for necklace table
	 int **Hf = int2D(nI, nF);//forward link
	 int **Hb = int2D(nI, nF);//backward link
	 int *BCBf = new int[nF];//forward link
	 int *BCBb = new int[nF];//backward link

							 // Column Scanning
							 //縦方向へスキャンし、下まで行ったら次の列処理対象に移動

	 for (int x = 0; x<cols; x++) {

		 // Reset histogram and BCB for each column
		 memset(BCB, 0, sizeof(int)*nF);
		 memset(H[0], 0, sizeof(int)*nF*nI);
		 for (int i = 0; i<nI; i++)Hf[i][0] = Hb[i][0] = 0;
		 BCBf[0] = BCBb[0] = 0;//BCBは0スタート

							   // Reset cut-point
		 int medianVal = -1;

		 // Precompute "x" range and checks boundary
		 int downX = max(0, x - r);
		 int upX = min(cols - 1, x + r);

		 // Initialize joint-histogram and BCB for the first window
		 {
			 //ウィンドウ内処理

			 int upY = min(rows - 1, r);
			 for (int i = 0; i <= upY; i++) {

				 int *IPtr = I.ptr<int>(i);//i行目の先頭画素のポインタを取得
				 int *FPtr = F.ptr<int>(i);
				 uchar *maskPtr = mask.ptr<uchar>(i);

				 for (int j = downX; j <= upX; j++) {

					 //もし対象画素(i,j)がマスク領域だったら処理を飛ばす
					 if (!maskPtr[j])continue;

					 int fval = IPtr[j];//対象画素値（index）
					 int *curHist = H[fval];//画素値indexがfval(対象画素値)のヒストグラムの先頭
					 int gval = FPtr[j];//対象画素の特徴量（index）

										// Maintain necklace table of joint-histogram
					 if (!curHist[gval] && gval) {
						 //第1項は、ヒストグラムが空だったら、なので、つまりHf,Hbにまだ加えていない、新たなindexが来たら、ということ
						 //第2項は、おそらくだがgval=0というか追跡時の？最初のindexは必ず操作するので加えていない
						 //index=0にNecklace tableのheadの役割を担わせる？ことによりヘッド一つ分要素を増やさずに済んでいるのでは
						 int *curHf = Hf[fval];//対象画素の画素indexがfvalのときのnecklace tableの先頭のポインタを取得
						 int *curHb = Hb[fval];
						 //curHf,curHbは1次元のnecklace tableとなる

						 int p1 = 0, p2 = curHf[0];
						 curHf[p1] = gval;
						 curHf[gval] = p2;
						 curHb[p2] = gval;
						 curHb[gval] = p1;
					 }

					 //H[gval][fval]の要素数を1増やす
					 curHist[gval]++;

					 // Maintain necklace table of BCB
					 updateBCB(BCB[gval], BCBf, BCBb, gval, -1);
				 }
			 }
		 }

		 //縦方向に処理
		 for (int y = 0; y<rows; y++) {

			 // Find weighted median with help of BCB and joint-histogram
			 {

				 float balanceWeight = 0;
				 int curIndex = F.ptr<int>(y, x)[0];//注目画素の特徴index
				 float *fPtr = wMap[curIndex];//重みマップの、片方の特徴がcurIndexのときの重み候補ベクトル
				 int &curMedianVal = medianVal;//前の中央値で現在の中央値を更新

											   // Compute current balance
				 int i = 0;
				 do {
					 //Σ B(f)*g(f_f, f(p))
					 //necklace tableを用いてデータのあるBCBについてのみ取り出して和を計算
					 //BCBfは次のデータのある場所を指示している
					 balanceWeight += BCB[i] * fPtr[i];
					 i = BCBf[i];
				 } while (i);

				 // Move cut-point to the left
				 if (balanceWeight >= 0) {
					 //カットポイント見つかるまで繰り返す
					 for (; balanceWeight >= 0 && curMedianVal; curMedianVal--) {
						 float curWeight = 0;
						 int *nextHist = H[curMedianVal];
						 int *nextHf = Hf[curMedianVal];

						 // Compute weight change by shift cut-point
						 int i = 0;
						 do {
							 //カットポイントを1つ横にずらすと、ずらすことによって、
							 //プラスしてたやつがマイナスに、またはマイナスしてたやつがプラスに
							 //転じるので、-w⇒+wと変更したい場合は2w足さないといけないので、<<1により2倍している
							 //ヒストグラムをfloatとかにすると多分これが使えない
							 //ここでやっているのはバランスbをカットポイントの移動に合わせて更新している
							 //ヒストグラムのカウント*重みが、式でいうH(i,f)*g(f,f)
							 curWeight += (nextHist[i] << 1)*fPtr[i];

							 // Update BCB and maintain the necklace table of BCB
							 updateBCB(BCB[i], BCBf, BCBb, i, -(nextHist[i] << 1));

							 i = nextHf[i];
						 } while (i);

						 balanceWeight -= curWeight;
					 }
				 }
				 // Move cut-point to the right
				 else if (balanceWeight < 0) {
					 for (; balanceWeight < 0 && curMedianVal != nI - 1; curMedianVal++) {
						 float curWeight = 0;
						 int *nextHist = H[curMedianVal + 1];
						 int *nextHf = Hf[curMedianVal + 1];

						 // Compute weight change by shift cut-point
						 int i = 0;
						 do {
							 curWeight += (nextHist[i] << 1)*fPtr[i];

							 // Update BCB and maintain the necklace table of BCB
							 updateBCB(BCB[i], BCBf, BCBb, i, nextHist[i] << 1);

							 i = nextHf[i];
						 } while (i);
						 balanceWeight += curWeight;
					 }
				 }
				 //負になったということはカットポイントを左に移動していったということなので、中央値はその一つ右のindexなのでcurMedianVal+1
				 //正の場合はその逆
				 //でメディアン結果入れる
				 // Weighted median is found and written to the output image
				 if (balanceWeight<0)outImg.ptr<int>(y, x)[0] = curMedianVal + 1;
				 else outImg.ptr<int>(y, x)[0] = curMedianVal;
			 }

			 // Update joint-histogram and BCB when local window is shifted.
			 {
				 int fval, gval, *curHist;
				 // Add entering pixels into joint-histogram and BCB
				 {
					 int rownum = y + r + 1;
					 if (rownum < rows) {
						 //次の行が存在するなら、更新
						 int *inputImgPtr = I.ptr<int>(rownum);
						 int *guideImgPtr = F.ptr<int>(rownum);
						 uchar *maskPtr = mask.ptr<uchar>(rownum);

						 for (int j = downX; j <= upX; j++) {

							 if (!maskPtr[j])continue;

							 fval = inputImgPtr[j];
							 curHist = H[fval];
							 gval = guideImgPtr[j];

							 // Maintain necklace table of joint-histogram
							 if (!curHist[gval] && gval) {//
								 int *curHf = Hf[fval];
								 int *curHb = Hb[fval];
								 //初期化と同じ処理のはずなのになんか書き方違う
								 //関数化してしまって良いのでは
								 int p1 = 0, p2 = curHf[0];
								 curHf[gval] = p2;
								 curHb[gval] = p1;
								 curHf[p1] = curHb[p2] = gval;
							 }

							 curHist[gval]++;

							 // Maintain necklace table of BCB
							 //追加対象の画素indexがカットポイントより左なら１足す、右なら１引く
							 //最後の引数がそれだが、なかなかトリッキー　0or1を2倍して-1することで、±1を作り出している
							 updateBCB(BCB[gval], BCBf, BCBb, gval, ((fval <= medianVal) << 1) - 1);
						 }
					 }
				 }

				 // Delete leaving pixels into joint-histogram and BCB
				 {
					 int rownum = y - r;
					 if (rownum >= 0) {

						 int *inputImgPtr = I.ptr<int>(rownum);
						 int *guideImgPtr = F.ptr<int>(rownum);
						 uchar *maskPtr = mask.ptr<uchar>(rownum);

						 for (int j = downX; j <= upX; j++) {

							 if (!maskPtr[j])continue;

							 fval = inputImgPtr[j];
							 curHist = H[fval];
							 gval = guideImgPtr[j];

							 //まずヒストグラムから削除
							 curHist[gval]--;

							 // Maintain necklace table of joint-histogram
							 if (!curHist[gval] && gval) {
								 //gval=0は必ず初めに見るので、gval=0のときはnecklace tableは特にいじらない
								 //gval≠0で、かつヒストグラムからの削除によってそのビンが空(=0)になったとき、
								 //necklace tableを更新しないといけないので、ここが実行される

								 //テーブルからindex gvalを削除したい
								 int *curHf = Hf[fval];//まずindex fvalの１Dテーブルを持ってくる
								 int *curHb = Hb[fval];
								 //ここで削除更新が行われている
								 //curHf[gval]はgvalがつながっている次（前？）のgval(index)を表している
								 //curHb[gval]は・・・
								 //まあ結局gvalを削除しているということで。
								 int p1 = curHb[gval], p2 = curHf[gval];
								 curHf[p1] = p2;
								 curHb[p2] = p1;
							 }

							 // Maintain necklace table of BCB
							 updateBCB(BCB[gval], BCBf, BCBb, gval, -((fval <= medianVal) << 1) + 1);
						 }
					 }
				 }
			 }
		 }

	 }

	 // Deallocate the memory
	 {
		 delete[]BCB;
		 delete[]BCBf;
		 delete[]BCBb;
		 int2D_release(H);
		 int2D_release(Hf);
		 int2D_release(Hb);
	 }

	 // end of the function
	 return outImg;
 }

cv::Mat l1Solver::filterCoreParallel(cv::Mat &I, cv::Mat &F, float **wMap, int r, int nF, int nI, cv::Mat mask) {

	 // Check validation
	 assert(I.depth() == CV_32S && I.channels() == 1);//input image: 32SC1
	 assert(F.depth() == CV_32S && F.channels() == 1);//feature image: 32SC1

													  // Configuration and declaration
	 int rows = I.rows, cols = I.cols;
	 int alls = rows * cols;
	 int winSize = (2 * r + 1)*(2 * r + 1);
	 cv::Mat outImg = I.clone();

	 // Handle Mask
	 if (mask.empty()) {
		 mask = cv::Mat(I.size(), CV_8U);
		 mask = cv::Scalar(1);
	 }


	 // Column Scanning
	 //縦方向へスキャンし、下まで行ったら次の列処理対象に移動
#ifndef _DEBUG
#pragma omp parallel for
#endif // !_DEBUG
	 for (int x = 0; x<cols; x++) {

		 // Allocate memory for joint-histogram and BCB
		 int **H = int2D(nI, nF);
		 int *BCB = new int[nF];
		 // Hは画素index分×特徴index分、BCBは特徴index分の長さが確保されているが、そのindex順に並んでいるわけではない（necklace tableの構造なので）

		 // Allocate links for necklace table
		 int **Hf = int2D(nI, nF);//forward link
		 int **Hb = int2D(nI, nF);//backward link
		 int *BCBf = new int[nF];//forward link
		 int *BCBb = new int[nF];//backward link

								 // Reset histogram and BCB for each column
		 memset(BCB, 0, sizeof(int)*nF);
		 memset(H[0], 0, sizeof(int)*nF*nI);
		 for (int i = 0; i<nI; i++)Hf[i][0] = Hb[i][0] = 0;
		 BCBf[0] = BCBb[0] = 0;//BCBは0スタート

							   // Reset cut-point
		 int medianVal = -1;

		 // Precompute "x" range and checks boundary
		 int downX = max(0, x - r);
		 int upX = min(cols - 1, x + r);

		 // Initialize joint-histogram and BCB for the first window
		 {
			 //ウィンドウ内処理

			 int upY = min(rows - 1, r);
			 for (int i = 0; i <= upY; i++) {

				 int *IPtr = I.ptr<int>(i);//i行目の先頭画素のポインタを取得
				 int *FPtr = F.ptr<int>(i);
				 uchar *maskPtr = mask.ptr<uchar>(i);

				 for (int j = downX; j <= upX; j++) {

					 //もし対象画素(i,j)がマスク領域だったら処理を飛ばす
					 if (!maskPtr[j])continue;

					 int fval = IPtr[j];//対象画素値（index）
					 int *curHist = H[fval];//画素値indexがfval(対象画素値)のヒストグラムの先頭
					 int gval = FPtr[j];//対象画素の特徴量（index）

										// Maintain necklace table of joint-histogram
					 if (!curHist[gval] && gval) {
						 //第1項は、ヒストグラムが空だったら、なので、つまりHf,Hbにまだ加えていない、新たなindexが来たら、ということ
						 //第2項は、おそらくだがgval=0というか追跡時の？最初のindexは必ず操作するので加えていない
						 //index=0にNecklace tableのheadの役割を担わせる？ことによりヘッド一つ分要素を増やさずに済んでいるのでは
						 int *curHf = Hf[fval];//対象画素の画素indexがfvalのときのnecklace tableの先頭のポインタを取得
						 int *curHb = Hb[fval];
						 //curHf,curHbは1次元のnecklace tableとなる

						 int p1 = 0, p2 = curHf[0];
						 curHf[p1] = gval;
						 curHf[gval] = p2;
						 curHb[p2] = gval;
						 curHb[gval] = p1;
					 }

					 //H[gval][fval]の要素数を1増やす
					 curHist[gval]++;

					 // Maintain necklace table of BCB
					 updateBCB(BCB[gval], BCBf, BCBb, gval, -1);
				 }
			 }
		 }

		 //縦方向に処理
		 for (int y = 0; y<rows; y++) {

			 // Find weighted median with help of BCB and joint-histogram
			 {

				 float balanceWeight = 0;
				 int curIndex = F.ptr<int>(y, x)[0];//注目画素の特徴index
				 float *fPtr = wMap[curIndex];//重みマップの、片方の特徴がcurIndexのときの重み候補ベクトル
				 int &curMedianVal = medianVal;//前の中央値で現在の中央値を更新

											   // Compute current balance
				 int i = 0;
				 do {
					 //Σ B(f)*g(f_f, f(p))
					 //necklace tableを用いてデータのあるBCBについてのみ取り出して和を計算
					 //BCBfは次のデータのある場所を指示している
					 balanceWeight += BCB[i] * fPtr[i];
					 i = BCBf[i];
				 } while (i);

				 // Move cut-point to the left
				 if (balanceWeight >= 0) {
					 //カットポイント見つかるまで繰り返す
					 for (; balanceWeight >= 0 && curMedianVal; curMedianVal--) {
						 float curWeight = 0;
						 int *nextHist = H[curMedianVal];
						 int *nextHf = Hf[curMedianVal];

						 // Compute weight change by shift cut-point
						 int i = 0;
						 do {
							 //カットポイントを1つ横にずらすと、ずらすことによって、
							 //プラスしてたやつがマイナスに、またはマイナスしてたやつがプラスに
							 //転じるので、-w⇒+wと変更したい場合は2w足さないといけないので、<<1により2倍している
							 //ヒストグラムをfloatとかにすると多分これが使えない
							 //ここでやっているのはバランスbをカットポイントの移動に合わせて更新している
							 //ヒストグラムのカウント*重みが、式でいうH(i,f)*g(f,f)
							 curWeight += (nextHist[i] << 1)*fPtr[i];

							 // Update BCB and maintain the necklace table of BCB
							 updateBCB(BCB[i], BCBf, BCBb, i, -(nextHist[i] << 1));

							 i = nextHf[i];
						 } while (i);

						 balanceWeight -= curWeight;
					 }
				 }
				 // Move cut-point to the right
				 else if (balanceWeight < 0) {
					 for (; balanceWeight < 0 && curMedianVal != nI - 1; curMedianVal++) {
						 float curWeight = 0;
						 int *nextHist = H[curMedianVal + 1];
						 int *nextHf = Hf[curMedianVal + 1];

						 // Compute weight change by shift cut-point
						 int i = 0;
						 do {
							 curWeight += (nextHist[i] << 1)*fPtr[i];

							 // Update BCB and maintain the necklace table of BCB
							 updateBCB(BCB[i], BCBf, BCBb, i, nextHist[i] << 1);

							 i = nextHf[i];
						 } while (i);
						 balanceWeight += curWeight;
					 }
				 }
				 //負になったということはカットポイントを左に移動していったということなので、中央値はその一つ右のindexなのでcurMedianVal+1
				 //正の場合はその逆
				 //でメディアン結果入れる
				 // Weighted median is found and written to the output image
				 if (balanceWeight<0)outImg.ptr<int>(y, x)[0] = curMedianVal + 1;
				 else outImg.ptr<int>(y, x)[0] = curMedianVal;
			 }

			 // Update joint-histogram and BCB when local window is shifted.
			 {
				 int fval, gval, *curHist;
				 // Add entering pixels into joint-histogram and BCB
				 {
					 int rownum = y + r + 1;
					 if (rownum < rows) {
						 //次の行が存在するなら、更新
						 int *inputImgPtr = I.ptr<int>(rownum);
						 int *guideImgPtr = F.ptr<int>(rownum);
						 uchar *maskPtr = mask.ptr<uchar>(rownum);

						 for (int j = downX; j <= upX; j++) {

							 if (!maskPtr[j])continue;

							 fval = inputImgPtr[j];
							 curHist = H[fval];
							 gval = guideImgPtr[j];

							 // Maintain necklace table of joint-histogram
							 if (!curHist[gval] && gval) {//
								 int *curHf = Hf[fval];
								 int *curHb = Hb[fval];
								 //初期化と同じ処理のはずなのになんか書き方違う
								 //関数化してしまって良いのでは
								 int p1 = 0, p2 = curHf[0];
								 curHf[gval] = p2;
								 curHb[gval] = p1;
								 curHf[p1] = curHb[p2] = gval;
							 }

							 curHist[gval]++;

							 // Maintain necklace table of BCB
							 //追加対象の画素indexがカットポイントより左なら１足す、右なら１引く
							 //最後の引数がそれだが、なかなかトリッキー　0or1を2倍して-1することで、±1を作り出している
							 updateBCB(BCB[gval], BCBf, BCBb, gval, ((fval <= medianVal) << 1) - 1);
						 }
					 }
				 }

				 // Delete leaving pixels into joint-histogram and BCB
				 {
					 int rownum = y - r;
					 if (rownum >= 0) {

						 int *inputImgPtr = I.ptr<int>(rownum);
						 int *guideImgPtr = F.ptr<int>(rownum);
						 uchar *maskPtr = mask.ptr<uchar>(rownum);

						 for (int j = downX; j <= upX; j++) {

							 if (!maskPtr[j])continue;

							 fval = inputImgPtr[j];
							 curHist = H[fval];
							 gval = guideImgPtr[j];

							 //まずヒストグラムから削除
							 curHist[gval]--;

							 // Maintain necklace table of joint-histogram
							 if (!curHist[gval] && gval) {
								 //gval=0は必ず初めに見るので、gval=0のときはnecklace tableは特にいじらない
								 //gval≠0で、かつヒストグラムからの削除によってそのビンが空(=0)になったとき、
								 //necklace tableを更新しないといけないので、ここが実行される

								 //テーブルからindex gvalを削除したい
								 int *curHf = Hf[fval];//まずindex fvalの１Dテーブルを持ってくる
								 int *curHb = Hb[fval];
								 //ここで削除更新が行われている
								 //curHf[gval]はgvalがつながっている次（前？）のgval(index)を表している
								 //curHb[gval]は・・・
								 //まあ結局gvalを削除しているということで。
								 int p1 = curHb[gval], p2 = curHf[gval];
								 curHf[p1] = p2;
								 curHb[p2] = p1;
							 }

							 // Maintain necklace table of BCB
							 updateBCB(BCB[gval], BCBf, BCBb, gval, -((fval <= medianVal) << 1) + 1);
						 }
					 }
				 }
			 }
		 }

		 // Deallocate the memory
		 {
			 delete[]BCB;
			 delete[]BCBf;
			 delete[]BCBb;
			 int2D_release(H);
			 int2D_release(Hf);
			 int2D_release(Hb);
		 }
	 }


	 // end of the function
	 return outImg;
 }



