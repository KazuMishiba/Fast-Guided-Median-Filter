//#include "stdafx.h"
#include "l1solver.h"
//#include "opencv2/core/core.hpp"

#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
//using namespace cv;

//I�͏�������l�iMedian�����摜�jF�͏d�݌v�Z�p�摜�i�e���_3�`�����l���摜�j
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

//�t�B���^�A�����Ă�����p�@���͂̓K���I�ʎq���͂��Ȃ�
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
	 // H�͉�findex���~����index���ABCB�͓���index���̒������m�ۂ���Ă��邪�A����index���ɕ���ł���킯�ł͂Ȃ��inecklace table�̍\���Ȃ̂Łj

	 // Allocate links for necklace table
	 int **Hf = int2D(nI, nF);//forward link
	 int **Hb = int2D(nI, nF);//backward link
	 int *BCBf = new int[nF];//forward link
	 int *BCBb = new int[nF];//backward link

							 // Column Scanning
							 //�c�����փX�L�������A���܂ōs�����玟�̗񏈗��ΏۂɈړ�

	 for (int x = 0; x<cols; x++) {

		 // Reset histogram and BCB for each column
		 memset(BCB, 0, sizeof(int)*nF);
		 memset(H[0], 0, sizeof(int)*nF*nI);
		 for (int i = 0; i<nI; i++)Hf[i][0] = Hb[i][0] = 0;
		 BCBf[0] = BCBb[0] = 0;//BCB��0�X�^�[�g

							   // Reset cut-point
		 int medianVal = -1;

		 // Precompute "x" range and checks boundary
		 int downX = max(0, x - r);
		 int upX = min(cols - 1, x + r);

		 // Initialize joint-histogram and BCB for the first window
		 {
			 //�E�B���h�E������

			 int upY = min(rows - 1, r);
			 for (int i = 0; i <= upY; i++) {

				 int *IPtr = I.ptr<int>(i);//i�s�ڂ̐擪��f�̃|�C���^���擾
				 int *FPtr = F.ptr<int>(i);
				 uchar *maskPtr = mask.ptr<uchar>(i);

				 for (int j = downX; j <= upX; j++) {

					 //�����Ώۉ�f(i,j)���}�X�N�̈悾�����珈�����΂�
					 if (!maskPtr[j])continue;

					 int fval = IPtr[j];//�Ώۉ�f�l�iindex�j
					 int *curHist = H[fval];//��f�lindex��fval(�Ώۉ�f�l)�̃q�X�g�O�����̐擪
					 int gval = FPtr[j];//�Ώۉ�f�̓����ʁiindex�j

										// Maintain necklace table of joint-histogram
					 if (!curHist[gval] && gval) {
						 //��1���́A�q�X�g�O�������󂾂�����A�Ȃ̂ŁA�܂�Hf,Hb�ɂ܂������Ă��Ȃ��A�V����index��������A�Ƃ�������
						 //��2���́A�����炭����gval=0�Ƃ������ǐՎ��́H�ŏ���index�͕K�����삷��̂ŉ����Ă��Ȃ�
						 //index=0��Necklace table��head�̖�����S�킹��H���Ƃɂ��w�b�h����v�f�𑝂₳���ɍς�ł���̂ł�
						 int *curHf = Hf[fval];//�Ώۉ�f�̉�findex��fval�̂Ƃ���necklace table�̐擪�̃|�C���^���擾
						 int *curHb = Hb[fval];
						 //curHf,curHb��1������necklace table�ƂȂ�

						 int p1 = 0, p2 = curHf[0];
						 curHf[p1] = gval;
						 curHf[gval] = p2;
						 curHb[p2] = gval;
						 curHb[gval] = p1;
					 }

					 //H[gval][fval]�̗v�f����1���₷
					 curHist[gval]++;

					 // Maintain necklace table of BCB
					 updateBCB(BCB[gval], BCBf, BCBb, gval, -1);
				 }
			 }
		 }

		 //�c�����ɏ���
		 for (int y = 0; y<rows; y++) {

			 // Find weighted median with help of BCB and joint-histogram
			 {

				 float balanceWeight = 0;
				 int curIndex = F.ptr<int>(y, x)[0];//���ډ�f�̓���index
				 float *fPtr = wMap[curIndex];//�d�݃}�b�v�́A�Е��̓�����curIndex�̂Ƃ��̏d�݌��x�N�g��
				 int &curMedianVal = medianVal;//�O�̒����l�Ō��݂̒����l���X�V

											   // Compute current balance
				 int i = 0;
				 do {
					 //�� B(f)*g(f_f, f(p))
					 //necklace table��p���ăf�[�^�̂���BCB�ɂ��Ă̂ݎ��o���Ęa���v�Z
					 //BCBf�͎��̃f�[�^�̂���ꏊ���w�����Ă���
					 balanceWeight += BCB[i] * fPtr[i];
					 i = BCBf[i];
				 } while (i);

				 // Move cut-point to the left
				 if (balanceWeight >= 0) {
					 //�J�b�g�|�C���g������܂ŌJ��Ԃ�
					 for (; balanceWeight >= 0 && curMedianVal; curMedianVal--) {
						 float curWeight = 0;
						 int *nextHist = H[curMedianVal];
						 int *nextHf = Hf[curMedianVal];

						 // Compute weight change by shift cut-point
						 int i = 0;
						 do {
							 //�J�b�g�|�C���g��1���ɂ��炷�ƁA���炷���Ƃɂ���āA
							 //�v���X���Ă�����}�C�i�X�ɁA�܂��̓}�C�i�X���Ă�����v���X��
							 //�]����̂ŁA-w��+w�ƕύX�������ꍇ��2w�����Ȃ��Ƃ����Ȃ��̂ŁA<<1�ɂ��2�{���Ă���
							 //�q�X�g�O������float�Ƃ��ɂ���Ƒ������ꂪ�g���Ȃ�
							 //�����ł���Ă���̂̓o�����Xb���J�b�g�|�C���g�̈ړ��ɍ��킹�čX�V���Ă���
							 //�q�X�g�O�����̃J�E���g*�d�݂��A���ł���H(i,f)*g(f,f)
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
				 //���ɂȂ����Ƃ������Ƃ̓J�b�g�|�C���g�����Ɉړ����Ă������Ƃ������ƂȂ̂ŁA�����l�͂��̈�E��index�Ȃ̂�curMedianVal+1
				 //���̏ꍇ�͂��̋t
				 //�Ń��f�B�A�����ʓ����
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
						 //���̍s�����݂���Ȃ�A�X�V
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
								 //�������Ɠ��������̂͂��Ȃ̂ɂȂ񂩏������Ⴄ
								 //�֐������Ă��܂��ėǂ��̂ł�
								 int p1 = 0, p2 = curHf[0];
								 curHf[gval] = p2;
								 curHb[gval] = p1;
								 curHf[p1] = curHb[p2] = gval;
							 }

							 curHist[gval]++;

							 // Maintain necklace table of BCB
							 //�ǉ��Ώۂ̉�findex���J�b�g�|�C���g��荶�Ȃ�P�����A�E�Ȃ�P����
							 //�Ō�̈��������ꂾ���A�Ȃ��Ȃ��g���b�L�[�@0or1��2�{����-1���邱�ƂŁA�}1�����o���Ă���
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

							 //�܂��q�X�g�O��������폜
							 curHist[gval]--;

							 // Maintain necklace table of joint-histogram
							 if (!curHist[gval] && gval) {
								 //gval=0�͕K�����߂Ɍ���̂ŁAgval=0�̂Ƃ���necklace table�͓��ɂ�����Ȃ�
								 //gval��0�ŁA���q�X�g�O��������̍폜�ɂ���Ă��̃r������(=0)�ɂȂ����Ƃ��A
								 //necklace table���X�V���Ȃ��Ƃ����Ȃ��̂ŁA���������s�����

								 //�e�[�u������index gval���폜������
								 int *curHf = Hf[fval];//�܂�index fval�̂PD�e�[�u���������Ă���
								 int *curHb = Hb[fval];
								 //�����ō폜�X�V���s���Ă���
								 //curHf[gval]��gval���Ȃ����Ă��鎟�i�O�H�j��gval(index)��\���Ă���
								 //curHb[gval]�́E�E�E
								 //�܂�����gval���폜���Ă���Ƃ������ƂŁB
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
	 //�c�����փX�L�������A���܂ōs�����玟�̗񏈗��ΏۂɈړ�
#ifndef _DEBUG
#pragma omp parallel for
#endif // !_DEBUG
	 for (int x = 0; x<cols; x++) {

		 // Allocate memory for joint-histogram and BCB
		 int **H = int2D(nI, nF);
		 int *BCB = new int[nF];
		 // H�͉�findex���~����index���ABCB�͓���index���̒������m�ۂ���Ă��邪�A����index���ɕ���ł���킯�ł͂Ȃ��inecklace table�̍\���Ȃ̂Łj

		 // Allocate links for necklace table
		 int **Hf = int2D(nI, nF);//forward link
		 int **Hb = int2D(nI, nF);//backward link
		 int *BCBf = new int[nF];//forward link
		 int *BCBb = new int[nF];//backward link

								 // Reset histogram and BCB for each column
		 memset(BCB, 0, sizeof(int)*nF);
		 memset(H[0], 0, sizeof(int)*nF*nI);
		 for (int i = 0; i<nI; i++)Hf[i][0] = Hb[i][0] = 0;
		 BCBf[0] = BCBb[0] = 0;//BCB��0�X�^�[�g

							   // Reset cut-point
		 int medianVal = -1;

		 // Precompute "x" range and checks boundary
		 int downX = max(0, x - r);
		 int upX = min(cols - 1, x + r);

		 // Initialize joint-histogram and BCB for the first window
		 {
			 //�E�B���h�E������

			 int upY = min(rows - 1, r);
			 for (int i = 0; i <= upY; i++) {

				 int *IPtr = I.ptr<int>(i);//i�s�ڂ̐擪��f�̃|�C���^���擾
				 int *FPtr = F.ptr<int>(i);
				 uchar *maskPtr = mask.ptr<uchar>(i);

				 for (int j = downX; j <= upX; j++) {

					 //�����Ώۉ�f(i,j)���}�X�N�̈悾�����珈�����΂�
					 if (!maskPtr[j])continue;

					 int fval = IPtr[j];//�Ώۉ�f�l�iindex�j
					 int *curHist = H[fval];//��f�lindex��fval(�Ώۉ�f�l)�̃q�X�g�O�����̐擪
					 int gval = FPtr[j];//�Ώۉ�f�̓����ʁiindex�j

										// Maintain necklace table of joint-histogram
					 if (!curHist[gval] && gval) {
						 //��1���́A�q�X�g�O�������󂾂�����A�Ȃ̂ŁA�܂�Hf,Hb�ɂ܂������Ă��Ȃ��A�V����index��������A�Ƃ�������
						 //��2���́A�����炭����gval=0�Ƃ������ǐՎ��́H�ŏ���index�͕K�����삷��̂ŉ����Ă��Ȃ�
						 //index=0��Necklace table��head�̖�����S�킹��H���Ƃɂ��w�b�h����v�f�𑝂₳���ɍς�ł���̂ł�
						 int *curHf = Hf[fval];//�Ώۉ�f�̉�findex��fval�̂Ƃ���necklace table�̐擪�̃|�C���^���擾
						 int *curHb = Hb[fval];
						 //curHf,curHb��1������necklace table�ƂȂ�

						 int p1 = 0, p2 = curHf[0];
						 curHf[p1] = gval;
						 curHf[gval] = p2;
						 curHb[p2] = gval;
						 curHb[gval] = p1;
					 }

					 //H[gval][fval]�̗v�f����1���₷
					 curHist[gval]++;

					 // Maintain necklace table of BCB
					 updateBCB(BCB[gval], BCBf, BCBb, gval, -1);
				 }
			 }
		 }

		 //�c�����ɏ���
		 for (int y = 0; y<rows; y++) {

			 // Find weighted median with help of BCB and joint-histogram
			 {

				 float balanceWeight = 0;
				 int curIndex = F.ptr<int>(y, x)[0];//���ډ�f�̓���index
				 float *fPtr = wMap[curIndex];//�d�݃}�b�v�́A�Е��̓�����curIndex�̂Ƃ��̏d�݌��x�N�g��
				 int &curMedianVal = medianVal;//�O�̒����l�Ō��݂̒����l���X�V

											   // Compute current balance
				 int i = 0;
				 do {
					 //�� B(f)*g(f_f, f(p))
					 //necklace table��p���ăf�[�^�̂���BCB�ɂ��Ă̂ݎ��o���Ęa���v�Z
					 //BCBf�͎��̃f�[�^�̂���ꏊ���w�����Ă���
					 balanceWeight += BCB[i] * fPtr[i];
					 i = BCBf[i];
				 } while (i);

				 // Move cut-point to the left
				 if (balanceWeight >= 0) {
					 //�J�b�g�|�C���g������܂ŌJ��Ԃ�
					 for (; balanceWeight >= 0 && curMedianVal; curMedianVal--) {
						 float curWeight = 0;
						 int *nextHist = H[curMedianVal];
						 int *nextHf = Hf[curMedianVal];

						 // Compute weight change by shift cut-point
						 int i = 0;
						 do {
							 //�J�b�g�|�C���g��1���ɂ��炷�ƁA���炷���Ƃɂ���āA
							 //�v���X���Ă�����}�C�i�X�ɁA�܂��̓}�C�i�X���Ă�����v���X��
							 //�]����̂ŁA-w��+w�ƕύX�������ꍇ��2w�����Ȃ��Ƃ����Ȃ��̂ŁA<<1�ɂ��2�{���Ă���
							 //�q�X�g�O������float�Ƃ��ɂ���Ƒ������ꂪ�g���Ȃ�
							 //�����ł���Ă���̂̓o�����Xb���J�b�g�|�C���g�̈ړ��ɍ��킹�čX�V���Ă���
							 //�q�X�g�O�����̃J�E���g*�d�݂��A���ł���H(i,f)*g(f,f)
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
				 //���ɂȂ����Ƃ������Ƃ̓J�b�g�|�C���g�����Ɉړ����Ă������Ƃ������ƂȂ̂ŁA�����l�͂��̈�E��index�Ȃ̂�curMedianVal+1
				 //���̏ꍇ�͂��̋t
				 //�Ń��f�B�A�����ʓ����
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
						 //���̍s�����݂���Ȃ�A�X�V
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
								 //�������Ɠ��������̂͂��Ȃ̂ɂȂ񂩏������Ⴄ
								 //�֐������Ă��܂��ėǂ��̂ł�
								 int p1 = 0, p2 = curHf[0];
								 curHf[gval] = p2;
								 curHb[gval] = p1;
								 curHf[p1] = curHb[p2] = gval;
							 }

							 curHist[gval]++;

							 // Maintain necklace table of BCB
							 //�ǉ��Ώۂ̉�findex���J�b�g�|�C���g��荶�Ȃ�P�����A�E�Ȃ�P����
							 //�Ō�̈��������ꂾ���A�Ȃ��Ȃ��g���b�L�[�@0or1��2�{����-1���邱�ƂŁA�}1�����o���Ă���
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

							 //�܂��q�X�g�O��������폜
							 curHist[gval]--;

							 // Maintain necklace table of joint-histogram
							 if (!curHist[gval] && gval) {
								 //gval=0�͕K�����߂Ɍ���̂ŁAgval=0�̂Ƃ���necklace table�͓��ɂ�����Ȃ�
								 //gval��0�ŁA���q�X�g�O��������̍폜�ɂ���Ă��̃r������(=0)�ɂȂ����Ƃ��A
								 //necklace table���X�V���Ȃ��Ƃ����Ȃ��̂ŁA���������s�����

								 //�e�[�u������index gval���폜������
								 int *curHf = Hf[fval];//�܂�index fval�̂PD�e�[�u���������Ă���
								 int *curHb = Hb[fval];
								 //�����ō폜�X�V���s���Ă���
								 //curHf[gval]��gval���Ȃ����Ă��鎟�i�O�H�j��gval(index)��\���Ă���
								 //curHb[gval]�́E�E�E
								 //�܂�����gval���폜���Ă���Ƃ������ƂŁB
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



