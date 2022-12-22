#pragma once
#include "FGMF_base.h"
/*
�i�P�j�X�p�[�X�q�X�g�O�����{�\�[�g
high-precision�f�[�^�ɑΉ����邽�߂̕���

�����̍œK�����Ȃ���Ă��Ȃ��̂�����Ƃ͎v�����A
�ǐՂ����肵�Ȃ��ق���10%�����B
���ƁA�ǐՂ̂�͎����̂ǂ��������������āA���ʂ����S�Ɉ�v���Ȃ��B

*/



class FGMF1
{
public:
	//fgSumUpToIndex�Ȃ�
	template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);

	//�ǐՂ���Łi�����l�T�����Ƀm�[�h1�̏ꍇ�ɓ����Ȃ��o�O����j
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D_2(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);

	//cxdx�ۑ�
	template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D_saveCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);
	//cxdx�g�p
	template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D_useCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);


	//Thread�����w��
	static cv::Mat filter2DInterface(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2 = 25.5f * 25.5f, int Imax = 256);



	//Single Thread
	//I1
	static cv::Mat filter2DI1G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax); 
	static cv::Mat FGMF1::filter2DI1G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	//I3
	static cv::Mat filter2DI3G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	static cv::Mat FGMF1::filter2DI3G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);

	//Multi Thread
	//cols (������)
	//I1
	static cv::Mat filter2DI1G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	static cv::Mat filter2DI1G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	//I3
	static cv::Mat filter2DI3G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	static cv::Mat filter2DI3G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);


};











////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
//�ǐՂ���Łi�ǐՖ����̕��������j
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF1::filter2D_2(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
{
	//check validation
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && ((std::is_same<GTYPE, int>::value && G.channels() == 1) || (std::is_same<GTYPE, cv::Vec3i>::value && G.channels() == 3)));
	assert(I.isContinuous() && G.isContinuous());
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//�����l
	const float half = 0.5f;

	//������
	const int size_dim0 = I.cols;
	const int size_dim1 = I.rows;

	//
	const int r_dim0 = radius;
	const int r_dim1 = radius;

	const int maxPixelNum = (radius * 2 + 1) * (radius * 2 + 1);

	/*
	* �ȉ��͈̔͂́`X�@��X���܂ށiX�ȉ��j
	* �����Ώۉ�f�͈͂� [dim1Start �` dim1End, dim0Start �` dim0End]
	* �E�B���h�E�͈͂�[max(0, dim1Start - r_dim1) �` (std::min)(dim1End, size_dim1) , [max(0, dim0Start) �` (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1�����E�B���h�E��[�A���[
	const int upmostWindowDim1 = std::max(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0�����̃E�B���h�E���[�A�E�[
	const int leftmostWindowDim0 = std::max(0, dim0Start - r_dim0);
	const int rightmostWindowDim0 = std::min(size_dim0 - 1, dim0End + r_dim0);
	//��������
	const int memoryLength = rightmostWindowDim0 - leftmostWindowDim0 + 1;
	//�P�s�ڗp dim1�����E�B���h�E���[
	const int bottomWindowDim1ForFirstLine = std::min(size_dim1 - 1, dim1Start + r_dim1);
	//�P��ڗp dim0�����E�B���h�E�E�[
	const int rightWindowDim0ForFirstLine = std::min(size_dim0 - 1, dim0Start + r_dim0);
	//(1,X)�̎��̃E�B���h�E��
	int pixel_num_dim0_firstLine = rightWindowDim0ForFirstLine - leftmostWindowDim0 + 1;


	//�������m�ہE������

	//�e��p�q�X�g�O�����i�[�ϐ�
	std::list<FG>* histo_win1 = new std::list<FG> [memoryLength];
	//sum(g), sum(g*g)�����v�Z�p�ϐ�
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);
	//index_win1�ȉ��̗�q�X�g�O�����a
	FGSumUpToIndex* sumUpToIndex_win1 = new FGSumUpToIndex[memoryLength];
	memset(sumUpToIndex_win1, 0, sizeof(FGSumUpToIndex) * memoryLength);

	//Window���q�X�g�O����
	std::list<FG> histo_window;



	///////////////////////////////
	//�Ή������f�ւ̃|�C���^
	//�������̉�f
	GTYPE* G_center = G.ptr<GTYPE>(dim1Start) + dim0Start;
	int* result_center = result.ptr<int>(dim1Start) + dim0Start;
	//�|�C���^�W�����v��
	const int stepForNextRow = size_dim0 - (dim0End - dim0Start + 1);
	const int stepForNextRow2 = stepForNextRow + 1;
	///////////////////////////////


	//1�s�ڂ����ʏ���
	{
		//������
		//Window�֌W
		GSum gSum_window = { 0 };
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(dim1Start);

		//�E�B���h�E�q�X�g�O����������
		//memset(histo_window, 0, sizeof(FG) * Imax);
		histo_window.clear();
		//1�񓖂���̉�f��
		int pixel_sum_dim1 = bottomWindowDim1ForFirstLine - upmostWindowDim1 + 1;
		//�E�B���h�E����f��
		int pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;



		//�������p
		int x_add_mem = 0;
		int x_rem_mem = 0;

		//����ɂP��ڂ����ʏ���
		//(1,1)����
		{
			// leftmostWindowDim0�`rightWindowDim1ForFirstLine �̃q�X�g�O�������\�z
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				//1�s�ڗp�����@�񂷂ׂĒǉ�
				for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx], gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}
			//�����l�̌v�Z
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				G_center++;
				result_center++;
			}
		}
		//(2,1)�`
		{
			int x_add_dim0 = r_dim0 + 1 + dim0Start;// �ǉ�������
			int x_rem_dim0 = -r_dim0 + dim0Start;// �폜������
			int x_dim0 = 1 + dim0Start;// �����Ώے��S��f��

			for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
			{
				//�ǉ����鎟�̗񂪂��邩
				const int hasAdd_dim0 = x_add_dim0 < size_dim0;
				//�폜����O�̗񂪂��邩
				const int hasRem_dim0 = x_rem_dim0 >= 0;
				//�����ʒu���摜����
				const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
				const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

				//window����f���̍X�V
				if (hasAddOnly_dim0)
					addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
				else if (hasRemOnly_dim0)
					subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�

				//���̗񂪂���Ȃ�X�V
				if (hasAdd_dim0) {
					//1�s�ڗp�����@�񂷂ׂĒǉ�
					for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//���̍s���q�X�g�O�����ɒǉ�
						addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0], gSum_win1[x_add_mem]);

					//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
					updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
					x_add_mem++;
				}
				//�폜�񂪂���Ȃ�
				if (hasRem_dim0)
				{
					//���C��window����subwindow�̃q�X�g�O�������폜
					removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//�����l�̌v�Z
				{
					findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
					G_center++;
					result_center++;
				}
			}
			G_center += stepForNextRow;
			result_center += stepForNextRow;
		}
	}
	//������
	int x_add_dim1 = r_dim1 + 1 + dim1Start;//�ǉ�������
	int x_rem_dim1 = -r_dim1 + dim1Start;//�폜�����s
	int x_dim1 = 1 + dim1Start;// �����Ώے��S��f�s
	//
	//��̍폜��f
	int* I_rem = I.ptr<int>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_rem = G.ptr<GTYPE>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	//��̒ǉ���f
	int* I_add = I.ptr<int>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_add = G.ptr<GTYPE>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;

	/*
	I_rem = I.ptr<int>(x_dim1 - r_dim1 - 1)[x_add_dim0]
	I_add = I.ptr<int>(x_dim1 + r_dim1)[x_add_dim0]

	*/

	//2�s�ڈȍ~�����J�n
	for (; x_dim1 <= dim1End; x_dim1++, x_add_dim1++, x_rem_dim1++)
	{
		const int hasAdd_dim1 = x_add_dim1 < size_dim1;
		const int hasRem_dim1 = x_rem_dim1 >= 0;
		const int hasAddOnly_dim1 = !hasRem_dim1 && hasAdd_dim1;
		const int hasRemOnly_dim1 = hasRem_dim1 && !hasAdd_dim1;

		//������
		//Window�֌W
		GSum gSum_window = { 0 };
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(x_dim1);

		//�E�B���h�E�q�X�g�O����������
		//memset(histo_window, 0, sizeof(FG) * Imax);
		histo_window.clear();
		//�E�B���h�E����f��
		int pixel_sum_window = 0;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 0.0f;
		//1�񓖂���̉�f��
		int pixel_sum_dim1;

		if (hasAddOnly_dim1)
			pixel_sum_dim1 = x_dim1 + r_dim1 + 1;
		else if (hasRemOnly_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + r_dim1;
		else
			pixel_sum_dim1 = 2 * r_dim1 + 1;



		//�������p
		int x_add_mem = 0;
		int x_rem_mem = 0;


		//1��ڏ���
		{
			//�E�B���h�E����f��
			pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
			//�E�B���h�E����f���̋t��
			pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;

			// leftmostWindowDim0�`rightWindowDim1ForFirstLine �̃q�X�g�O�������\�z
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				if (hasRem_dim1)//�O�̍s���q�X�g�O��������폜
					removePixelFromWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx], gSum_win1[x_add_mem]);
				if (hasAdd_dim1)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx], gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//�����l�̌v�Z
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				G_center++;
				result_center++;
			}

		}
		//2��ڈȍ~����
		int x_add_dim0 = r_dim0 + 1 + dim0Start;// �ǉ�������
		int x_rem_dim0 = -r_dim0 + dim0Start;// �폜������
		int x_dim0 = 1 + dim0Start;// �����Ώے��S��f��
		for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
		{
			const int isNotFirstDim0 = x_dim0 != dim0Start;

			//�ǉ����鎟�̗񂪂��邩
			const int hasAdd_dim0 = x_add_dim0 < size_dim0;
			//�폜����O�̗񂪂��邩
			const int hasRem_dim0 = x_rem_dim0 >= 0;
			//�����ʒu���摜����
			const int isInside_image = x_dim0 >= dim0Start;
			const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
			const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

			//window����f���̍X�V
			if (hasAddOnly_dim0)
				addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
			else if (hasRemOnly_dim0)
				subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�


			//���̗񂪂���Ȃ�X�V
			if (hasAdd_dim0) {
				//�O�̍s������Ȃ�X�V
				if (hasRem_dim1)//�O�̍s���q�X�g�O��������폜
					removePixelFromWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_rem, *G_rem, gSum_win1[x_add_mem]);
				//���̍s������Ȃ�X�V
				if (hasAdd_dim1) //���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_add, *G_add, gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//�폜�񂪂���Ȃ�
			if (hasRem_dim0)
			{
				//���C��window����subwindow�̃q�X�g�O�������폜
				removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//�����l�̌v�Z
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				G_center++;
				result_center++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		G_center += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//�������J��
	delete[] histo_win1;
	delete[] gSum_win1;
	delete[] sumUpToIndex_win1;

	//return result;
}


//FGSumUpToIndex ������(�ǐՖ���)�o�[�W����
template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
void FGMF1::filter2D(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
{
	//check validation
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && ((std::is_same<GTYPE, int>::value && G.channels() == 1) || (std::is_same<GTYPE, cv::Vec3i>::value && G.channels() == 3)));
	assert(I.isContinuous() && G.isContinuous());
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//�����l
	const float half = 0.5f;

	//������
	const int size_dim0 = I.cols;
	const int size_dim1 = I.rows;

	//
	const int r_dim0 = radius;
	const int r_dim1 = radius;

	const int maxPixelNum = (radius * 2 + 1) * (radius * 2 + 1);

	/*
	* �ȉ��͈̔͂́`X�@��X���܂ށiX�ȉ��j
	* �����Ώۉ�f�͈͂� [dim1Start �` dim1End, dim0Start �` dim0End]
	* �E�B���h�E�͈͂�[max(0, dim1Start - r_dim1) �` (std::min)(dim1End, size_dim1) , [max(0, dim0Start) �` (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1�����E�B���h�E��[�A���[
	const int upmostWindowDim1 = std::max(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0�����̃E�B���h�E���[�A�E�[
	const int leftmostWindowDim0 = std::max(0, dim0Start - r_dim0);
	const int rightmostWindowDim0 = std::min(size_dim0 - 1, dim0End + r_dim0);
	//��������
	const int memoryLength = rightmostWindowDim0 - leftmostWindowDim0 + 1;
	//�P�s�ڗp dim1�����E�B���h�E���[
	const int bottomWindowDim1ForFirstLine = std::min(size_dim1 - 1, dim1Start + r_dim1);
	//�P��ڗp dim0�����E�B���h�E�E�[
	const int rightWindowDim0ForFirstLine = std::min(size_dim0 - 1, dim0Start + r_dim0);
	//(1,X)�̎��̃E�B���h�E��
	int pixel_num_dim0_firstLine = rightWindowDim0ForFirstLine - leftmostWindowDim0 + 1;


	//�������m�ہE������

	//�e��p�q�X�g�O�����i�[�ϐ�
	std::list<FG>* histo_win1 = new std::list<FG>[memoryLength];
	//sum(g), sum(g*g)�����v�Z�p�ϐ�
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);

	//Window���q�X�g�O����
	std::list<FG> histo_window;



	///////////////////////////////
	//�Ή������f�ւ̃|�C���^
	//�������̉�f
	GTYPE* G_center = G.ptr<GTYPE>(dim1Start) + dim0Start;
	int* result_center = result.ptr<int>(dim1Start) + dim0Start;
	//�|�C���^�W�����v��
	const int stepForNextRow = size_dim0 - (dim0End - dim0Start + 1);
	const int stepForNextRow2 = stepForNextRow + 1;
	///////////////////////////////


	//1�s�ڂ����ʏ���
	{
		//������
		//Window�֌W
		GSum gSum_window = { 0 };

		//�E�B���h�E�q�X�g�O����������
		histo_window.clear();
		//1�񓖂���̉�f��
		int pixel_sum_dim1 = bottomWindowDim1ForFirstLine - upmostWindowDim1 + 1;
		//�E�B���h�E����f��
		int pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;



		//�������p
		int x_add_mem = 0;
		int x_rem_mem = 0;

		//����ɂP��ڂ����ʏ���
		//(1,1)����
		{
			// leftmostWindowDim0�`rightWindowDim1ForFirstLine �̃q�X�g�O�������\�z
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				//1�s�ڗp�����@�񂷂ׂĒǉ�
				for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx], gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}
			//�����l�̌v�Z
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				G_center++;
				result_center++;
			}
		}
		//(2,1)�`
		{
			int x_add_dim0 = r_dim0 + 1 + dim0Start;// �ǉ�������
			int x_rem_dim0 = -r_dim0 + dim0Start;// �폜������
			int x_dim0 = 1 + dim0Start;// �����Ώے��S��f��

			for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
			{
				//�ǉ����鎟�̗񂪂��邩
				const int hasAdd_dim0 = x_add_dim0 < size_dim0;
				//�폜����O�̗񂪂��邩
				const int hasRem_dim0 = x_rem_dim0 >= 0;
				//�����ʒu���摜����
				const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
				const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

				//window����f���̍X�V
				if (hasAddOnly_dim0)
					addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
				else if (hasRemOnly_dim0)
					subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�

				//���̗񂪂���Ȃ�X�V
				if (hasAdd_dim0) {
					//1�s�ڗp�����@�񂷂ׂĒǉ�
					for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//���̍s���q�X�g�O�����ɒǉ�
						addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0], gSum_win1[x_add_mem]);

					//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
					updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
					x_add_mem++;
				}
				//�폜�񂪂���Ȃ�
				if (hasRem_dim0)
				{
					//���C��window����subwindow�̃q�X�g�O�������폜
					removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//�����l�̌v�Z
				{
					findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
					G_center++;
					result_center++;
				}
			}
			G_center += stepForNextRow;
			result_center += stepForNextRow;
		}
	}
	//������
	int x_add_dim1 = r_dim1 + 1 + dim1Start;//�ǉ�������
	int x_rem_dim1 = -r_dim1 + dim1Start;//�폜�����s
	int x_dim1 = 1 + dim1Start;// �����Ώے��S��f�s
	//
	//��̍폜��f
	int* I_rem = I.ptr<int>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_rem = G.ptr<GTYPE>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	//��̒ǉ���f
	int* I_add = I.ptr<int>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_add = G.ptr<GTYPE>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;

	/*
	I_rem = I.ptr<int>(x_dim1 - r_dim1 - 1)[x_add_dim0]
	I_add = I.ptr<int>(x_dim1 + r_dim1)[x_add_dim0]

	*/

	//2�s�ڈȍ~�����J�n
	for (; x_dim1 <= dim1End; x_dim1++, x_add_dim1++, x_rem_dim1++)
	{
		const int hasAdd_dim1 = x_add_dim1 < size_dim1;
		const int hasRem_dim1 = x_rem_dim1 >= 0;
		const int hasAddOnly_dim1 = !hasRem_dim1 && hasAdd_dim1;
		const int hasRemOnly_dim1 = hasRem_dim1 && !hasAdd_dim1;

		//������
		//Window�֌W
		GSum gSum_window = { 0 };

		//�E�B���h�E�q�X�g�O����������
		//memset(histo_window, 0, sizeof(FG) * Imax);
		histo_window.clear();
		//�E�B���h�E����f��
		int pixel_sum_window = 0;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 0.0f;
		//1�񓖂���̉�f��
		int pixel_sum_dim1;

		if (hasAddOnly_dim1)
			pixel_sum_dim1 = x_dim1 + r_dim1 + 1;
		else if (hasRemOnly_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + r_dim1;
		else
			pixel_sum_dim1 = 2 * r_dim1 + 1;



		//�������p
		int x_add_mem = 0;
		int x_rem_mem = 0;


		//1��ڏ���
		{
			//�E�B���h�E����f��
			pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
			//�E�B���h�E����f���̋t��
			pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;

			// leftmostWindowDim0�`rightWindowDim1ForFirstLine �̃q�X�g�O�������\�z
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				if (hasRem_dim1)//�O�̍s���q�X�g�O��������폜
					removePixelFromWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx], gSum_win1[x_add_mem]);
				if (hasAdd_dim1)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx], gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//�����l�̌v�Z
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				G_center++;
				result_center++;
			}

		}
		//2��ڈȍ~����
		int x_add_dim0 = r_dim0 + 1 + dim0Start;// �ǉ�������
		int x_rem_dim0 = -r_dim0 + dim0Start;// �폜������
		int x_dim0 = 1 + dim0Start;// �����Ώے��S��f��
		for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
		{
			const int isNotFirstDim0 = x_dim0 != dim0Start;

			//�ǉ����鎟�̗񂪂��邩
			const int hasAdd_dim0 = x_add_dim0 < size_dim0;
			//�폜����O�̗񂪂��邩
			const int hasRem_dim0 = x_rem_dim0 >= 0;
			//�����ʒu���摜����
			const int isInside_image = x_dim0 >= dim0Start;
			const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
			const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

			//window����f���̍X�V
			if (hasAddOnly_dim0)
				addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
			else if (hasRemOnly_dim0)
				subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�


			//���̗񂪂���Ȃ�X�V
			if (hasAdd_dim0) {
				//�O�̍s������Ȃ�X�V
				if (hasRem_dim1)//�O�̍s���q�X�g�O��������폜
					removePixelFromWindow_gSum(histo_win1[x_add_mem], *I_rem, *G_rem, gSum_win1[x_add_mem]);
				//���̍s������Ȃ�X�V
				if (hasAdd_dim1) //���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], *I_add, *G_add, gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//�폜�񂪂���Ȃ�
			if (hasRem_dim0)
			{
				//���C��window����subwindow�̃q�X�g�O�������폜
				removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//�����l�̌v�Z
			{
				findMedian(gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				G_center++;
				result_center++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		G_center += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//�������J��
	delete[] histo_win1;
	delete[] gSum_win1;
}




//FGSumUpToIndex ������(�ǐՖ���)�o�[�W����  cxdx��ۑ����Ȃ�����s�i���̓J���[�摜�p�j
template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
void FGMF1::filter2D_saveCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
{
	//check validation
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && ((std::is_same<GTYPE, int>::value && G.channels() == 1) || (std::is_same<GTYPE, cv::Vec3i>::value && G.channels() == 3)));
	assert(I.isContinuous() && G.isContinuous());
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//�����l
	const float half = 0.5f;

	//������
	const int size_dim0 = I.cols;
	const int size_dim1 = I.rows;

	//
	const int r_dim0 = radius;
	const int r_dim1 = radius;

	const int maxPixelNum = (radius * 2 + 1) * (radius * 2 + 1);

	/*
	* �ȉ��͈̔͂́`X�@��X���܂ށiX�ȉ��j
	* �����Ώۉ�f�͈͂� [dim1Start �` dim1End, dim0Start �` dim0End]
	* �E�B���h�E�͈͂�[max(0, dim1Start - r_dim1) �` (std::min)(dim1End, size_dim1) , [max(0, dim0Start) �` (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1�����E�B���h�E��[�A���[
	const int upmostWindowDim1 = std::max(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0�����̃E�B���h�E���[�A�E�[
	const int leftmostWindowDim0 = std::max(0, dim0Start - r_dim0);
	const int rightmostWindowDim0 = std::min(size_dim0 - 1, dim0End + r_dim0);
	//��������
	const int memoryLength = rightmostWindowDim0 - leftmostWindowDim0 + 1;
	//�P�s�ڗp dim1�����E�B���h�E���[
	const int bottomWindowDim1ForFirstLine = std::min(size_dim1 - 1, dim1Start + r_dim1);
	//�P��ڗp dim0�����E�B���h�E�E�[
	const int rightWindowDim0ForFirstLine = std::min(size_dim0 - 1, dim0Start + r_dim0);
	//(1,X)�̎��̃E�B���h�E��
	int pixel_num_dim0_firstLine = rightWindowDim0ForFirstLine - leftmostWindowDim0 + 1;


	//�������m�ہE������

	//�e��p�q�X�g�O�����i�[�ϐ�
	std::list<FG>* histo_win1 = new std::list<FG>[memoryLength];
	//sum(g), sum(g*g)�����v�Z�p�ϐ�
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);

	//Window���q�X�g�O����
	std::list<FG> histo_window;



	///////////////////////////////
	//�Ή������f�ւ̃|�C���^
	//�������̉�f
	GTYPE* G_center = G.ptr<GTYPE>(dim1Start) + dim0Start;
	int* result_center = result.ptr<int>(dim1Start) + dim0Start;
	CTYPE* cx = _cx.ptr<CTYPE>(dim1Start) + dim0Start;
	float* dx = _dx.ptr<float>(dim1Start) + dim0Start;
	//�|�C���^�W�����v��
	const int stepForNextRow = size_dim0 - (dim0End - dim0Start + 1);
	const int stepForNextRow2 = stepForNextRow + 1;
	///////////////////////////////


	//1�s�ڂ����ʏ���
	{
		//������
		//Window�֌W
		GSum gSum_window = { 0 };

		//�E�B���h�E�q�X�g�O����������
		histo_window.clear();
		//1�񓖂���̉�f��
		int pixel_sum_dim1 = bottomWindowDim1ForFirstLine - upmostWindowDim1 + 1;
		//�E�B���h�E����f��
		int pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;



		//�������p
		int x_add_mem = 0;
		int x_rem_mem = 0;

		//����ɂP��ڂ����ʏ���
		//(1,1)����
		{
			// leftmostWindowDim0�`rightWindowDim1ForFirstLine �̃q�X�g�O�������\�z
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				//1�s�ڗp�����@�񂷂ׂĒǉ�
				for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx], gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}
			//�����l�̌v�Z
			{
				findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}
		}
		//(2,1)�`
		{
			int x_add_dim0 = r_dim0 + 1 + dim0Start;// �ǉ�������
			int x_rem_dim0 = -r_dim0 + dim0Start;// �폜������
			int x_dim0 = 1 + dim0Start;// �����Ώے��S��f��

			for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
			{
				//�ǉ����鎟�̗񂪂��邩
				const int hasAdd_dim0 = x_add_dim0 < size_dim0;
				//�폜����O�̗񂪂��邩
				const int hasRem_dim0 = x_rem_dim0 >= 0;
				//�����ʒu���摜����
				const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
				const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

				//window����f���̍X�V
				if (hasAddOnly_dim0)
					addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
				else if (hasRemOnly_dim0)
					subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�

				//���̗񂪂���Ȃ�X�V
				if (hasAdd_dim0) {
					//1�s�ڗp�����@�񂷂ׂĒǉ�
					for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//���̍s���q�X�g�O�����ɒǉ�
						addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0], gSum_win1[x_add_mem]);

					//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
					updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
					x_add_mem++;
				}
				//�폜�񂪂���Ȃ�
				if (hasRem_dim0)
				{
					//���C��window����subwindow�̃q�X�g�O�������폜
					removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//�����l�̌v�Z
				{
					findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
					cx++;
					dx++;
					G_center++;
					result_center++;
				}
			}
			cx += stepForNextRow;
			dx += stepForNextRow;
			G_center += stepForNextRow;
			result_center += stepForNextRow;
		}
	}
	//������
	int x_add_dim1 = r_dim1 + 1 + dim1Start;//�ǉ�������
	int x_rem_dim1 = -r_dim1 + dim1Start;//�폜�����s
	int x_dim1 = 1 + dim1Start;// �����Ώے��S��f�s
	//
	//��̍폜��f
	int* I_rem = I.ptr<int>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_rem = G.ptr<GTYPE>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	//��̒ǉ���f
	int* I_add = I.ptr<int>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_add = G.ptr<GTYPE>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;

	/*
	I_rem = I.ptr<int>(x_dim1 - r_dim1 - 1)[x_add_dim0]
	I_add = I.ptr<int>(x_dim1 + r_dim1)[x_add_dim0]

	*/

	//2�s�ڈȍ~�����J�n
	for (; x_dim1 <= dim1End; x_dim1++, x_add_dim1++, x_rem_dim1++)
	{
		const int hasAdd_dim1 = x_add_dim1 < size_dim1;
		const int hasRem_dim1 = x_rem_dim1 >= 0;
		const int hasAddOnly_dim1 = !hasRem_dim1 && hasAdd_dim1;
		const int hasRemOnly_dim1 = hasRem_dim1 && !hasAdd_dim1;

		//������
		//Window�֌W
		GSum gSum_window = { 0 };

		//�E�B���h�E�q�X�g�O����������
		//memset(histo_window, 0, sizeof(FG) * Imax);
		histo_window.clear();
		//�E�B���h�E����f��
		int pixel_sum_window = 0;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 0.0f;
		//1�񓖂���̉�f��
		int pixel_sum_dim1;

		if (hasAddOnly_dim1)
			pixel_sum_dim1 = x_dim1 + r_dim1 + 1;
		else if (hasRemOnly_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + r_dim1;
		else
			pixel_sum_dim1 = 2 * r_dim1 + 1;



		//�������p
		int x_add_mem = 0;
		int x_rem_mem = 0;


		//1��ڏ���
		{
			//�E�B���h�E����f��
			pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
			//�E�B���h�E����f���̋t��
			pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;

			// leftmostWindowDim0�`rightWindowDim1ForFirstLine �̃q�X�g�O�������\�z
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				if (hasRem_dim1)//�O�̍s���q�X�g�O��������폜
					removePixelFromWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx], gSum_win1[x_add_mem]);
				if (hasAdd_dim1)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx], gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//�����l�̌v�Z
			{
				findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}

		}
		//2��ڈȍ~����
		int x_add_dim0 = r_dim0 + 1 + dim0Start;// �ǉ�������
		int x_rem_dim0 = -r_dim0 + dim0Start;// �폜������
		int x_dim0 = 1 + dim0Start;// �����Ώے��S��f��
		for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
		{
			const int isNotFirstDim0 = x_dim0 != dim0Start;

			//�ǉ����鎟�̗񂪂��邩
			const int hasAdd_dim0 = x_add_dim0 < size_dim0;
			//�폜����O�̗񂪂��邩
			const int hasRem_dim0 = x_rem_dim0 >= 0;
			//�����ʒu���摜����
			const int isInside_image = x_dim0 >= dim0Start;
			const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
			const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

			//window����f���̍X�V
			if (hasAddOnly_dim0)
				addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
			else if (hasRemOnly_dim0)
				subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�


			//���̗񂪂���Ȃ�X�V
			if (hasAdd_dim0) {
				//�O�̍s������Ȃ�X�V
				if (hasRem_dim1)//�O�̍s���q�X�g�O��������폜
					removePixelFromWindow_gSum(histo_win1[x_add_mem], *I_rem, *G_rem, gSum_win1[x_add_mem]);
				//���̍s������Ȃ�X�V
				if (hasAdd_dim1) //���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], *I_add, *G_add, gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//�폜�񂪂���Ȃ�
			if (hasRem_dim0)
			{
				//���C��window����subwindow�̃q�X�g�O�������폜
				removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//�����l�̌v�Z
			{
				findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		cx += stepForNextRow;
		dx += stepForNextRow;
		G_center += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//�������J��
	delete[] histo_win1;
	delete[] gSum_win1;
}




//FGSumUpToIndex ������(�ǐՖ���)�o�[�W����  cxdx��ۑ����Ȃ�����s�i���̓J���[�摜�p�j
template<typename GSum, typename FG, typename GTYPE, typename CTYPE>
void FGMF1::filter2D_useCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
{
	//check validation
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && ((std::is_same<GTYPE, int>::value && G.channels() == 1) || (std::is_same<GTYPE, cv::Vec3i>::value && G.channels() == 3)));
	assert(I.isContinuous() && G.isContinuous());
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//�����l
	const float half = 0.5f;

	//������
	const int size_dim0 = I.cols;
	const int size_dim1 = I.rows;

	//
	const int r_dim0 = radius;
	const int r_dim1 = radius;

	const int maxPixelNum = (radius * 2 + 1) * (radius * 2 + 1);

	/*
	* �ȉ��͈̔͂́`X�@��X���܂ށiX�ȉ��j
	* �����Ώۉ�f�͈͂� [dim1Start �` dim1End, dim0Start �` dim0End]
	* �E�B���h�E�͈͂�[max(0, dim1Start - r_dim1) �` (std::min)(dim1End, size_dim1) , [max(0, dim0Start) �` (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1�����E�B���h�E��[�A���[
	const int upmostWindowDim1 = std::max(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0�����̃E�B���h�E���[�A�E�[
	const int leftmostWindowDim0 = std::max(0, dim0Start - r_dim0);
	const int rightmostWindowDim0 = std::min(size_dim0 - 1, dim0End + r_dim0);
	//��������
	const int memoryLength = rightmostWindowDim0 - leftmostWindowDim0 + 1;
	//�P�s�ڗp dim1�����E�B���h�E���[
	const int bottomWindowDim1ForFirstLine = std::min(size_dim1 - 1, dim1Start + r_dim1);
	//�P��ڗp dim0�����E�B���h�E�E�[
	const int rightWindowDim0ForFirstLine = std::min(size_dim0 - 1, dim0Start + r_dim0);
	//(1,X)�̎��̃E�B���h�E��
	int pixel_num_dim0_firstLine = rightWindowDim0ForFirstLine - leftmostWindowDim0 + 1;


	//�������m�ہE������

	//�e��p�q�X�g�O�����i�[�ϐ�
	std::list<FG>* histo_win1 = new std::list<FG>[memoryLength];
	//sum(g), sum(g*g)�����v�Z�p�ϐ�
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);

	//Window���q�X�g�O����
	std::list<FG> histo_window;



	///////////////////////////////
	//�Ή������f�ւ̃|�C���^
	//�������̉�f
	GTYPE* G_center = G.ptr<GTYPE>(dim1Start) + dim0Start;
	int* result_center = result.ptr<int>(dim1Start) + dim0Start;
	CTYPE* cx = _cx.ptr<CTYPE>(dim1Start) + dim0Start;
	float* dx = _dx.ptr<float>(dim1Start) + dim0Start;
	//�|�C���^�W�����v��
	const int stepForNextRow = size_dim0 - (dim0End - dim0Start + 1);
	const int stepForNextRow2 = stepForNextRow + 1;
	///////////////////////////////


	//1�s�ڂ����ʏ���
	{
		//������
		//Window�֌W
		GSum gSum_window = { 0 };

		//�E�B���h�E�q�X�g�O����������
		histo_window.clear();
		//1�񓖂���̉�f��
		int pixel_sum_dim1 = bottomWindowDim1ForFirstLine - upmostWindowDim1 + 1;
		//�E�B���h�E����f��
		int pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;



		//�������p
		int x_add_mem = 0;
		int x_rem_mem = 0;

		//����ɂP��ڂ����ʏ���
		//(1,1)����
		{
			// leftmostWindowDim0�`rightWindowDim1ForFirstLine �̃q�X�g�O�������\�z
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				//1�s�ڗp�����@�񂷂ׂĒǉ�
				for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx], gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}
			//�����l�̌v�Z
			{
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				findMedian(*cx, *dx, half, histo_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}
		}
		//(2,1)�`
		{
			int x_add_dim0 = r_dim0 + 1 + dim0Start;// �ǉ�������
			int x_rem_dim0 = -r_dim0 + dim0Start;// �폜������
			int x_dim0 = 1 + dim0Start;// �����Ώے��S��f��

			for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
			{
				//�ǉ����鎟�̗񂪂��邩
				const int hasAdd_dim0 = x_add_dim0 < size_dim0;
				//�폜����O�̗񂪂��邩
				const int hasRem_dim0 = x_rem_dim0 >= 0;
				//�����ʒu���摜����
				const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
				const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

				//window����f���̍X�V
				if (hasAddOnly_dim0)
					addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
				else if (hasRemOnly_dim0)
					subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�

				//���̗񂪂���Ȃ�X�V
				if (hasAdd_dim0) {
					//1�s�ڗp�����@�񂷂ׂĒǉ�
					for (int yy = upmostWindowDim1; yy <= bottomWindowDim1ForFirstLine; yy++)//���̍s���q�X�g�O�����ɒǉ�
						addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0], gSum_win1[x_add_mem]);

					//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
					updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
					x_add_mem++;
				}
				//�폜�񂪂���Ȃ�
				if (hasRem_dim0)
				{
					//���C��window����subwindow�̃q�X�g�O�������폜
					removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//�����l�̌v�Z
				{
					//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
					findMedian(*cx, *dx, half, histo_window, *result_center);
					cx++;
					dx++;
					G_center++;
					result_center++;
				}
			}
			cx += stepForNextRow;
			dx += stepForNextRow;
			G_center += stepForNextRow;
			result_center += stepForNextRow;
		}
	}
	//������
	int x_add_dim1 = r_dim1 + 1 + dim1Start;//�ǉ�������
	int x_rem_dim1 = -r_dim1 + dim1Start;//�폜�����s
	int x_dim1 = 1 + dim1Start;// �����Ώے��S��f�s
	//
	//��̍폜��f
	int* I_rem = I.ptr<int>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_rem = G.ptr<GTYPE>(dim1Start) - r_dim1 * size_dim0 + dim0Start + r_dim0 + 1;
	//��̒ǉ���f
	int* I_add = I.ptr<int>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;
	GTYPE* G_add = G.ptr<GTYPE>(dim1Start) + (r_dim1 + 1) * size_dim0 + dim0Start + r_dim0 + 1;

	/*
	I_rem = I.ptr<int>(x_dim1 - r_dim1 - 1)[x_add_dim0]
	I_add = I.ptr<int>(x_dim1 + r_dim1)[x_add_dim0]

	*/

	//2�s�ڈȍ~�����J�n
	for (; x_dim1 <= dim1End; x_dim1++, x_add_dim1++, x_rem_dim1++)
	{
		const int hasAdd_dim1 = x_add_dim1 < size_dim1;
		const int hasRem_dim1 = x_rem_dim1 >= 0;
		const int hasAddOnly_dim1 = !hasRem_dim1 && hasAdd_dim1;
		const int hasRemOnly_dim1 = hasRem_dim1 && !hasAdd_dim1;

		//������
		//Window�֌W
		GSum gSum_window = { 0 };

		//�E�B���h�E�q�X�g�O����������
		//memset(histo_window, 0, sizeof(FG) * Imax);
		histo_window.clear();
		//�E�B���h�E����f��
		int pixel_sum_window = 0;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 0.0f;
		//1�񓖂���̉�f��
		int pixel_sum_dim1;

		if (hasAddOnly_dim1)
			pixel_sum_dim1 = x_dim1 + r_dim1 + 1;
		else if (hasRemOnly_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + r_dim1;
		else
			pixel_sum_dim1 = 2 * r_dim1 + 1;



		//�������p
		int x_add_mem = 0;
		int x_rem_mem = 0;


		//1��ڏ���
		{
			//�E�B���h�E����f��
			pixel_sum_window = pixel_sum_dim1 * pixel_num_dim0_firstLine;
			//�E�B���h�E����f���̋t��
			pixel_sum_window_inv = 1.0f / (float)pixel_sum_window;

			// leftmostWindowDim0�`rightWindowDim1ForFirstLine �̃q�X�g�O�������\�z
			for (int xx = leftmostWindowDim0; xx <= rightWindowDim0ForFirstLine; xx++)
			{
				if (hasRem_dim1)//�O�̍s���q�X�g�O��������폜
					removePixelFromWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx], gSum_win1[x_add_mem]);
				if (hasAdd_dim1)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx], gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//�����l�̌v�Z
			{
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				findMedian(*cx, *dx, half, histo_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}

		}
		//2��ڈȍ~����
		int x_add_dim0 = r_dim0 + 1 + dim0Start;// �ǉ�������
		int x_rem_dim0 = -r_dim0 + dim0Start;// �폜������
		int x_dim0 = 1 + dim0Start;// �����Ώے��S��f��
		for (; x_dim0 <= dim0End; x_dim0++, x_add_dim0++, x_rem_dim0++)
		{
			const int isNotFirstDim0 = x_dim0 != dim0Start;

			//�ǉ����鎟�̗񂪂��邩
			const int hasAdd_dim0 = x_add_dim0 < size_dim0;
			//�폜����O�̗񂪂��邩
			const int hasRem_dim0 = x_rem_dim0 >= 0;
			//�����ʒu���摜����
			const int isInside_image = x_dim0 >= dim0Start;
			const int hasAddOnly_dim0 = !hasRem_dim0 && hasAdd_dim0;
			const int hasRemOnly_dim0 = hasRem_dim0 && !hasAdd_dim0;

			//window����f���̍X�V
			if (hasAddOnly_dim0)
				addPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�ǉ��̂�
			else if (hasRemOnly_dim0)
				subtractPixelSum(pixel_sum_dim1, pixel_sum_window, pixel_sum_window_inv);//�폜�̂�


			//���̗񂪂���Ȃ�X�V
			if (hasAdd_dim0) {
				//�O�̍s������Ȃ�X�V
				if (hasRem_dim1)//�O�̍s���q�X�g�O��������폜
					removePixelFromWindow_gSum(histo_win1[x_add_mem], *I_rem, *G_rem, gSum_win1[x_add_mem]);
				//���̍s������Ȃ�X�V
				if (hasAdd_dim1) //���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow_gSum(histo_win1[x_add_mem], *I_add, *G_add, gSum_win1[x_add_mem]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow_gSum(Imax, histo_window, histo_win1[x_add_mem], gSum_window, gSum_win1[x_add_mem]);
				x_add_mem++;
			}

			//�폜�񂪂���Ȃ�
			if (hasRem_dim0)
			{
				//���C��window����subwindow�̃q�X�g�O�������폜
				removeSubWindowFromWindow_gSum(Imax, histo_window, histo_win1[x_rem_mem], gSum_window, gSum_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//�����l�̌v�Z
			{
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, *result_center);
				findMedian(*cx, *dx, half, histo_window, *result_center);
				cx++;
				dx++;
				G_center++;
				result_center++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		cx += stepForNextRow;
		dx += stepForNextRow;
		G_center += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//�������J��
	delete[] histo_win1;
	delete[] gSum_win1;
}

