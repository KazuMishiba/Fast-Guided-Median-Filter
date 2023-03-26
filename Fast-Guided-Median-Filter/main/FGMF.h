#pragma once
#include "FGMF_base.h"

#include "CxDxPrecalculation.cuh"

//#include "tbb/parallel_for.h"
//#include "tbb/task_scheduler_init.h"


//#define NEARLY_ZERO 0.000000000001f

//using namespace std;
//using namespace cv;

/*
���̌�̗\��
�E�]�v�Ȋ֐��̍폜�A�e���v���[�g�ɂ�铝��
�E�}���`�`�����l����
�E�RD��
�E�SD��
�E�������\�z
�E�ic,d ��GPU�v�Z�j

2Dcolor�����̑��x�ȉ��̏�
�i�����j
�E3�`�����l���ʂ�cx,dx�v�Z
�E1�`�����l���ڂ�cx,dx�v�Z���A2,3�`�����l���ڂŗ��p����
�E3�`�����l�������v�Z�v�Z�ŁAcx,dx1�`�����l���ڂŌv�Z�����̂𑱂��Ďg��
�i�x���j
�v�Z�ʂ����Ō���΁A��ԉ�����ԏ��Ȃ����A��Ԓx���B
����͂����炭�������̓ǂݍ��݂Ƃ��L���b�V���̖��B



sumuptoindex�g��Ȃ��e�X�g�������Ƃ���A10�{���炢�x�������B���Ȃ�����Ă���悤�B


*/


class FGMF
{
public:
	//2D�ėp
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D_useCD(cv::Mat& I, cv::Mat& G, cv::Mat& cx, cv::Mat& dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2D_saveCD(cv::Mat& I, cv::Mat& G, cv::Mat& cx, cv::Mat& dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);

	/*
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2DColor(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);

	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2DColor(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim0Start, int dim0End, int radius, float eps2, int Imax);
	*/
	//sumuptoindex�g��Ȃ��e�X�g
	static cv::Mat filter2DInterfaceTest(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2 = 25.5f * 25.5f, int Imax = 256);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static void filter2Dtest(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax);

	//Thread�����w��
	static cv::Mat filter2DInterface(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2 = 25.5f * 25.5f, int Imax = 256);




	//Single Thread
	//I1
	static cv::Mat filter2DI1G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	static cv::Mat filter2DI1G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	//I3
	static cv::Mat filter2DI3G1_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	static cv::Mat filter2DI3G3_SingleThread(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_SingleThread(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	//Multi Thread
	//cols (������)
	//I1
	static cv::Mat filter2DI1G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	static cv::Mat filter2DI1G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	//I3
	static cv::Mat filter2DI3G1_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	static cv::Mat filter2DI3G3_MultiThread(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_MultiThread(int cx_cv32, cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax);





	static cv::Mat filter2DI3G3_MultiThread2(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_MultiThread2(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	static cv::Mat filter2DI3G3_MultiThread3(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	static cv::Mat filter2DI3_MultiThread3(int cx_cv32, cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax);


	//cx,dx GPU�v�Z
	static void calculateCxDxOnGPU(cv::Mat& G, int radius, float eps2, SizeInfo& sizeInfo, cv::Mat& cx, cv::Mat& dx);

	static cv::Mat gpuTestCxDx(cv::Mat& I, cv::Mat& G, int radius, float eps2, int Imax, SizeInfo& sizeInfo);



};











////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////

template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF::filter2D(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
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
	FG** histo_win1 = new FG *[memoryLength];
	for (int i = 0; i < memoryLength; i++)
	{
		histo_win1[i] = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(FG) * Imax);
	}
	//sum(g), sum(g*g)�����v�Z�p�ϐ�
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);
	//index_win1�ȉ��̗�q�X�g�O�����a
	FGSumUpToIndex* sumUpToIndex_win1 = new FGSumUpToIndex[memoryLength];
	memset(sumUpToIndex_win1, 0, sizeof(FGSumUpToIndex) * memoryLength);

	//Window���q�X�g�O����
	FG* histo_window = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);



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
		memset(histo_window, 0, sizeof(FG) * Imax);
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
		memset(histo_window, 0, sizeof(FG) * Imax);
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
	for (int i = 0; i < memoryLength; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] gSum_win1;
	delete[] sumUpToIndex_win1;

	//return result;
}


template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF::filter2D_useCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
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

	/*
	* �ȉ��͈̔͂́`X�@��X���܂ށiX�ȉ��j
	* �����Ώۉ�f�͈͂� [dim1Start �` dim1End, dim0Start �` dim0End]
	* �E�B���h�E�͈͂�[max(0, dim1Start - r_dim1) �` (std::min)(dim1End, size_dim1) , [max(0, dim0Start) �` (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1�����E�B���h�E��[�A���[
	const int upmostWindowDim1 = (std::max)(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0�����̃E�B���h�E���[�A�E�[
	const int leftmostWindowDim0 = (std::max)(0, dim0Start - r_dim0);
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
	FG** histo_win1 = new FG *[memoryLength];
	for (int i = 0; i < memoryLength; i++)
	{
		histo_win1[i] = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(FG) * Imax);
	}
	//index_win1�ȉ��̗�q�X�g�O�����a
	FGSumUpToIndex* sumUpToIndex_win1 = new FGSumUpToIndex[memoryLength];
	memset(sumUpToIndex_win1, 0, sizeof(FGSumUpToIndex) * memoryLength);

	//Window���q�X�g�O����
	FG* histo_window = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);



	///////////////////////////////
	//�Ή������f�ւ̃|�C���^
	//�������̉�f
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
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(dim1Start);

		//�E�B���h�E�q�X�g�O����������
		memset(histo_window, 0, sizeof(FG) * Imax);
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
					addPixelToWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[xx], G.ptr<GTYPE>(yy)[xx]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem]);
				x_add_mem++;
			}
			//�����l�̌v�Z
			{
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				result_center++;
				cx++;
				dx++;
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
						addPixelToWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(yy)[x_add_dim0], G.ptr<GTYPE>(yy)[x_add_dim0]);

					//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
					updateSubWindowAndAddToWindow(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem]);
					x_add_mem++;
				}
				//�폜�񂪂���Ȃ�
				if (hasRem_dim0)
				{
					//���C��window����subwindow�̃q�X�g�O�������폜
					removeSubWindowFromWindow(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem]);
					x_rem_mem++;
				}
				//�����l�̌v�Z
				{
					findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
					result_center++;
					cx++;
					dx++;
				}
			}
			cx += stepForNextRow;
			dx += stepForNextRow;
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
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(x_dim1);

		//�E�B���h�E�q�X�g�O����������
		memset(histo_window, 0, sizeof(FG) * Imax);
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
					removePixelFromWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_rem_dim1)[xx], G.ptr<GTYPE>(x_rem_dim1)[xx]);
				if (hasAdd_dim1)//���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], I.ptr<int>(x_add_dim1)[xx], G.ptr<GTYPE>(x_add_dim1)[xx]);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem]);
				x_add_mem++;
			}

			//�����l�̌v�Z
			{
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				result_center++;
				cx++;
				dx++;
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
					removePixelFromWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_rem, *G_rem);
				//���̍s������Ȃ�X�V
				if (hasAdd_dim1) //���̍s���q�X�g�O�����ɒǉ�
					addPixelToWindow(histo_win1[x_add_mem], sumUpToIndex_win1[x_add_mem], *I_add, *G_add);

				//subwindow���X�V���A��������C��window�̃q�X�g�O�����ɒǉ�
				updateSubWindowAndAddToWindow(Imax, histo_window, histo_win1[x_add_mem], sumUpToIndex_window, sumUpToIndex_win1[x_add_mem]);
				x_add_mem++;
			}

			//�폜�񂪂���Ȃ�
			if (hasRem_dim0)
			{
				//���C��window����subwindow�̃q�X�g�O�������폜
				removeSubWindowFromWindow(Imax, histo_window, histo_win1[x_rem_mem], sumUpToIndex_window, sumUpToIndex_win1[x_rem_mem]);
				x_rem_mem++;
			}

			//�����l�̌v�Z
			{
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
				result_center++;
				cx++;
				dx++;
			}
			I_rem++;
			G_rem++;
			I_add++;
			G_add++;
		}
		cx += stepForNextRow;
		dx += stepForNextRow;
		result_center += stepForNextRow;
		I_add += stepForNextRow2;
		G_add += stepForNextRow2;
		I_rem += stepForNextRow2;
		G_rem += stepForNextRow2;

	}
	//�������J��
	for (int i = 0; i < memoryLength; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] sumUpToIndex_win1;

	//return result;
}


template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF::filter2D_saveCD(cv::Mat& I, cv::Mat& G, cv::Mat& _cx, cv::Mat& _dx, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
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

	/*
	* �ȉ��͈̔͂́`X�@��X���܂ށiX�ȉ��j
	* �����Ώۉ�f�͈͂� [dim1Start �` dim1End, dim0Start �` dim0End]
	* �E�B���h�E�͈͂�[max(0, dim1Start - r_dim1) �` (std::min)(dim1End, size_dim1) , [max(0, dim0Start) �` (std::min)(dim0End, size_dim0)]
	*/

	///////////////////////////////
	//dim1�����E�B���h�E��[�A���[
	const int upmostWindowDim1 = (std::max)(0, dim1Start - r_dim1);
	const int bottommostWindowDim1 = std::min(size_dim1 - 1, dim1End + r_dim1);
	//dim0�����̃E�B���h�E���[�A�E�[
	const int leftmostWindowDim0 = (std::max)(0, dim0Start - r_dim0);
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
	FG** histo_win1 = new FG *[memoryLength];
	for (int i = 0; i < memoryLength; i++)
	{
		histo_win1[i] = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(FG) * Imax);
	}
	//sum(g), sum(g*g)�����v�Z�p�ϐ�
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);
	//index_win1�ȉ��̗�q�X�g�O�����a
	FGSumUpToIndex* sumUpToIndex_win1 = new FGSumUpToIndex[memoryLength];
	memset(sumUpToIndex_win1, 0, sizeof(FGSumUpToIndex) * memoryLength);

	//Window���q�X�g�O����
	FG* histo_window = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);



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
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(dim1Start);

		//�E�B���h�E�q�X�g�O����������
		memset(histo_window, 0, sizeof(FG) * Imax);
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
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
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
					//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
					calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
					findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
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
		FGSumUpToIndex sumUpToIndex_window = { 0 };
		sumUpToIndex_window.index = *I.ptr<int>(x_dim1);

		//�E�B���h�E�q�X�g�O����������
		memset(histo_window, 0, sizeof(FG) * Imax);
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
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
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
				//findMedian(*cx, *dx, gSum_window, pixel_sum_window_inv, eps2, *G_center, half, histo_window, sumUpToIndex_window, *result_center);
				calculateCxDx(gSum_window, pixel_sum_window_inv, eps2, *G_center, *cx, *dx);
				findMedian(*cx, *dx, half, histo_window, sumUpToIndex_window, *result_center);
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
	for (int i = 0; i < memoryLength; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] gSum_win1;
	delete[] sumUpToIndex_win1;

	//return result;
}







//sumuptoindex�����e�X�g
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
void FGMF::filter2Dtest(cv::Mat& I, cv::Mat& G, cv::Mat& result, int dim1Start, int dim1End, int dim0Start, int dim0End, int radius, float eps2, int Imax)
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
	FG** histo_win1 = new FG *[memoryLength];
	for (int i = 0; i < memoryLength; i++)
	{
		histo_win1[i] = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(FG) * Imax);
	}
	//sum(g), sum(g*g)�����v�Z�p�ϐ�
	GSum* gSum_win1 = new GSum[memoryLength];
	memset(gSum_win1, 0, sizeof(GSum) * memoryLength);

	//Window���q�X�g�O����
	FG* histo_window = (FG*)_aligned_malloc(sizeof(FG) * Imax, MEMORY_ALIGNMENT);



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
		memset(histo_window, 0, sizeof(FG) * Imax);
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
		memset(histo_window, 0, sizeof(FG) * Imax);
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
	for (int i = 0; i < memoryLength; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] gSum_win1;

	//return result;
}