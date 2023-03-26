#pragma once

//#include "FGMF_base.h"
#include "CalculateC_D_ver0.cuh"

#include "FGMF_GPU_Or_ver0.cuh"


namespace FGMF_GPU_Or_ver0
{
	cv::Mat filter_2d(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange);
	cv::Mat filter_2d_multichannel(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange);


	class WMF
	{
	public:
		WMF(cv::Mat& f_img, cv::Mat& g_img, int radius, float eps2, int fRange)
			: f_img_(f_img), g_img_(g_img), radius_(radius), eps2_(eps2), fRange_(fRange), M_(f_img.rows), N_(f_img.cols), channelNum_f_(f_img_.channels()), channelNum_g_(g_img_.channels()), sizeInfo_(f_img.cols, f_img.rows, fRange)
		{
			//check validation
			assert(f_img_.depth() == CV_32S || f_img_.depth() == CV_8U);
			assert(g_img_.depth() == CV_32S || g_img_.depth() == CV_8U);
			if (f_img_.depth() != CV_32S)
				f_img_.convertTo(f_img_, CV_32S);
			if (g_img_.depth() != CV_32S)
				g_img_.convertTo(g_img_, CV_32S);

			
			f_device_ = new Helper::DeviceArray<int>(f_img_, sizeInfo_);
			g_device_ = new Helper::DeviceArray<int>(g_img_, sizeInfo_);
			result_device_ = new Helper::DeviceArray<int>(channelNum_f_, sizeInfo_);
		}

		void apply_2d_filter();
		void apply_2d_filter_multichannel();
		cv::Mat downloadResultAsMat();

	private:
		cv::Mat f_img_;
		cv::Mat g_img_;
		int radius_;
		float eps2_;
		int fRange_;

		int M_; // Image height
		int N_; // Image width
		int channelNum_f_; // Input image channel num
		int channelNum_g_; // Guide image channel num
//		int p_; // x^+ = x + r = x + p_
//		int m_; // x^- = x - r - 1 = x + m_

		Helper::SizeInfo sizeInfo_;
		Helper::DeviceArray<int>* f_device_;
		Helper::DeviceArray<int>* g_device_;
		Helper::DeviceArray<int>* result_device_;


		//////////////////////////////////
		//     GPU
		//////////////////////////////////
		//2D
		template<typename DC_TYPE>
		static void filter2DGPU(Helper::DeviceArray<int>*& f, Helper::DeviceArray<int>*& g, Helper::DeviceArray<int>*& result, int radius, float eps2, int fRange, Helper::SizeInfo& sizeInfo);


		//Multichannel
		static void filter2DGPU_MultiChannel(Helper::DeviceArray<int>*& f, Helper::DeviceArray<int>*& g, Helper::DeviceArray<int>*& result, int radius, float eps2, int fRange, Helper::SizeInfo& sizeInfo);

#if 0

		//�e�X�g�p multi channel�f�o�b�O�p
		//static void filter2DGPU_Test(Helper::DeviceArray<int>*& I, Helper::DeviceArray<int>*& G, Helper::DeviceArray<int>*& result, int radius, float eps2, int fRange, Helper::SizeInfo& sizeInfo);


		//3D
		//I1G1, I3G1 template ITYPE = int or Helper::DeviceArray<int>
		template<typename ITYPE>
		static void filter3DGPU(std::vector<ITYPE*>& I, std::vector<int*>& G, std::vector<ITYPE*>& result, int radius_spacial, int radius_temporal, float eps2, int fRange, Helper::SizeInfo& sizeInfo);
		//I1G3, I3G3 template GTYPE = int or Helper::DeviceArray<int>
		template<typename ITYPE>
		static void filter3DGPU(std::vector<ITYPE*>& I, std::vector<Helper::DeviceArray<int>*>& G, std::vector<ITYPE*>& result, int radius_spacial, int radius_temporal, float eps2, int fRange, Helper::SizeInfo& sizeInfo);

		//4D
		//I1G1, I3G1 template ITYPE = int or Helper::DeviceArray<int>
		template<typename ITYPE>
		static void filter4DGPU(std::vector<ITYPE*>& I, std::vector<int*>& G, std::vector<ITYPE*>& result, int radius_spacial, int radius_temporal, float eps2, int fRange, Helper::SizeInfo& sizeInfo, int viewColumnNum);
		//I1G3, I3G3 template GTYPE = int or Helper::DeviceArray<int>
		template<typename ITYPE>
		static void filter4DGPU(std::vector<ITYPE*>& I, std::vector<Helper::DeviceArray<int>*>& G, std::vector<ITYPE*>& result, int radius_spacial, int radius_temporal, float eps2, int fRange, Helper::SizeInfo& sizeInfo, int viewColumnNum);

#endif
	};





#if 0

	//IXNGYN ITYPE = int or Helper::DeviceArray<int>
	template<typename ITYPE>
	static void WMF::filter2DGPU_MultiChannel(ITYPE*& I, Helper::DeviceArray<int>*& G, ITYPE*& result, int radius, float eps2, int fRange, Helper::SizeInfo& sizeInfo)
	{
		int yn = G->arrayLength;
		Helper::DeviceArray<float>* cxdx = new Helper::DeviceArray<float>(yn + 1, sizeInfo);
		Helper::DeviceArray<int>* sumG = new Helper::DeviceArray<int>(yn, sizeInfo);
		Helper::DeviceArray<int>* tempG = new Helper::DeviceArray<int>(yn, sizeInfo);
		Helper::DeviceArray<int>* sumGG = new Helper::DeviceArray<int>((yn + 1) * yn / 2, sizeInfo);
		Helper::DeviceArray<int>* tempGG = new Helper::DeviceArray<int>((yn + 1) * yn / 2, sizeInfo);
		int pixelNumInWindow = (radius * 2 + 1) * (radius * 2 + 1);
		//cx, dx�v�Z
		cu_calculateCxXDxFromG(sizeInfo, NULL, G, radius, pixelNumInWindow, eps2, cxdx, sumG, sumGG, tempG, tempGG);

		//median�v�Z
		FGMF_GPU_Or_ver0::cu_filter2DMultiChannel(sizeInfo, NULL, radius, fRange, result, I, G, cxdx);


		int* tmp2;
		UtilityForCUDA::allocateDeviceMemory(tmp2, sizeInfo);
		cudaFree(tmp2);

		//�������J��
		delete cxdx;
		delete sumG;
		delete tempG;
		delete sumGG;
		delete tempGG;
	}



	//////////////////////////////////////////////////////////////////////
	//3D
	//I1G1, I3G1 ITYPE = int or Helper::DeviceArray<int>
	template<typename ITYPE>
	void WMF::filter3DGPU(std::vector<ITYPE*>& I, std::vector<int*>& G, std::vector<ITYPE*>& result, int radius_spacial, int radius_temporal, float eps2, int fRange, Helper::SizeInfo& sizeInfo)
	{
		const int temporalLength = I.size();
		const int temporalMemoryLength = radius_temporal * 2 + 2;
		const int pixelNumInFrameWindow = (radius_spacial * 2 + 1) * (radius_spacial * 2 + 1);
		int pixelNumInWindow = 0;

		float2* cxdx;
		int2* sumG, * temp;
		UtilityForCUDA::allocateDeviceMemory(cxdx, sizeInfo);
		UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo);
		UtilityForCUDA::allocateDeviceMemory(temp, sizeInfo);
		std::vector<int2*> sumG_planes(temporalMemoryLength);
		for (int i = 0; i < temporalMemoryLength; i++)
			UtilityForCUDA::allocateDeviceMemory(sumG_planes[i], sizeInfo);

		//1�t���[���ڂ̏����̂��߂� radius_temporal - 1�܂ł� g���v�Z �i�������j
		if (radius_temporal != 0) {
			cu_calculateSumG(sizeInfo, NULL, G[0], radius_spacial, sumG_planes[0], temp);
			UtilityForCUDA::copyDeviceMemory(sumG_planes[0], sumG, sizeInfo, NULL);
			pixelNumInWindow += pixelNumInFrameWindow;
			for (int i = 1; i < radius_temporal; i++) {
				cu_addSumG(sizeInfo, NULL, G[i], radius_spacial, sumG_planes[i], sumG, temp);
				pixelNumInWindow += pixelNumInFrameWindow;
			}
		}
		else
		{
			//����0(=2D�ŏ������邱�ƂƓ���)�Ȃ�sumG��0�ŏ��������Ă���
			UtilityForCUDA::initializeDeviceMemoryWithZero(sumG, sizeInfo, NULL);
		}

		int currentFrame = 0;
		int remFrame = currentFrame - radius_temporal - 1;
		int addFrame = currentFrame + radius_temporal;
		//����
		for (; currentFrame < temporalLength; currentFrame++, remFrame++, addFrame++)
		{
			//�폜�����邩
			int hasRem = remFrame >= 0;
			int hasAdd = addFrame < temporalLength;

			if (hasAdd && hasRem)
			{
				//�ǉ��ƍ폜
				cu_updateSumG(sizeInfo, NULL, G[addFrame], radius_spacial, sumG_planes[addFrame % temporalMemoryLength], sumG_planes[remFrame % temporalMemoryLength], sumG, temp);
			}
			else if (hasAdd)
			{
				//�ǉ��̂�
				pixelNumInWindow += pixelNumInFrameWindow;
				cu_addSumG(sizeInfo, NULL, G[addFrame], radius_spacial, sumG_planes[addFrame % temporalMemoryLength], sumG, temp);
			}
			else if (hasRem)
			{
				//�폜�̂�
				pixelNumInWindow -= pixelNumInFrameWindow;
				cu_remSumG(sizeInfo, NULL, sumG_planes[remFrame % temporalMemoryLength], sumG);
			}
			//cxdx�v�Z
			cu_calculateCxDx(sizeInfo, NULL, G[currentFrame], radius_spacial, pixelNumInWindow, eps2, cxdx, sumG);
			//median�v�Z
			std::vector<ITYPE*> I_inside(I.begin() + std::max(0, remFrame + 1), I.begin() + std::min(addFrame + 1, temporalLength));
			std::vector<int*> G_inside(G.begin() + std::max(0, remFrame + 1), G.begin() + std::min(addFrame + 1, temporalLength));
			cu_filter3D(sizeInfo, NULL, radius_spacial, fRange, result[currentFrame], I_inside, G_inside, cxdx);
		}

		//�������J��
		cudaFree(cxdx);
		cudaFree(sumG);
		cudaFree(temp);
		for (int i = 0; i < temporalMemoryLength; i++)
			cudaFree(sumG_planes[i]);
	}

	//I1G3, I3G3 ITYPE = int or Helper::DeviceArray<int>
	template<typename ITYPE>
	void WMF::filter3DGPU(std::vector<ITYPE*>& I, std::vector<Helper::DeviceArray<int>*>& G, std::vector<ITYPE*>& result, int radius_spacial, int radius_temporal, float eps2, int fRange, Helper::SizeInfo& sizeInfo)
	{
		/*
		* //�K�C�h�ɕ�����������
		std::vector<Helper::DeviceArray<int>*> tmp(I.size());
		for (int i = 0; i < I.size(); i++)
		{
			Helper::DeviceArray<int>* tt = new Helper::DeviceArray<int>(3, sizeInfo);
			Utility::copyDeviceMemory(G[i], tt, 3, sizeInfo);
			WMF::filter2DGPU(G[i], G[i], tt, radius_spacial, eps2, fRange, sizeInfo);
			//Utility::showDevice(tt, sizeInfo, "GGG", false, 256);
			Utility::copyDeviceMemory(tt, G[i], 3, sizeInfo);
			//Utility::showDevice(G[i], sizeInfo, "GGG", false, 256);
		}
		*/

		const int temporalLength = I.size();
		const int temporalMemoryLength = radius_temporal * 2 + 2;
		const int pixelNumInFrameWindow = (radius_spacial * 2 + 1) * (radius_spacial * 2 + 1);
		int pixelNumInWindow = 0;

		float4* cxdx;
		UtilityForCUDA::allocateDeviceMemory(cxdx, sizeInfo);
		Helper::DeviceArray<int>* sumG = new Helper::DeviceArray<int>(3, sizeInfo, true);
		Helper::DeviceArray<int>* tempG = new Helper::DeviceArray<int>(3, sizeInfo);
		Helper::DeviceArray<int>* sumGG = new Helper::DeviceArray<int>(6, sizeInfo, true);
		Helper::DeviceArray<int>* tempGG = new Helper::DeviceArray<int>(6, sizeInfo);
		std::vector<Helper::DeviceArray<int>*> sumG_planes(temporalMemoryLength);
		std::vector<Helper::DeviceArray<int>*> sumGG_planes(temporalMemoryLength);
		for (int i = 0; i < temporalMemoryLength; i++)
		{
			sumG_planes[i] = new Helper::DeviceArray<int>(3, sizeInfo, true);
			sumGG_planes[i] = new Helper::DeviceArray<int>(6, sizeInfo, true);
		}

		//1�t���[���ڂ̏����̂��߂� radius_temporal - 1�܂ł� g���v�Z �i�������j
		if (radius_temporal != 0) {
			cu_calculateSumG3(sizeInfo, NULL, G[0], radius_spacial, sumG_planes[0], sumGG_planes[0], tempG, tempGG);
			for (int i = 0; i < 3; i++)
				UtilityForCUDA::copyDeviceMemory(sumG_planes[0]->host[i], sumG->host[i], sizeInfo, NULL);
			for (int i = 0; i < 6; i++)
				UtilityForCUDA::copyDeviceMemory(sumGG_planes[0]->host[i], sumGG->host[i], sizeInfo, NULL);
			pixelNumInWindow += pixelNumInFrameWindow;
			for (int i = 1; i < radius_temporal; i++) {
				cu_addSumG3(sizeInfo, NULL, G[i], radius_spacial, sumG_planes[i], sumGG_planes[i], sumG, sumGG, tempG, tempGG);
				pixelNumInWindow += pixelNumInFrameWindow;
			}
		}

		int currentFrame = 0;
		int remFrame = currentFrame - radius_temporal - 1;
		int addFrame = currentFrame + radius_temporal;
		//����
		for (; currentFrame < temporalLength; currentFrame++, remFrame++, addFrame++)
		{
			//�폜�����邩
			int hasRem = remFrame >= 0;
			int hasAdd = addFrame < temporalLength;

			if (hasAdd && hasRem)
			{
				cu_updateSumG3(sizeInfo, NULL, G[addFrame], radius_spacial, sumG_planes[addFrame % temporalMemoryLength], sumGG_planes[addFrame % temporalMemoryLength], sumG_planes[remFrame % temporalMemoryLength], sumGG_planes[remFrame % temporalMemoryLength], sumG, sumGG, tempG, tempGG);
			}
			else if (hasAdd)
			{
				pixelNumInWindow += pixelNumInFrameWindow;
				cu_addSumG3(sizeInfo, NULL, G[addFrame], radius_spacial, sumG_planes[addFrame % temporalMemoryLength], sumGG_planes[addFrame % temporalMemoryLength], sumG, sumGG, tempG, tempGG);
			}
			else if (hasRem)
			{
				pixelNumInWindow -= pixelNumInFrameWindow;
				cu_remSumG3(sizeInfo, NULL, sumG_planes[remFrame % temporalMemoryLength], sumGG_planes[remFrame % temporalMemoryLength], sumG, sumGG);
			}
			//cxdx�v�Z
			cu_calculateCx3Dx(sizeInfo, NULL, G[currentFrame], radius_spacial, pixelNumInWindow, eps2, cxdx, sumG, sumGG);

			//median�v�Z
			std::vector<ITYPE*> I_inside(I.begin() + std::max(0, remFrame + 1), I.begin() + std::min(addFrame + 1, temporalLength));
			std::vector<Helper::DeviceArray<int>*> G_inside(G.begin() + std::max(0, remFrame + 1), G.begin() + std::min(addFrame + 1, temporalLength));
			cu_filter3D(sizeInfo, NULL, radius_spacial, fRange, result[currentFrame], I_inside, G_inside, cxdx);
		}
		//�������J��
		cudaFree(cxdx);
	}


	//////////////////////////////////////////////////////////////////////
	//4D
	//I1G1, I3G1 ITYPE = int or Helper::DeviceArray<int>
	template<typename ITYPE>
	void WMF::filter4DGPU(std::vector<ITYPE*>& I, std::vector<int*>& G, std::vector<ITYPE*>& result, int radius_spacial, int radius_temporal, float eps2, int fRange, Helper::SizeInfo& sizeInfo, int viewColumnNum)
	{
		//3D�̘g�g�݂��g�����Ď�������
		/*
		�f�[�^�����_�ԍ���(�X�L�������C��)��1��ɕ��ׂ��x�N�g���i��3D�Ɠ����f�[�^�`��)�Ƃ���B
		�������́A���㎋�_����A���_�������ɏ���������̂Ƃ���B
		�Ⴆ�Ύ��_index��
		 1  2  3  4  5
		 6  7  8  9 10
		11 12 13 14 15
		16 17 18 19 20
		21 22 23 24 25
		�̏ꍇ�A���_���a(radius_temporal)��2���Ƃ��āA
		1,2,3,6,7,8,11,12,13�@���܂މ摜�Q�ɑ΂���1�̒����l���v�Z����B���̌㎋�_�E�B���h�E�͉��ɃX���C�h���A
		6,7,8,11,12,13,16,17,18�ɑ΂��ď�������B
		*/

		const int temporalLength = I.size();//���_��
		const int viewRowNum = temporalLength / viewColumnNum;//�c�������_��
		const int temporalMemoryLength = (radius_temporal * 2 + 1) * (radius_temporal * 2 + 1) + 1;
		const int pixelNumInFrameWindow = (radius_spacial * 2 + 1) * (radius_spacial * 2 + 1);

		float2* cxdx;
		int2* sumG, * temp;
		UtilityForCUDA::allocateDeviceMemory(cxdx, sizeInfo);
		UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo);
		UtilityForCUDA::allocateDeviceMemory(temp, sizeInfo);
		std::vector<int2*> sumG_planes(temporalMemoryLength);
		for (int i = 0; i < temporalMemoryLength; i++)
			UtilityForCUDA::allocateDeviceMemory(sumG_planes[i], sizeInfo);


		//�e���_�񂲂Ƃɏ�������
		for (int v = 0; v < viewColumnNum; v++)//���_�����J��Ԃ�
		{
			int pixelNumInWindow = 0;

			//���_���E�[�C���f�b�N�X
			const int viewLeftBound = std::max(0, v - radius_temporal);
			const int viewRightBound = std::min(viewColumnNum - 1, v + radius_temporal);
			//���_���a����I,G���i�[
			std::vector<ITYPE*> I_inside;
			std::vector<int*> G_inside;

			//sumG�̏�����
			UtilityForCUDA::initializeDeviceMemoryWithZero(sumG, sizeInfo, NULL);
			//sumG_planes�̒ǉ��A�폜�p�C���f�b�N�X
			int sumG_index_add = 0;
			int sumG_index_rem = 0;

			//���_1�s�ڂ̏����̂��߂� �����radius_temporal - 1�܂ł� g���v�Z �i�������j
			if (radius_temporal != 0) {
				for (int j = 0; j < radius_temporal; j++) {
					//���_�s�����͍��Eradius_temporal�܂œǂݍ��ށi�X���C�h�����������������j
					for (int i = viewLeftBound; i <= viewRightBound; i++) {
						const int viewIndex = j * viewColumnNum + i;//���_index
						cu_addSumG(sizeInfo, NULL, G[viewIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumG, temp);
						sumG_index_add++;
						pixelNumInWindow += pixelNumInFrameWindow;

						I_inside.push_back(I[viewIndex]);
						G_inside.push_back(G[viewIndex]);
					}
				}

			}//����0(=2D�ŏ������邱�ƂƓ���)�Ȃ�sumG�͏�������Ԃɂ��Ă���


			int currentRow = 0;
			int remRow = currentRow - radius_temporal - 1;
			int addRow = currentRow + radius_temporal;
			//���_������X���C�h����
			for (; currentRow < viewRowNum; currentRow++, remRow++, addRow++)
			{
				//�������S���_�C���f�b�N�X
				int currentIndex = currentRow * viewColumnNum + v;

				//�폜�����邩
				int hasRem = remRow >= 0;
				int hasAdd = addRow < viewRowNum;

				if (hasAdd && hasRem)
				{
					//�ǉ��ƍ폜
					for (int i = viewLeftBound; i <= viewRightBound; i++) {
						const int addIndex = addRow * viewColumnNum + i;
						const int remIndex = remRow * viewColumnNum + i;
						cu_updateSumG(sizeInfo, NULL, G[addIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumG_planes[sumG_index_rem % temporalMemoryLength], sumG, temp);
						sumG_index_add++;
						sumG_index_rem++;

						I_inside.push_back(I[addIndex]);
						G_inside.push_back(G[addIndex]);
						I_inside.erase(I_inside.begin());
						G_inside.erase(G_inside.begin());
					}
				}
				else if (hasAdd)
				{
					//�ǉ��̂�
					for (int i = viewLeftBound; i <= viewRightBound; i++) {
						const int addIndex = addRow * viewColumnNum + i;
						pixelNumInWindow += pixelNumInFrameWindow;
						cu_addSumG(sizeInfo, NULL, G[addIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumG, temp);
						sumG_index_add++;

						I_inside.push_back(I[addIndex]);
						G_inside.push_back(G[addIndex]);
					}
				}
				else if (hasRem)
				{
					//�폜�̂�
					for (int i = viewLeftBound; i <= viewRightBound; i++) {
						const int remIndex = remRow * viewColumnNum + i;
						pixelNumInWindow -= pixelNumInFrameWindow;
						cu_remSumG(sizeInfo, NULL, sumG_planes[sumG_index_rem % temporalMemoryLength], sumG);
						sumG_index_rem++;

						I_inside.erase(I_inside.begin());
						G_inside.erase(G_inside.begin());
					}
				}
				//cxdx�v�Z
				cu_calculateCxDx(sizeInfo, NULL, G[currentIndex], radius_spacial, pixelNumInWindow, eps2, cxdx, sumG);

				//median�v�Z
				cu_filter3D(sizeInfo, NULL, radius_spacial, fRange, result[currentIndex], I_inside, G_inside, cxdx);
			}
		}
		//�������J��
		cudaFree(cxdx);
		cudaFree(sumG);
		cudaFree(temp);
		for (int i = 0; i < temporalMemoryLength; i++)
			cudaFree(sumG_planes[i]);
	}



	//I1G3, I3G3 ITYPE = int or Helper::DeviceArray<int>
	template<typename ITYPE>
	void WMF::filter4DGPU(std::vector<ITYPE*>& I, std::vector<Helper::DeviceArray<int>*>& G, std::vector<ITYPE*>& result, int radius_spacial, int radius_temporal, float eps2, int fRange, Helper::SizeInfo& sizeInfo, int viewColumnNum)
	{
		//3D�̘g�g�݂��g�����Ď�������
		/*
		�f�[�^�����_�ԍ���(�X�L�������C��)��1��ɕ��ׂ��x�N�g���i��3D�Ɠ����f�[�^�`��)�Ƃ���B
		�������́A���㎋�_����A���_�������ɏ���������̂Ƃ���B
		�Ⴆ�Ύ��_index��
		 1  2  3  4  5
		 6  7  8  9 10
		11 12 13 14 15
		16 17 18 19 20
		21 22 23 24 25
		�̏ꍇ�A���_���a(radius_temporal)��2���Ƃ��āA
		1,2,3,6,7,8,11,12,13�@���܂މ摜�Q�ɑ΂���1�̒����l���v�Z����B���̌㎋�_�E�B���h�E�͉��ɃX���C�h���A
		6,7,8,11,12,13,16,17,18�ɑ΂��ď�������B
		*/

		const int temporalLength = I.size();//���_��
		const int viewRowNum = temporalLength / viewColumnNum;//�c�������_��
		const int temporalMemoryLength = (radius_temporal * 2 + 1) * (radius_temporal * 2 + 1) + 1;
		const int pixelNumInFrameWindow = (radius_spacial * 2 + 1) * (radius_spacial * 2 + 1);

		//float2* cxdx;
		//int2* sumG, * temp;
		//UtilityForCUDA::allocateDeviceMemory(cxdx, sizeInfo);
		//UtilityForCUDA::allocateDeviceMemory(sumG, sizeInfo);
		//UtilityForCUDA::allocateDeviceMemory(temp, sizeInfo);
		//std::vector<int2*> sumG_planes(temporalMemoryLength);
		float4* cxdx;
		UtilityForCUDA::allocateDeviceMemory(cxdx, sizeInfo);
		Helper::DeviceArray<int>* tempG = new Helper::DeviceArray<int>(3, sizeInfo);
		Helper::DeviceArray<int>* tempGG = new Helper::DeviceArray<int>(6, sizeInfo);
		std::vector<Helper::DeviceArray<int>*> sumG_planes(temporalMemoryLength);
		std::vector<Helper::DeviceArray<int>*> sumGG_planes(temporalMemoryLength);
		for (int i = 0; i < temporalMemoryLength; i++) {
			//UtilityForCUDA::allocateDeviceMemory(sumG_planes[i], sizeInfo);
			sumG_planes[i] = new Helper::DeviceArray<int>(3, sizeInfo);
			sumGG_planes[i] = new Helper::DeviceArray<int>(6, sizeInfo);
		}


		//�e���_�񂲂Ƃɏ�������
		for (int v = 0; v < viewColumnNum; v++)//���_�����J��Ԃ�
		{

			int pixelNumInWindow = 0;

			//���_���E�[�C���f�b�N�X
			const int viewLeftBound = std::max(0, v - radius_temporal);
			const int viewRightBound = std::min(viewColumnNum - 1, v + radius_temporal);
			//���_���a����I,G���i�[
			std::vector<ITYPE*> I_inside;
			std::vector<Helper::DeviceArray<int>*> G_inside;
			//sumG�̏�����(�錾�@�����̖ʓ|������AdeviceArray�̐錾���̏������𗘗p�i���邽�ߖ���new����j)
			Helper::DeviceArray<int>* sumG = new Helper::DeviceArray<int>(3, sizeInfo, true);
			Helper::DeviceArray<int>* sumGG = new Helper::DeviceArray<int>(6, sizeInfo, true);
			//UtilityForCUDA::initializeDeviceMemoryWithZero(sumG, sizeInfo, NULL);
			//UtilityForCUDA::initializeDeviceMemoryWithZero(sumGG, sizeInfo, NULL);
			// 
			//sumG_planes�̒ǉ��A�폜�p�C���f�b�N�X
			int sumG_index_add = 0;
			int sumG_index_rem = 0;


			//���_1�s�ڂ̏����̂��߂� �����radius_temporal - 1�܂ł� g���v�Z �i�������j
			if (radius_temporal != 0) {
				for (int j = 0; j < radius_temporal; j++) {
					//���_�s�����͍��Eradius_temporal�܂œǂݍ��ށi�X���C�h�����������������j
					for (int i = viewLeftBound; i <= viewRightBound; i++) {
						const int viewIndex = j * viewColumnNum + i;//���_index
						//cu_addSumG(sizeInfo, NULL, G[viewIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumG, tempG);
						cu_addSumG3(sizeInfo, NULL, G[viewIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumGG_planes[sumG_index_add % temporalMemoryLength], sumG, sumGG, tempG, tempGG);
						sumG_index_add++;
						pixelNumInWindow += pixelNumInFrameWindow;

						I_inside.push_back(I[viewIndex]);
						G_inside.push_back(G[viewIndex]);
					}
				}

			}//����0(=2D�ŏ������邱�ƂƓ���)�Ȃ�sumG�͏�������Ԃɂ��Ă���



			int currentRow = 0;
			int remRow = currentRow - radius_temporal - 1;
			int addRow = currentRow + radius_temporal;
			//���_������X���C�h����
			for (; currentRow < viewRowNum; currentRow++, remRow++, addRow++)
			{
				//�������S���_�C���f�b�N�X
				int currentIndex = currentRow * viewColumnNum + v;

				//�폜�����邩
				int hasRem = remRow >= 0;
				int hasAdd = addRow < viewRowNum;

				if (hasAdd && hasRem)
				{
					//�ǉ��ƍ폜
					for (int i = viewLeftBound; i <= viewRightBound; i++) {
						const int addIndex = addRow * viewColumnNum + i;
						const int remIndex = remRow * viewColumnNum + i;
						//cu_updateSumG(sizeInfo, NULL, G[addIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumG_planes[sumG_index_rem % temporalMemoryLength], sumG, tempG);
						cu_updateSumG3(sizeInfo, NULL, G[addIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumGG_planes[sumG_index_add % temporalMemoryLength], sumG_planes[sumG_index_rem % temporalMemoryLength], sumGG_planes[sumG_index_rem % temporalMemoryLength], sumG, sumGG, tempG, tempGG);
						sumG_index_add++;
						sumG_index_rem++;

						I_inside.push_back(I[addIndex]);
						G_inside.push_back(G[addIndex]);
						I_inside.erase(I_inside.begin());
						G_inside.erase(G_inside.begin());
					}
				}
				else if (hasAdd)
				{
					//�ǉ��̂�
					for (int i = viewLeftBound; i <= viewRightBound; i++) {
						const int addIndex = addRow * viewColumnNum + i;
						pixelNumInWindow += pixelNumInFrameWindow;
						//cu_addSumG(sizeInfo, NULL, G[addIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumG, tempG);
						cu_addSumG3(sizeInfo, NULL, G[addIndex], radius_spacial, sumG_planes[sumG_index_add % temporalMemoryLength], sumGG_planes[sumG_index_add % temporalMemoryLength], sumG, sumGG, tempG, tempGG);
						sumG_index_add++;

						I_inside.push_back(I[addIndex]);
						G_inside.push_back(G[addIndex]);
					}
				}
				else if (hasRem)
				{
					//�폜�̂�
					for (int i = viewLeftBound; i <= viewRightBound; i++) {
						const int remIndex = remRow * viewColumnNum + i;
						pixelNumInWindow -= pixelNumInFrameWindow;
						//cu_remSumG(sizeInfo, NULL, sumG_planes[sumG_index_rem % temporalMemoryLength], sumG);
						cu_remSumG3(sizeInfo, NULL, sumG_planes[sumG_index_rem % temporalMemoryLength], sumGG_planes[sumG_index_rem % temporalMemoryLength], sumG, sumGG);
						sumG_index_rem++;

						I_inside.erase(I_inside.begin());
						G_inside.erase(G_inside.begin());
					}
				}
				//cxdx�v�Z
				//cu_calculateCxDx(sizeInfo, NULL, G[currentIndex], radius_spacial, pixelNumInWindow, eps2, cxdx, sumG);
				cu_calculateCx3Dx(sizeInfo, NULL, G[currentIndex], radius_spacial, pixelNumInWindow, eps2, cxdx, sumG, sumGG);

				//median�v�Z
				cu_filter3D(sizeInfo, NULL, radius_spacial, fRange, result[currentIndex], I_inside, G_inside, cxdx);
			}
			delete sumG;
			delete sumGG;
		}
		//�������J��
		cudaFree(cxdx);
		delete tempG;
		delete tempGG;
		for (int i = 0; i < temporalMemoryLength; i++)
		{
			delete sumG_planes[i];
			delete sumGG_planes[i];
		}

		/*
		cudaFree(sumG);
		cudaFree(temp);
		for (int i = 0; i < temporalMemoryLength; i++)
			cudaFree(sumG_planes[i]);
			*/
	}






	//�쐬���@�܂��͒��~

	//CPU-O(r) ����1�`�����l��
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	cv::Mat WMF::filter2D_CPU_Or_AblationStudy(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int fRange)
	{

		const int Ichannels = I.channels();

		//check validation
		assert(I.depth() == CV_32S);
		assert(G.depth() == CV_32S);
#if defined(USE_AVX2)
		assert(fRange % 4 == 0);
#elif defined(USE_AVX512)
		assert(fRange % 8 == 0);
#endif


		//���͎���
		const int DIM = 2;
		//�����l
		const float half = 0.5f;
		//������
		//�T�C�Y
		const std::vector<int> size_dim{ I.cols , I.rows };
		//���a
		const std::vector<int> r_dim{ radius, radius };

		//�}���`�X���b�h�p
		std::vector<int> dim0Start_vec(threadNum);//dim0�����J�n�ʒu
		std::vector<int> dim0End_vec(threadNum);//�����I���ʒu
		std::vector<int> memoryLength_vec(threadNum);//������dim0��������
		std::vector<int> insideImageStart_vec(threadNum);//���W�A���v�Z�Ώۉ�f�J�n�ʒu
		calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

		//�������m�ہE������
		cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

		//W1
		std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
		//W2(Wmain)
		std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			std::vector<int> size_dim_memory;
			copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
			size_dim_memory[0] = memoryLength_vec[k];
			W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(fRange, size_dim_memory, 1, true)));
			Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(fRange)));
		}

#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			//��f��
			std::vector<int> pixel_sum(DIM);
			//�ʒu
			std::vector<Pos> x(DIM);
			//�X�e�[�^�X
			std::vector<DimStatus> status(DIM);

			const int dim0Start = dim0Start_vec[k];
			const int dim0End = dim0End_vec[k];
			const int insideImageStart = insideImageStart_vec[k];


			//�Ή������f�ւ̃|�C���^�i�����ɂ���Đݒ肪�قȂ邱�Ƃɒ��Ӂj
			GTYPE* G_center_rowStart = G.ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* result_center_rowStart = result.ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* W0_rem_f_rowStart = I.ptr<int>(0) + dim0Start + r_dim[0] - (2 * r_dim[1] + 1) * size_dim[0];
			GTYPE* W0_rem_g_rowStart = G.ptr<GTYPE>(0) + dim0Start + r_dim[0] - (2 * r_dim[1] + 1) * size_dim[0];
			int* W0_add_f_rowStart = I.ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_add_g_rowStart = G.ptr<GTYPE>(0) + dim0Start + r_dim[0];


			//���̊K�w�̏����ʒu�Z�b�g
			setPos(x[1], r_dim[1]);
			for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
			{
				//�X�e�[�^�X�Z�b�g
				setStatusAtOutermostLoop(x[1], size_dim[1], status[1]);
				//1�񓖂���̉�f��
				pixel_sum[1] = calculatePixelNumAtOutermostLoop(x[1], size_dim[1], r_dim[1], status[1]);


				//������
				//W2
				Wmain[k]->initialize();
				//�E�B���h�E����f��
				int pixel_sum_window = 0;
				//�E�B���h�E����f���̋t��
				float pixel_sum_window_inv = 0.0f;


				//��f�ւ̃|�C���^������
				GTYPE* G_center = G_center_rowStart;
				int* result_center = result_center_rowStart;
				int* W0_rem_f = W0_rem_f_rowStart;
				GTYPE* W0_rem_g = W0_rem_g_rowStart;
				int* W0_add_f = W0_add_f_rowStart;
				GTYPE* W0_add_g = W0_add_g_rowStart;



				//W1�������ʒu���Z�b�g
				W1[k]->resetPos();

				//���̊K�w�̏����ʒu�Z�b�g
				setPosAtDim0(x[0], r_dim[0], dim0Start);
				const int remStartPos = x[0].add;
				for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
				{
					//�X�e�[�^�X�Z�b�g
					setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);

					//window����f���̍X�V
					calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

					//W2�̍X�V
					if (status[0].hasAdd)
					{
						//W1[x[0].add] �̍X�V
						if (status[1].hasAdd) //��f�ǉ�	(W1[x[0].add]) + W0[x[1].add, x[0].add]
							addPixelToWindow_gSum(*W1[k], W0_add_f, W0_add_g);
						if (status[1].hasRem)//��f�폜	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
							removePixelFromWindow_gSum(*W1[k], W0_rem_f, W0_rem_g);
						//W1�ǉ�	(W2) + W1[x[0].add]
						updateSubWindowAndAddToWindow_gSum(fRange, *Wmain[k], *W1[k]);
					}
					if (status[0].hasRem)//W1�폜	(W2) - W1[x[0].rem]
						removeSubWindowFromWindow_gSum(fRange, *Wmain[k], *W1[k]);

					//�����l�̌v�Z
					if (status[0].isInside_image)
					{
						CTYPE cx;
						float dx;
						calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
						findMedian(cx, dx, half, *Wmain[k], *result_center);
						G_center++;
						result_center++;
					}
				}
				G_center_rowStart += size_dim[0];
				result_center_rowStart += size_dim[0];
				W0_add_f_rowStart += size_dim[0];
				W0_add_g_rowStart += size_dim[0];
				W0_rem_f_rowStart += size_dim[0];
				W0_rem_g_rowStart += size_dim[0];
			}
		}

		return result;

	}

	//CPU-O(r) median tracking ����
	template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
	cv::Mat WMF::filter2D_CPU_Or_woMedianTracking_AblationStudy(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int fRange)
	{
		return c::Mat();
	}




#endif

}