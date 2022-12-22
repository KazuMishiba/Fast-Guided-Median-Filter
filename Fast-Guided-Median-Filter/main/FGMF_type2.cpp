#include "FGMF_type2.h"

cv::Mat FGMF2::filter2DInterface(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax) {
	//���͉摜�`�����l����
	const int IchannelNum = I.channels();
	//�K�C�h�摜�`�����l����
	const int GchannelNum = G.channels();

	//return FGMF2::filter2DWindow<gSum, fgSumUpToIndex, fg, int, float>(I, G, threadNum, radius, eps2, Imax);
	
	
	if (IchannelNum == 1 && GchannelNum == 1)
		return FGMF2::filter2DWindow<gSum, fgSumUpToIndex, fg, int, float>(I, G, threadNum, radius, eps2, Imax);
	else if (IchannelNum == 1 && GchannelNum == 3)
		return FGMF2::filter2DWindow<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(I, G, threadNum, radius, eps2, Imax);
	else if (IchannelNum == 3 && GchannelNum == 1) {

		return FGMF2::filter2DWindowI3<gSum, fgSumUpToIndex, fg, int, float>(CV_32FC1, I, G, threadNum, radius, eps2, Imax);
	}
	else if (IchannelNum == 3 && GchannelNum == 3) {
		return FGMF2::filter2DWindowI3<g3Sum, fg3SumUpToIndex, fg3, cv::Vec3i, cv::Vec3f>(CV_32FC3, I, G, threadNum, radius, eps2, Imax);
	}
	return cv::Mat();
}



#if 0
//GPU�g�p�e�X�g
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
std::vector<cv::Mat> FGMF2::filter3DI1withGPU(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, SizeInfo sizeInfo, int radius_space, int radius_depth, float eps2, int Imax)
{
	//check validation
	assert(I[0].depth() == CV_32S && I[0].channels() == 1);
	assert(G[0].depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif



	//���͎���
	const int DIM = 3;
	//���͉摜�`�����l����
	const int Ichannels = I[0].channels();
	//�����l
	const float half = 0.5f;
	//������
	//�T�C�Y
	const std::vector<int> size_dim{ I[0].cols , I[0].rows, (int)I.size() };
	//���a
	const std::vector<int> r_dim{ radius_space, radius_space, radius_depth };


	//GPU�v�Z�p
	cv::Mat cx;// = cv::Mat(size_dim[1], size_dim[0], CV_32FC(G[0].channels));
	cv::Mat dx;// = cv::Mat(size_dim[1], size_dim[0], CV_32FC(1));
	//mat GMat �� int* G�ɃR�s�[
	int* GDevice;
	float* cxDevice;//������CTYPE
	float* dxDevice;
	std::vector<int4*> sumG(size_dim[2]);//������r_dim[2]*2+1 or 2�̃T�C�Y�ɏk��
	int4* sumGwindow;//window�����v�i�[�p
	int4* temp;
	//�ʃX���b�h�Ŏ��s
	Utility::allocateDeviceMemory(GDevice, G[0], sizeInfo);
	//Utility::showDevice(G, sizeInfo, "test", false, 255);
	//float* cx, dx���m��
	Utility::allocateDeviceMemory(cxDevice, sizeInfo);
	Utility::allocateDeviceMemory(dxDevice, sizeInfo);
	Utility::allocateDeviceMemory(sumG[0], sizeInfo);
	Utility::allocateDeviceMemory(sumGwindow, 0, sizeInfo);
	Utility::allocateDeviceMemory(temp, sizeInfo);
	cu_addSumG(sizeInfo, NULL, GDevice, radius_space, sumG[0], sumGwindow, temp);
	//cx,dx��cxMat, dxMat�ɃR�s�[
	cx = Utility::downloadLinearArrayAsMat(cxDevice, sizeInfo);
	dx = Utility::downloadLinearArrayAsMat(dxDevice, sizeInfo);




	//�}���`�X���b�h�p
	std::vector<int> dim0Start_vec(threadNum);
	std::vector<int> dim0End_vec(threadNum);
	std::vector<int> memoryLength_vec(threadNum);
	std::vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//�������m�ہE������
	//���ʕۑ�
	std::vector<cv::Mat> result(I.size());
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = cv::Mat(size_dim[1], size_dim[0], CV_32SC(Ichannels));
	}

	// (gSum, sumUpToIndex, histo)
	//W0:��f�Ȃ̂ŋL�^�̕K�v�Ȃ��iI,G�j
	//W1
	std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
	//= Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 1);
	//W2
	std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W2(threadNum);
	//Window_vector<GSum, FGSumUpToIndex, FG> W2 = Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 2);
	//W3(Wmain)
	std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		std::vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, false)));
		W2[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 2, false)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}






	//���̊K�w�̏����ʒu�Z�b�g
	Pos x2;
	DimStatus status2, status2GPU;
	int pixelSum2;
	setPos(x2, r_dim[2]);
	for (; x2.center < size_dim[2]; x2.center++, x2.add++, x2.rem++)
	{
		//�X�e�[�^�X�Z�b�g
		setStatusAtOutermostLoop(x2, size_dim[2], status2);
		//1�񓖂���̉�f��
		int pixel_sum2 = calculatePixelNumAtOutermostLoop(x2, size_dim[2], r_dim[2], status2);

		Pos vecPos;
		vecPos.center = (std::max)(0, x2.center);
		vecPos.add = (std::min)(size_dim[2] - 1, x2.add);
		vecPos.rem = (std::max)(0, x2.rem);

		//GPU�p
		setNextStatus(x2, size_dim[2], 1, status2GPU);


#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			//��f��
			std::vector<int> pixel_sum(2);
			//�ʒu
			std::vector<Pos> x(2);
			//�X�e�[�^�X
			std::vector<DimStatus> status(2);


			const int dim0Start = dim0Start_vec[k];
			const int dim0End = dim0End_vec[k];
			const int insideImageStart = insideImageStart_vec[k];


			//�Ή������f�ւ̃|�C���^�i�����ɂ���Đݒ肪�قȂ邱�Ƃɒ��Ӂj
			//3�����ȍ~�ɂ��ẮA��f�̍��W�͓��������Avector�̂ǂ�cv::Mat�����o���̂����ς��
			GTYPE* G_center_rowStart = G[vecPos.center].ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* result_center_rowStart = result[vecPos.center].ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
			CTYPE* cx_center_rowStart = cx.ptr<CTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			float* dx_center_rowStart = dx.ptr<float>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* W0_rem_f_rowStart = I[vecPos.rem].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_rem_g_rowStart = G[vecPos.rem].ptr<GTYPE>(0) + dim0Start + r_dim[0];
			int* W0_add_f_rowStart = I[vecPos.add].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_add_g_rowStart = G[vecPos.add].ptr<GTYPE>(0) + dim0Start + r_dim[0];

			//W1�������ʒu���Z�b�g
			W1[k]->resetPos();
			//���̊K�w�̏����ʒu�Z�b�g
			setPos(x[1], r_dim[1]);
			for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
			{
				//�X�e�[�^�X�Z�b�g
				setStatus(x[1], size_dim[1], status2.isInside_image, status[1]);
				//1�񓖂���̉�f��
				pixel_sum[1] = calculatePixelNumAtDim(x[1], size_dim[1], r_dim[1], status[1], pixel_sum2);


				//������
				//W3
				Wmain[k]->initialize();
				//�E�B���h�E����f��
				int pixel_sum_window = 0;
				//�E�B���h�E����f���̋t��
				float pixel_sum_window_inv = 0.0f;

				//��f�ւ̃|�C���^������
				GTYPE* G_center = G_center_rowStart;
				int* result_center = result_center_rowStart;
				CTYPE* cx_center = cx_center_rowStart;
				float* dx_center = dx_center_rowStart;
				int* W0_rem_f = W0_rem_f_rowStart;
				GTYPE* W0_rem_g = W0_rem_g_rowStart;
				int* W0_add_f = W0_add_f_rowStart;
				GTYPE* W0_add_g = W0_add_g_rowStart;


				//W2�������ʒu���Z�b�g
				W2[k]->resetPos();
				//���̊K�w�̏����ʒu�Z�b�g
				setPosAtDim0(x[0], r_dim[0], dim0Start);
				const int remStartPos = x[0].add;
				for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
				{
					//�X�e�[�^�X�Z�b�g
					setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);

					//window����f���̍X�V
					calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

					//W3�̍X�V
					//W3 = W3 + W2[x[0].add] - W2[x[0].rem](W3��window)
					if (status[0].hasAdd)
					{
						//W2�̍X�V
						//W2[x[0].add] = W2[x[0].add] + W1[x[1].add, x[0].add] - W1[x[1].rem, x[0].add]
						if (status[1].hasAdd)
						{
							//W1[x[0].add] �̍X�V
							//W1[x[1].add, x[0].add] = W1[x[1].add, x[0].add] + W0[x[2].add, x[1].add, x[0].add] - W0[x[2].rem, x[1].add, x[0].add]
							if (status2.hasAdd) //��f�ǉ�	(W1[x[0].add]) + W0[x[1].add, x[0].add]
								addPixelToWindow(*W1[k], W0_add_f, W0_add_g);
							if (status2.hasRem)//��f�폜	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
								removePixelFromWindow(*W1[k], W0_rem_f, W0_rem_g);
							//W1�ǉ�	(W2) + W1[x[0].add]
							updateSubWindowAndAddToWindow(Imax, *W2[k], *W1[k]);
						}
						if (status[1].hasRem)//W1�폜	(W2) - W1[x[0].rem]
							removeSubWindowFromWindow(Imax, *W2[k], *W1[k]);
						//W2�ǉ�	(W3) + W1[x[0].add]
						updateSubWindowAndAddToWindow(Imax, *Wmain[k], *W2[k]);
					}
					if (status[0].hasRem)//W2�폜	(W3) - W2[x[0].rem]
						removeSubWindowFromWindow(Imax, *Wmain[k], *W2[k]);

					//�����l�̌v�Z
					if (status[0].isInside_image)
					{
						/*
						CTYPE cx;
						float dx;
						calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
						*/
						findMedian(*cx_center, *dx_center, half, *Wmain[k], *result_center);
						G_center++;
						result_center++;
						cx_center++;
						dx_center++;
					}
				}
				G_center_rowStart += size_dim[0];
				result_center_rowStart += size_dim[0];
				cx_center_rowStart += size_dim[0];
				dx_center_rowStart += size_dim[0];
				W0_add_f_rowStart += size_dim[0];
				W0_add_g_rowStart += size_dim[0];
				W0_rem_f_rowStart += size_dim[0];
				W0_rem_g_rowStart += size_dim[0];
			}
			//���g�̎���+1�̃��[�v���I������Ƃ��ɃE�B���h�E���e���Z�b�g
			W2[k]->setZero();

		}

		//gsum�X�V
		if (status2GPU.hasAddOnly)
		{
			Utility::uploadMatToDevice(G[x2.add + 1], GDevice, sizeInfo, NULL);
			Utility::allocateDeviceMemory(sumG[x2.add + 1], sizeInfo);
			cu_addSumG(sizeInfo, NULL, GDevice, radius_space, sumG[x2.add + 1], sumGwindow, temp);
		}
		else if (status2GPU.hasRemOnly)
		{
			cu_remSumG(sizeInfo, NULL, GDevice, sumG[x2.rem + 1], sumGwindow);
		}
		else
		{
			Utility::uploadMatToDevice(G[x2.add + 1], GDevice, sizeInfo, NULL);
			Utility::allocateDeviceMemory(sumG[x2.add + 1], sizeInfo);
			cu_updateSumG(sizeInfo, NULL, GDevice, radius_space, sumG[x2.add + 1], sumG[x2.rem + 1], sumGwindow, temp);
		}
		//cxdx�X�V
		if (status2GPU.isInside_image)
		{
			Utility::uploadMatToDevice(G[x2.center + 1], GDevice, sizeInfo, NULL);
			cu_calculateCxDx(sizeInfo, NULL, GDevice, radius_space, eps2, cxDevice, dxDevice, sumGwindow, temp);
			Utility::downloadLinearArrayAsMat(cxDevice, sizeInfo, cx);
			Utility::downloadLinearArrayAsMat(dxDevice, sizeInfo, dx);
			/*
			cv::imshow("cx", cv::abs(cx*100000));
			cv::imshow("dx", cv::abs(dx * 100));
			cv::waitKey(0);
			*/
		}

	}

	return result;
}

//GPU�g�p�e�X�g
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
std::vector<cv::Mat> FGMF2::filter3DI1withGPUThread(std::vector<cv::Mat>& I, std::vector<cv::Mat>& G, int threadNum, SizeInfo sizeInfo, int radius_space, int radius_depth, float eps2, int Imax)
{
	//check validation
	assert(I[0].depth() == CV_32S && I[0].channels() == 1);
	assert(G[0].depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif



	//���͎���
	const int DIM = 3;
	//���͉摜�`�����l����
	const int Ichannels = I[0].channels();
	//�����l
	const float half = 0.5f;
	//������
	//�T�C�Y
	const std::vector<int> size_dim{ I[0].cols , I[0].rows, (int)I.size() };
	//���a
	const std::vector<int> r_dim{ radius_space, radius_space, radius_depth };


	//GPU�v�Z�p
	cv::Mat cx;// = cv::Mat(size_dim[1], size_dim[0], CV_32FC(G[0].channels));
	cv::Mat dx;// = cv::Mat(size_dim[1], size_dim[0], CV_32FC(1));
	//mat GMat �� int* G�ɃR�s�[
	int* GDevice;
	float* cxDevice;//������CTYPE
	float* dxDevice;
	std::vector<int4*> sumG(size_dim[2]);//������r_dim[2]*2+1 or 2�̃T�C�Y�ɏk��
	int4* sumGwindow;//window�����v�i�[�p
	int4* temp;
	//�ʃX���b�h�Ŏ��s
	std::thread th0([&]() {
		Utility::allocateDeviceMemory(GDevice, G[0], sizeInfo);
		//Utility::showDevice(G, sizeInfo, "test", false, 255);
		//float* cx, dx���m��
		Utility::allocateDeviceMemory(cxDevice, sizeInfo);
		Utility::allocateDeviceMemory(dxDevice, sizeInfo);
		Utility::allocateDeviceMemory(sumG[0], sizeInfo);
		Utility::allocateDeviceMemory(sumGwindow, 0, sizeInfo);
		Utility::allocateDeviceMemory(temp, sizeInfo);
		cu_addSumG(sizeInfo, NULL, GDevice, radius_space, sumG[0], sumGwindow, temp);
		//cx,dx��cxMat, dxMat�ɃR�s�[
		cx = Utility::downloadLinearArrayAsMat(cxDevice, sizeInfo);
		dx = Utility::downloadLinearArrayAsMat(dxDevice, sizeInfo);
	});



	//�}���`�X���b�h�p
	std::vector<int> dim0Start_vec(threadNum);
	std::vector<int> dim0End_vec(threadNum);
	std::vector<int> memoryLength_vec(threadNum);
	std::vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//�������m�ہE������
	//���ʕۑ�
	std::vector<cv::Mat> result(I.size());
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = cv::Mat(size_dim[1], size_dim[0], CV_32SC(Ichannels));
	}

	// (gSum, sumUpToIndex, histo)
	//W0:��f�Ȃ̂ŋL�^�̕K�v�Ȃ��iI,G�j
	//W1
	std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);
	//= Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 1);
	//W2
	std::vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W2(threadNum);
	//Window_vector<GSum, FGSumUpToIndex, FG> W2 = Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 2);
	//W3(Wmain)
	std::vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		std::vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, false)));
		W2[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 2, false)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}



	th0.join();


	//���̊K�w�̏����ʒu�Z�b�g
	Pos x2;
	DimStatus status2, status2GPU;
	int pixelSum2;
	setPos(x2, r_dim[2]);
	for (; x2.center < size_dim[2]; x2.center++, x2.add++, x2.rem++)
	{
		//�X�e�[�^�X�Z�b�g
		setStatusAtOutermostLoop(x2, size_dim[2], status2);
		//1�񓖂���̉�f��
		int pixel_sum2 = calculatePixelNumAtOutermostLoop(x2, size_dim[2], r_dim[2], status2);

		Pos vecPos;
		vecPos.center = (std::max)(0, x2.center);
		vecPos.add = (std::min)(size_dim[2] - 1, x2.add);
		vecPos.rem = (std::max)(0, x2.rem);

		//GPU�p
		setNextStatus(x2, size_dim[2], 1, status2GPU);


		std::thread th1([&]() {
			//gsum�X�V
			if (status2GPU.hasAddOnly)
			{
				Utility::uploadMatToDevice(G[x2.add + 1], GDevice, sizeInfo, NULL);
				Utility::allocateDeviceMemory(sumG[x2.add + 1], sizeInfo);
				cu_addSumG(sizeInfo, NULL, GDevice, radius_space, sumG[x2.add + 1], sumGwindow, temp);
			}
			else if (status2GPU.hasRemOnly)
			{
				cu_remSumG(sizeInfo, NULL, GDevice, sumG[x2.rem + 1], sumGwindow);
			}
			else
			{
				Utility::uploadMatToDevice(G[x2.add + 1], GDevice, sizeInfo, NULL);
				Utility::allocateDeviceMemory(sumG[x2.add + 1], sizeInfo);
				cu_updateSumG(sizeInfo, NULL, GDevice, radius_space, sumG[x2.add + 1], sumG[x2.rem + 1], sumGwindow, temp);
			}
			//cxdx�X�V
			if (status2GPU.isInside_image)
			{
				Utility::uploadMatToDevice(G[x2.center + 1], GDevice, sizeInfo, NULL);
				cu_calculateCxDx(sizeInfo, NULL, GDevice, radius_space, eps2, cxDevice, dxDevice, sumGwindow, temp);
			}
		});

		//th1.join();

#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			//��f��
			std::vector<int> pixel_sum(2);
			//�ʒu
			std::vector<Pos> x(2);
			//�X�e�[�^�X
			std::vector<DimStatus> status(2);


			const int dim0Start = dim0Start_vec[k];
			const int dim0End = dim0End_vec[k];
			const int insideImageStart = insideImageStart_vec[k];


			//�Ή������f�ւ̃|�C���^�i�����ɂ���Đݒ肪�قȂ邱�Ƃɒ��Ӂj
			//3�����ȍ~�ɂ��ẮA��f�̍��W�͓��������Avector�̂ǂ�cv::Mat�����o���̂����ς��
			GTYPE* G_center_rowStart = G[vecPos.center].ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* result_center_rowStart = result[vecPos.center].ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
			CTYPE* cx_center_rowStart = cx.ptr<CTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			float* dx_center_rowStart = dx.ptr<float>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* W0_rem_f_rowStart = I[vecPos.rem].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_rem_g_rowStart = G[vecPos.rem].ptr<GTYPE>(0) + dim0Start + r_dim[0];
			int* W0_add_f_rowStart = I[vecPos.add].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_add_g_rowStart = G[vecPos.add].ptr<GTYPE>(0) + dim0Start + r_dim[0];

			//W1�������ʒu���Z�b�g
			W1[k]->resetPos();
			//���̊K�w�̏����ʒu�Z�b�g
			setPos(x[1], r_dim[1]);
			for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
			{
				//�X�e�[�^�X�Z�b�g
				setStatus(x[1], size_dim[1], status2.isInside_image, status[1]);
				//1�񓖂���̉�f��
				pixel_sum[1] = calculatePixelNumAtDim(x[1], size_dim[1], r_dim[1], status[1], pixel_sum2);


				//������
				//W3
				Wmain[k]->initialize();
				//�E�B���h�E����f��
				int pixel_sum_window = 0;
				//�E�B���h�E����f���̋t��
				float pixel_sum_window_inv = 0.0f;

				//��f�ւ̃|�C���^������
				GTYPE* G_center = G_center_rowStart;
				int* result_center = result_center_rowStart;
				CTYPE* cx_center = cx_center_rowStart;
				float* dx_center = dx_center_rowStart;
				int* W0_rem_f = W0_rem_f_rowStart;
				GTYPE* W0_rem_g = W0_rem_g_rowStart;
				int* W0_add_f = W0_add_f_rowStart;
				GTYPE* W0_add_g = W0_add_g_rowStart;


				//W2�������ʒu���Z�b�g
				W2[k]->resetPos();
				//���̊K�w�̏����ʒu�Z�b�g
				setPosAtDim0(x[0], r_dim[0], dim0Start);
				const int remStartPos = x[0].add;
				for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
				{
					//�X�e�[�^�X�Z�b�g
					setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);

					//window����f���̍X�V
					calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

					//W3�̍X�V
					//W3 = W3 + W2[x[0].add] - W2[x[0].rem](W3��window)
					if (status[0].hasAdd)
					{
						//W2�̍X�V
						//W2[x[0].add] = W2[x[0].add] + W1[x[1].add, x[0].add] - W1[x[1].rem, x[0].add]
						if (status[1].hasAdd)
						{
							//W1[x[0].add] �̍X�V
							//W1[x[1].add, x[0].add] = W1[x[1].add, x[0].add] + W0[x[2].add, x[1].add, x[0].add] - W0[x[2].rem, x[1].add, x[0].add]
							if (status2.hasAdd) //��f�ǉ�	(W1[x[0].add]) + W0[x[1].add, x[0].add]
								addPixelToWindow(*W1[k], W0_add_f, W0_add_g);
							if (status2.hasRem)//��f�폜	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
								removePixelFromWindow(*W1[k], W0_rem_f, W0_rem_g);
							//W1�ǉ�	(W2) + W1[x[0].add]
							updateSubWindowAndAddToWindow(Imax, *W2[k], *W1[k]);
						}
						if (status[1].hasRem)//W1�폜	(W2) - W1[x[0].rem]
							removeSubWindowFromWindow(Imax, *W2[k], *W1[k]);
						//W2�ǉ�	(W3) + W1[x[0].add]
						updateSubWindowAndAddToWindow(Imax, *Wmain[k], *W2[k]);
					}
					if (status[0].hasRem)//W2�폜	(W3) - W2[x[0].rem]
						removeSubWindowFromWindow(Imax, *Wmain[k], *W2[k]);

					//�����l�̌v�Z
					if (status[0].isInside_image)
					{
						/*
						CTYPE cx;
						float dx;
						calculateCxDx(Wmain[k]->gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
						*/
						findMedian(*cx_center, *dx_center, half, *Wmain[k], *result_center);
						G_center++;
						result_center++;
						cx_center++;
						dx_center++;
					}
				}
				G_center_rowStart += size_dim[0];
				result_center_rowStart += size_dim[0];
				cx_center_rowStart += size_dim[0];
				dx_center_rowStart += size_dim[0];
				W0_add_f_rowStart += size_dim[0];
				W0_add_g_rowStart += size_dim[0];
				W0_rem_f_rowStart += size_dim[0];
				W0_rem_g_rowStart += size_dim[0];
			}
			//���g�̎���+1�̃��[�v���I������Ƃ��ɃE�B���h�E���e���Z�b�g
			W2[k]->setZero();

		}


		th1.join();
		//�v�Z����cx,dx���_�E�����[�h
		if (status2GPU.isInside_image)
		{
			Utility::uploadMatToDevice(G[x2.center + 1], GDevice, sizeInfo, NULL);
			cu_calculateCxDx(sizeInfo, NULL, GDevice, radius_space, eps2, cxDevice, dxDevice, sumGwindow, temp);
			Utility::downloadLinearArrayAsMat(cxDevice, sizeInfo, cx);
			Utility::downloadLinearArrayAsMat(dxDevice, sizeInfo, dx);
		}


	}

	return result;
}

//3D�@����x�`�����l��
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
std::vector<cv::Mat> FGMF2::filter3DIx(std::vector<cv::Mat>& Is, std::vector<cv::Mat>& G, int threadNum, int radius_space, int radius_depth, float eps2, int Imax)
{
	//check validation
	assert(Is[0].depth() == CV_32S);
	assert(G[0].depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//���͎���
	const int DIM = 3;
	//���͉摜�`�����l����
	const int Ichannels = Is[0].channels();
	//�����l
	const float half = 0.5f;
	//������
	//�T�C�Y
	const std::vector<int> size_dim{ Is[0].cols , Is[0].rows, (int)Is.size() };
	//���a
	const std::vector<int> r_dim{ radius_space, radius_space, radius_depth };

	//�}���`�X���b�h�p
	std::vector<int> dim0Start_vec(threadNum);
	std::vector<int> dim0End_vec(threadNum);
	std::vector<int> memoryLength_vec(threadNum);
	std::vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//���ʕۑ�
	std::vector<cv::Mat> result(Is.size());
	for (int i = 0; i < result.size(); i++)
	{
		result[i] = cv::Mat(size_dim[1], size_dim[0], CV_32SC(Ichannels));
	}

	//W1
	std::vector < std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>>>> W1(Ichannels, std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>>>(threadNum));
	//W2
	std::vector < std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>>>> W2(Ichannels, std::vector< std::unique_ptr< Window_vector<GSum, FGSumUpToIndex, FG>>>(threadNum));
	//W3(Wmain)
	std::vector < std::vector< std::unique_ptr< Window_single<GSum, FGSumUpToIndex, FG>>>> Wmain(Ichannels, std::vector< std::unique_ptr< Window_single<GSum, FGSumUpToIndex, FG>>>(threadNum));

	for (int c = 0; c < Ichannels; c++)
	{
#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			std::vector<int> size_dim_memory;
			copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
			size_dim_memory[0] = memoryLength_vec[k];
			W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
			W2[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 2, true)));
			Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
		}
	}

	//���̊K�w�̏����ʒu�Z�b�g
	Pos x2;
	DimStatus status2;
	int pixelSum2;
	setPos(x2, r_dim[2]);
	for (; x2.center < size_dim[2]; x2.center++, x2.add++, x2.rem++)
	{
		//�X�e�[�^�X�Z�b�g
		setStatusAtOutermostLoop(x2, size_dim[2], status2);
		//1�񓖂���̉�f��
		int pixel_sum2 = calculatePixelNumAtOutermostLoop(x2, size_dim[2], r_dim[2], status2);

		Pos vecPos;
		vecPos.center = (std::max)(0, x2.center);
		vecPos.add = (std::min)(size_dim[2] - 1, x2.add);
		vecPos.rem = (std::max)(0, x2.rem);

		//�ŏ��̃`�����l��
#pragma omp parallel for
		for (int k = 0; k < threadNum; k++)
		{
			//��f��
			std::vector<int> pixel_sum(2);
			//�ʒu
			std::vector<Pos> x(2);
			//�X�e�[�^�X
			std::vector<DimStatus> status(2);


			const int dim0Start = dim0Start_vec[k];
			const int dim0End = dim0End_vec[k];
			const int insideImageStart = insideImageStart_vec[k];


			//�Ή������f�ւ̃|�C���^�i�����ɂ���Đݒ肪�قȂ邱�Ƃɒ��Ӂj
			//3�����ȍ~�ɂ��ẮA��f�̍��W�͓��������Avector�̂ǂ�cv::Mat�����o���̂����ς��
			GTYPE* G_center_rowStart = G[vecPos.center].ptr<GTYPE>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* result_center_rowStart = result[vecPos.center].ptr<int>(0) + insideImageStart - r_dim[1] * size_dim[0];
			int* W0_rem_f_rowStart = I[vecPos.rem].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_rem_g_rowStart = G[vecPos.rem].ptr<GTYPE>(0) + dim0Start + r_dim[0];
			int* W0_add_f_rowStart = I[vecPos.add].ptr<int>(0) + dim0Start + r_dim[0];
			GTYPE* W0_add_g_rowStart = G[vecPos.add].ptr<GTYPE>(0) + dim0Start + r_dim[0];

			//W1�������ʒu���Z�b�g
			W1[k]->resetPos();
			//���̊K�w�̏����ʒu�Z�b�g
			setPos(x[1], r_dim[1]);
			for (; x[1].center < size_dim[1]; x[1].center++, x[1].add++, x[1].rem++)
			{
				//�X�e�[�^�X�Z�b�g
				setStatus(x[1], size_dim[1], status2.isInside_image, status[1]);
				//1�񓖂���̉�f��
				pixel_sum[1] = calculatePixelNumAtDim(x[1], size_dim[1], r_dim[1], status[1], pixel_sum2);

				//������
				//W3
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

				//W2�������ʒu���Z�b�g
				W2[k]->resetPos();
				//���̊K�w�̏����ʒu�Z�b�g
				setPosAtDim0(x[0], r_dim[0], dim0Start);
				const int remStartPos = x[0].add;
				for (; x[0].center <= dim0End; x[0].center++, x[0].add++, x[0].rem++)
				{
					//�X�e�[�^�X�Z�b�g
					setStatusAtDim0(x[0], size_dim[0], remStartPos, insideImageStart, status[1].isInside_image, status[0]);
					//window����f���̍X�V
					calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

					//W3�̍X�V
					if (status[0].hasAdd) {
						//W2�̍X�V
						if (status[1].hasAdd) {
							//W1[x[0].add] �̍X�V
							if (status2.hasAdd) //��f�ǉ�
								addPixelToWindow_gSum(*W1[k], W0_add_f, W0_add_g);
							if (status2.hasRem)//��f�폜
								removePixelFromWindow_gSum(*W1[k], W0_rem_f, W0_rem_g);
							updateSubWindowAndAddToWindow_gSum(Imax, *W2[k], *W1[k]);//W1�ǉ�
						}
						if (status[1].hasRem)//W1�폜
							removeSubWindowFromWindow_gSum(Imax, *W2[k], *W1[k]);
						updateSubWindowAndAddToWindow_gSum(Imax, *Wmain[k], *W2[k]);//W2�ǉ�
					}
					if (status[0].hasRem)//W2�폜
						removeSubWindowFromWindow_gSum(Imax, *Wmain[k], *W2[k]);

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
			//���g�̎���+1�̃��[�v���I������Ƃ��ɃE�B���h�E���e���Z�b�g
			W2[k]->setZero();

		}

		//2�`�����l���ڈȍ~
		for (int c = 1; c < Ichannels; c++)
		{
		}
	}

	return result;
}






//2D�@����1�`�����l�� ���������p �}���`�X���b�h
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF2::filter2DWindow_Save(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	/*
	�������̏ꍇ�́A���[�v�̏���
	�EN�����`3����
	�EI�̃`�����l��
	�E�摜����
	�ƂȂ�̂ŁA�e�`�����l����ʂ̊֐����ŕ���������`�ł͂Ȃ��A�֐����ł��ׂčs��
	�����G�̎g���܂킵�Ō������グ�邽�߁B

	�܂��A���̂��߂Ƀ}���`�X���b�h�̈������قȂ�B
	�}���`�X���b�h�p���������̊֐����ōs���B
	�������ɂ��Ă̓X���b�h���Ƃɕ�������B
	�摜�̓u���b�N���ƂɃA�N�Z�X�J�n�ʒu��ς���B

	�Ⴆ�Ή摜�̃T�C�Y��h*w�ł���Ah*w�̃T�C�Y���������m�ۂ��K�v�ȏꍇ�i�RD�ȏ�̏ꍇ)�A
	dim0�����̔��a��r�Ƃ����Ƃ�
	�e�X���b�h�p�u���b�N����a[i]�̎��A�e�u���b�N���K�v��dim0�����̃�����������
	a[0]+r, a[1]+2r, ... a[j-1]+2r, a[j]+r

	//��f�A�N�Z�X
	//�ǉ���[x[n-1].add, x[n-2].add, ..., x[0].add]�ł���
	//�폜��[x[n-1].rem, x[n-2].add, ..., x[0].add]�ł���
	�v�f�ւ̃A�N�Z�X�́A��f���E�B���h�E���ňقȂ�B
	��f�̏ꍇ�A�������ɂȂ��3�����ȍ~�͗v�f��cv::Mat�̑�����vector�ɂȂ�B
	3�����ȍ~�̎�����1�i�ނ��тɃ������ʒu������̒l�ɃZ�b�g���āA2���������[�v�ł̓C���N�������g�őΉ�����B
	2D�̏ꍇ�AI.at[x[1].add][x[0].add], I.at[x[1].rem][x[0].add],
	3D�̏ꍇ�AI[x[2].add].at[x[1].add][x[0].add], I[x[2].rem].at[x[1].add][x[0].add]
	2���������[�v�̊J�n�|�C���^�ʒu��
	2D�̏ꍇ�AI.at[0]

	�C���N�������g�őΉ�����ꍇ�Afor���[�v1��̂��тɕK��1�C���N�������g�����A��O�𐶂������Ȃ����ƂŃv���O�������₷������B���̂��߂ɂ́A�C���N�������g�J�n�ʒu��K�؂ɐݒ�A���̍s�ɍs���Ƃ��ɓK�؂ɃW�����v������K�v������B
	for���[�v2���������ɂ��ẮA�����J�n�ʒu��
	x[1].add��0����n�܂�Ax[1].center��size_dim[1]-1�ɂȂ�܂ŌJ��Ԃ����B
	1���������ɂ��ẮA�����J�n�ʒu��
	dim0Start����dim0End�܂ŌJ��Ԃ����B
	�J�n�ʒu�́A�E�B���h�E���\�z���邽�߂̉�f�ǉ����ł���ŏ��̏ꏊ�ɂȂ�B���a��r�ł���Ƃ��A2�����ȏ�ɂ��ẮA�J�n�ʒu�i������f�j��-r�̈ʒu����n�܂�B
	1�����ɂ��ẮA�}���`�X���b�h�p�ɕ�������Ă��邽�߁A�����قȂ�B
	��ԏ��߂̃u���b�N�́A2�����ȏ�Ɠ��l�A-r����n�܂�B
	��Ԗڈȍ~�̃u���b�N�́A�ʏ�̊J�n�ʒu���-r�O����\�z���K�v�ł��邽�߁A�����Ώۉ�f�ʒu�ɑ΂���-2r�̈ʒu����n�߂�K�v������B

	2D�̏ꍇ�AI.ptr<int>(0)����Ƃ��āA
	x[0].add = 0 + dim0St



	�E�B���h�E�̏ꍇ�͑�������1�����ɂ��Ă���A����J�n�ʒu����K�����ɎQ�Ƃ����B
	�Ȃ̂ŁAadd,rem�̃|�C���^��0�ɃZ�b�g���Ă����A�ǉ��܂��͍폜�̂��тɃC���N�������g����悤�ɂ��Ă����΂悢�B
	����������̃^�C�~���O��0�ɃZ�b�g���Ȃ����K�v������A����̓E�B���h�E�̎����{�P�̗v�f���P�C���N�������g���ꂽ�Ƃ��B


	��f�ɂ��Ă�1�s���ɍs���Ƃ��ɃW�����v����B
	����ɂ��ẮA�������̍s�̐擪�̃|�C���^���L�^����悤�ɂ��A���̍s�J�n���Ƀ|�C���^��1�s���i�߂āi= + size_dim[0]�j����������΂悳�����B
	1�s�������ɂ́Ainside_image��True�ł��钆���l�v�Z�I�����Ƀ|�C���^���C���N�������g����΂悢�B
	center,add, rem�̏����|�C���^�ʒu�͑S�ċ��ʂ�inside_image�̊J�n�ʒu�ɂ���΂悢�B

	*/

	const int Ichannels = I.channels();

	//check validation
	assert(I.depth() == CV_32S);
	assert(G.depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif


	//���͎���
	const int DIM = 2;

	//�����l
	const float half = 0.5f;

	//������
	//�T�C�Y
	const vector<int> size_dim{ I.cols , I.rows };
	//���a
	const vector<int> r_dim{ radius, radius };


	//�}���`�X���b�h�p
	vector<int> dim0Start_vec(threadNum);
	vector<int> dim0End_vec(threadNum);
	vector<int> memoryLength_vec(threadNum);
	vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);



	//�E�B���h�E�A�N�Z�X�p
	//size_dim = {dim0, dim1, dim3, dim4}�̂Ƃ�
	// = {dim0, dim0*dim1, dim0*dim1*dim3}
	//�Ƃ��邱�ƂŁA�Ⴆ�� [x1,x2,x3,x4]�ɃA�N�Z�X����Ƃ��A�E�B���h�E��������ł͂��ꂪ1�����ŕ���ł���̂�
	// x4*dim0*dim1*dim3 + x3*dim0*dim1 + x2*dim0 + x1
	//const vector<int> size_prod{ size_dim[0] };


	//�������m�ہE������
	cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

	//W0:��f�Ȃ̂ŋL�^�̕K�v�Ȃ��iI,G�j
	//�}���`�X���b�h�p�Ƀx�N�g���Ŋm��
	//W1
	vector <std::unique_ptr < Window_vector<GSum, FGSumUpToIndex, FG>> > W1(threadNum);

	//W2(Wmain)
	vector<std::unique_ptr <Window_single<GSum, FGSumUpToIndex, FG>>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = (std::unique_ptr <Window_vector<GSum, FGSumUpToIndex, FG>>(new  Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true)));
		Wmain[k] = (std::unique_ptr < Window_single<GSum, FGSumUpToIndex, FG>>(new Window_single<GSum, FGSumUpToIndex, FG>(Imax)));
	}


#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		//��f��
		vector<int> pixel_sum(DIM);
		//�ʒu
		vector<Pos> x(DIM);
		//�X�e�[�^�X
		vector<DimStatus> status(DIM);

		const int dim0Start = dim0Start_vec[k];
		const int dim0End = dim0End_vec[k];
		const int insideImageStart = insideImageStart_vec[k];


		/*
		//��f�ւ̃|�C���^������
		//�Ή������f�ւ̃|�C���^�i�����ɂ���Đݒ肪�قȂ邱�Ƃɒ��Ӂj
		//�������̉�f
		GTYPE* G_center = G.ptr<GTYPE>(0) + insideImageStart_vec[k];
		int* result_center = result.ptr<int>(0) + insideImageStart_vec[k];
		//n������Ԃɂ����Ēǉ��A�폜�̑ΏۂƂȂ��f�̈ʒu�́A���ڎ��_���W�ɑ΂���
		//�ǉ���[x[n-1].add, x[n-2].add, ..., x[0].add]�ł���
		//�폜��[x[n-1].rem, x[n-2].add, ..., x[0].add]�ł���
		//2�����̎���[x[1].add, x[0].add]�A[x[1].rem, x[0].add]�̊֌W�����An�����̏ꍇ�͓��͂�cv::Mat�x�N�g���Ȃ̂ŁA�Ⴆ��
		//I[x[n-1].add][x[n-2].add]...] ��cv::Mat�ɂ�����[x[1].add, x[0].add]��
		//I[x[n-1].rem][x[n-2].add]...] ��cv::Mat�ɂ�����[x[1].add, x[0].add]�ɂȂ�B
		int* W0_rem_f = I.ptr<int>(0) + insideImageStart_vec[k];
		GTYPE* W0_rem_g = G.ptr<GTYPE>(0) + insideImageStart_vec[k];
		int* W0_add_f = I.ptr<int>(0) + insideImageStart_vec[k];
		GTYPE* W0_add_g = G.ptr<GTYPE>(0) + insideImageStart_vec[k];
		*/

		//�Ή������f�ւ̃|�C���^�i�����ɂ���Đݒ肪�قȂ邱�Ƃɒ��Ӂj
		//center�̍s�J�n�ʒu��insideImageStart
		//addrem�̍s�J�n�ʒu��dim0Start + r_dim[0]
		//�s�����I������if���ɂ��
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
			//W3
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



			/*
			�e�K�w�̃E�B���h�E�ɑ΂���|�C���^�����ƍl����΁A��f�����łȂ�win�n�������悤�ɏ�����̂ł�

			dim0�����ɃX���C�h (W0�͉�f)
			W1[x[1].add, x[0].add] = W1[x[1].add, x[0].add] + W0[x[2].add, x[1].add, x[0].add] - W0[x[2].rem, x[1].add, x[0].add]
			W2[x[0].add] = W2[x[0].add] + W1[x[1].add, x[0].add] - W1[x[1].rem, x[0].add]
			W3 = W3 + W2[x[0].add] - W2[x[0].rem] (W3��window)
			*/


			//W1�������ʒu���Z�b�g
			W1[k]->resetPos();

			//W2

			//���̊K�w�̏����ʒu�Z�b�g
			//dim0
			//setPos(x[0], r_dim[0], dim0Start);
			x[0].center = dim0Start;
			x[0].add = dim0Start + r_dim[0];
			x[0].rem = dim0Start - r_dim[0] - 1;
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
					/*
					* addPixelToWindow_gSum ����Ƃ��ɁA�E�B���h�E����2D�Ȃ烁�����P���������An�����̎���n-1�����Ȃ̂ŁA�ǂ̂悤�ɃA�N�Z�X����̂��B
					* �Ǝv�����������1������ɕ��ׂĂ���̂�����
					*/
					//W1[x[0].add] �̍X�V
					if (status[1].hasAdd) //��f�ǉ�	(W1[x[0].add]) + W0[x[1].add, x[0].add]
						addPixelToWindow_gSum(*W1[k], W0_add_f, W0_add_g);
					if (status[1].hasRem)//��f�폜	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
						removePixelFromWindow_gSum(*W1[k], W0_rem_f, W0_rem_g);
					//W1�ǉ�	(W2) + W1[x[0].add]
					updateSubWindowAndAddToWindow_gSum(Imax, *Wmain[k], *W1[k]);
				}
				if (status[0].hasRem)//W1�폜	(W2) - W1[x[0].rem]
					removeSubWindowFromWindow_gSum(Imax, *Wmain[k], *W1[k]);

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


//2D�@����1�`�����l�� ���������p �}���`�X���b�h
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
cv::Mat FGMF2::filter2DWindow2(cv::Mat& I, cv::Mat& G, int threadNum, int radius, float eps2, int Imax)
{
	const int Ichannels = I.channels();

	//check validation
	assert(I.depth() == CV_32S);
	assert(G.depth() == CV_32S);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif


	//���͎���
	const int DIM = 2;
	//�����l
	const float half = 0.5f;
	//������
	//�T�C�Y
	const vector<int> size_dim{ I.cols , I.rows };
	//���a
	const vector<int> r_dim{ radius, radius };

	//�}���`�X���b�h�p
	vector<int> dim0Start_vec(threadNum);
	vector<int> dim0End_vec(threadNum);
	vector<int> memoryLength_vec(threadNum);
	vector<int> insideImageStart_vec(threadNum);
	calculatePosForMultithread2D(size_dim[0], threadNum, r_dim[0], dim0Start_vec, dim0End_vec, memoryLength_vec, insideImageStart_vec);

	//�������m�ہE������
	cv::Mat result = cv::Mat(I.size(), CV_32SC(Ichannels));

	//W1
	vector <Window_vector<GSum, FGSumUpToIndex, FG>> W1(threadNum);
	//W2(Wmain)
	vector<Window_single<GSum, FGSumUpToIndex, FG>> Wmain(threadNum);

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		vector<int> size_dim_memory;
		copy(size_dim.begin(), size_dim.end(), back_inserter(size_dim_memory));
		size_dim_memory[0] = memoryLength_vec[k];
		W1[k] = Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim_memory, 1, true);
		Wmain[k] = Window_single<GSum, FGSumUpToIndex, FG>(Imax);
	}

#pragma omp parallel for
	for (int k = 0; k < threadNum; k++)
	{
		//��f��
		vector<int> pixel_sum(DIM);
		//�ʒu
		vector<Pos> x(DIM);
		//�X�e�[�^�X
		vector<DimStatus> status(DIM);

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
			Wmain[k].initialize();
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
			W1[k].resetPos();

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
						addPixelToWindow_gSum(W1[k], W0_add_f, W0_add_g);
					if (status[1].hasRem)//��f�폜	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
						removePixelFromWindow_gSum(W1[k], W0_rem_f, W0_rem_g);
					//W1�ǉ�	(W2) + W1[x[0].add]
					updateSubWindowAndAddToWindow_gSum(Imax, Wmain[k], W1[k]);
				}
				if (status[0].hasRem)//W1�폜	(W2) - W1[x[0].rem]
					removeSubWindowFromWindow_gSum(Imax, Wmain[k], W1[k]);

				//�����l�̌v�Z
				if (status[0].isInside_image)
				{
					CTYPE cx;
					float dx;
					calculateCxDx(Wmain[k].gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
					findMedian(cx, dx, half, Wmain[k], *result_center);
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


#endif