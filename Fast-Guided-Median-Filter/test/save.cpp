//#include "FGMF.h"
#if 0

//�x�[�X�v���O����
Mat FGMF::filter(Mat& I, Mat& G, int r, float eps2, int Imax)
{
	//check validation
	//�ǂ����int�^�A�`�����l����1
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && G.channels() == 1);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#endif

	//�����l
	const float half = 0.5f;

	//������
	int width_ = I.cols;
	int height_ = I.rows;

	//���ʕۑ�
	Mat med_idx = Mat(height_, width_, CV_32S);


	//�e��p�q�X�g�O�����i�[�ϐ�
	fg** histo_col = new fg *[width_];
	for (int i = 0; i < width_; i++)
	{
		histo_col[i] = (fg*)_aligned_malloc(sizeof(fg) * Imax, 32);
		memset(histo_col[i], 0, sizeof(fg) * Imax);
	}


	//mean(g), mean(gg)�����v�Z�p�ϐ��i�a�ŕۑ��j
	//under_i_index�Ŏ������l�ȉ��̗�q�X�g�O�����a
	struct colDataStruct {
		int g_sum;
		int gg_sum;
		int under_i_f;
		int under_i_g;

	};
	colDataStruct* colData = new colDataStruct[width_];
	memset(colData, 0, sizeof(colDataStruct) * width_);

	//under_i_index�Ŏ������l�ȉ� �̂Ƃ���index
	int* under_i_col_index = new int[width_];
	memset(under_i_col_index, 0, sizeof(int) * width_);

	//Window���q�X�g�O����
	fg* histo_window = (fg*)_aligned_malloc(sizeof(fg) * Imax, 32);


	//
	int y_add = r - 1;//�폜������
	int y_rem = -r - 2;//�폜�����s


	//�����J�n
	for (int y = 0; y < height_; y++)
	{
		//cout << "y:" << y << endl;

		y_add++;
		y_rem++;

		bool hasPreviousRow = y_rem >= 0;
		bool hasNestRow = y_add < height_;

		//������

		//Window�֌W
		//�a
		int g_sum_window = 0;
		int gg_sum_window = 0;
		int pixel_sum_window = 0;
		//�E�B���h�E�q�X�g�O����������
		memset(histo_window, 0, sizeof(fg) * Imax);

		//index�ȉ�
		int f_sum_under_i_window = 0;
		int g_sum_under_i_window = 0;



		//1�񓖂���̉�f��
		int pixel_sum_col = 2 * r + 1;
		if (y < r)
			pixel_sum_col = y + r + 1;
		else if ((y + r) >= height_)
			pixel_sum_col = height_ - y + r;


		//��������index y = 0�Ȃ���͂ŏ������Ay > 0�Ȃ�[y - 1, 0]�̒l�ŏ�����
		int index;
		if (y > 0)
			index = *med_idx.ptr<int>(y - 1);
		else
			index = *I.ptr<int>(0);

		int x_add = -1;// �ǉ�������
		int x_rem = -2 * r - 2;// �폜������
		//�Ή������f�ւ̃|�C���^
		//��̍폜��f
		int* I_rem;
		int* G_rem;
		if (hasPreviousRow) {
			I_rem = I.ptr<int>(y_rem);
			G_rem = G.ptr<int>(y_rem);
		}
		//��̒ǉ���f
		int* I_add;
		int* G_add;
		if (hasNestRow) {
			I_add = I.ptr<int>(y_add);
			G_add = G.ptr<int>(y_add);
		}
		//�������̉�f
		int* G_center = G.ptr<int>(y);
		int* med_center = med_idx.ptr<int>(y);

		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 0.0f;

		for (int x = -r; x < width_; x++)
		{
			x_add++;
			x_rem++;


			//�ǉ����鎟�̗񂪂��邩
			bool hasNextColumn = x_add < width_;
			//�폜����O�̗񂪂��邩
			bool hasPreviousColumn = x_rem >= 0;

			//window����f���̍X�V
			if (hasNextColumn && !hasPreviousColumn) {
				//�ǉ��̂�
				pixel_sum_window += pixel_sum_col;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}
			else if (!hasNextColumn && hasPreviousColumn)
			{
				//�폜�̂�
				pixel_sum_window -= pixel_sum_col;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}

			//��q�X�g�O�����̍X�V

			//���̗񂪂���Ȃ�X�V
			if (hasNextColumn) {
				if (y > 0) {

					//�O�̍s������Ȃ�q�X�g�O��������폜
					if (hasPreviousRow)
					{
						//�O�̍s��g��f�擾
						//�q�X�g�O�������猸�Z
						int fidx = I_rem[x_add];//�s�͍폜�s�A��͒ǉ��s�Ȃ̂ł����Ȃ�
						int gval = G_rem[x_add];
						histo_col[x_add][fidx].f -= 1;
						histo_col[x_add][fidx].g -= gval;

						//SIMD���\(���ʂ��邩������Ȃ�����)
						//g_sum_col, gg_sum_col���猸�Z
						colData[x_add].g_sum -= gval;
						colData[x_add].gg_sum -= gval * gval;
						if (fidx <= under_i_col_index[x_add]) {
							colData[x_add].under_i_f -= 1;
							colData[x_add].under_i_g -= gval;
						}

						/*
						__m128i a128 = _mm_load_si128((__m128i *)&colData[x_add]);
						int m1 = fidx <= under_i_col_index[x_add];
						__m128i m128 = _mm_set_epi32(gval * m1, m1, gval * gval, gval);
						__m128i c128 = _mm_sub_epi32(a128, m128);
						_mm_store_si128((__m128i *)&colData[x_add], c128);
						*/
					}
					//���̍s������Ȃ�X�V
					if (hasNestRow) {
						//���̍s���q�X�g�O�����ɒǉ�
						//���̍s��g��f�擾
						//�q�X�g�O�����ɉ��Z
						int fidx = I_add[x_add];
						int gval = G_add[x_add];
						histo_col[x_add][fidx].f += 1;
						histo_col[x_add][fidx].g += gval;

						//SIMD���\(���ʂ��邩������Ȃ�����)
						//g_sum_col, gg_sum_col�ɉ��Z
						colData[x_add].g_sum += gval;
						colData[x_add].gg_sum += gval * gval;
						//�ǉ��v�f��f������index�ȉ����ǂ����𔻒肵�A�ȉ��Ȃ�ǉ����邽�߂ɑ����A���傫���Ȃ�sum�Ɋ܂܂�Ȃ��̂ŉ������Ȃ�
						if (fidx <= under_i_col_index[x_add]) {
							colData[x_add].under_i_f += 1;
							colData[x_add].under_i_g += gval;
						}

						/*
						__m128i a128 = _mm_load_si128((__m128i *)&colData[x_add]);
						int m1 = fidx <= under_i_col_index[x_add];
						__m128i m128 = _mm_set_epi32(gval * m1, m1, gval * gval, gval);
						__m128i c128 = _mm_add_epi32(a128, m128);
						_mm_store_si128((__m128i *)&colData[x_add], c128);
						*/
					}
				}
				else
				{
					//1�s��(y=0)�p�����@�񂷂ׂĒǉ�
					for (int yy = 0; yy <= r; yy++)
					{
						//g��f�擾
						//�q�X�g�O�����ɉ��Z
						int fidx = I.ptr<int>(yy)[x_add];
						int gval = G.ptr<int>(yy)[x_add];
						histo_col[x_add][fidx].f += 1;
						histo_col[x_add][fidx].g += gval;
						//g_sum_col, gg_sum_col�ɉ��Z
						colData[x_add].g_sum += gval;
						colData[x_add].gg_sum += gval * gval;
					}

				}
				//window�X�V
				//sum_window�ɒǉ�������Z
				g_sum_window += colData[x_add].g_sum;
				gg_sum_window += colData[x_add].gg_sum;
				//�q�X�g�O�����ɒǉ�
				addHistogram(Imax, histo_window, histo_col[x_add]);

				int u_idx = under_i_col_index[x_add];
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//����SIMD���\
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//�ǉ�
					colData[x_add].under_i_f += histo_col[x_add][i].f * sign1;
					colData[x_add].under_i_g += histo_col[x_add][i].g * sign1;
				}

				//under index�̍X�V
				under_i_col_index[x_add] = index;

				//�E�B���h�E��f_sum, g_sum�̍X�V
				f_sum_under_i_window += colData[x_add].under_i_f;
				g_sum_under_i_window += colData[x_add].under_i_g;

			}

			//�폜�񂪂���Ȃ�
			if (hasPreviousColumn)
			{
				//window�X�V
				//�q�X�g�O��������폜
				remHistogram(Imax, histo_window, histo_col[x_rem]);


				//sum_window����폜������Z
				g_sum_window -= colData[x_rem].g_sum;
				gg_sum_window -= colData[x_rem].gg_sum;
				//�E�B���h�E��f_sum, g_sum�̍X�V
				f_sum_under_i_window -= colData[x_rem].under_i_f;
				g_sum_under_i_window -= colData[x_rem].under_i_g;


				int u_idx = under_i_col_index[x_rem];
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//����SIMD���\
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//�ǉ�
					f_sum_under_i_window -= histo_col[x_rem][i].f * sign1;
					g_sum_under_i_window -= histo_col[x_rem][i].g * sign1;
				}
			}

			//�����l�̌v�Z
			if (x >= 0)
			{
				float g_ave = g_sum_window * pixel_sum_window_inv;
				float gg_ave = gg_sum_window * pixel_sum_window_inv;
				float vx = gg_ave - g_ave * g_ave + eps2;
				float tmp = G_center[x] - g_ave;
				float cx = tmp * pixel_sum_window_inv / vx;
				float dx = pixel_sum_window_inv - g_ave * cx;
				float h = cx * g_sum_under_i_window + dx * f_sum_under_i_window;

				int flag1 = h < half;
				int flag2 = flag1 - 1;
				int sign1 = flag1 * 2 - 1;
				while (true)
				{
					index += flag1;
					int fnum = histo_window[index].f;
					if (fnum != 0)
					{
						//����Ȃ�X�V����h�`�F�b�N
						f_sum_under_i_window += fnum * sign1;
						g_sum_under_i_window += histo_window[index].g * sign1;
						h = cx * g_sum_under_i_window + dx * f_sum_under_i_window;
						if ((h - half) * sign1 >= 0)
						{
							//�������̂ł���index��median
							med_center[x] = index;
							index += flag2;
							break;
						}
					}
					index += flag2;
				}
			}

		}

	}



	//�������J��
	for (int i = 0; i < width_; i++)
	{
		_aligned_free(histo_col[i]);
	}
	delete[] histo_col;
	_aligned_free(histo_window);
	delete[] colData;


	return med_idx;
}




//���\��������Ȃ��悤�ɂ��Ȃ��烊�t�@�N�^�����O
//�M���M���܂ł�����ꍇ�B����ȏ���Ɛ��\�ቺ�������A�͂��Ȃ̂ł��
Mat FGMF::filter4(Mat& I, Mat& G, int radius, float eps2, int Imax)
{
	//check validation
	//�ǂ����int�^�A�`�����l����1
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && G.channels() == 1);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#elif defined(USE_AVX512)
	assert(Imax % 8 == 0);
#endif

	//�����l
	const float half = 0.5f;

	//������
	int size_dim0 = I.cols;
	int size_dim1 = I.rows;
	//
	int r_dim0 = radius;
	int r_dim1 = radius;

	//���ʕۑ�
	Mat result = Mat(size_dim1, size_dim0, CV_32S);


	//�e��p�q�X�g�O�����i�[�ϐ�
	fg** histo_win1 = new fg *[size_dim0];
	for (int i = 0; i < size_dim0; i++)
	{
		histo_win1[i] = (fg*)_aligned_malloc(sizeof(fg) * Imax, MEMORY_ALIGNMENT);
		memset(histo_win1[i], 0, sizeof(fg) * Imax);
	}


	//mean(g), mean(gg)�����v�Z�p�ϐ��i�a�ŕۑ��j
	//under_i_index�Ŏ������l�ȉ��̗�q�X�g�O�����a
	struct windowDataStruct {
		int g_sum;
		int gg_sum;
		int f_sum_upto_index;
		int g_sum_upto_index;
		int index;

	};
	windowDataStruct* sum_win1 = new windowDataStruct[size_dim0];
	memset(sum_win1, 0, sizeof(windowDataStruct) * size_dim0);

	//�T�u�E�B���h�E��sumUpToIndex�L�^���̃C���f�b�N�X
	//int* index_win1 = new int[size_dim0];
	//memset(index_win1, 0, sizeof(int) * size_dim0);

	//Window���q�X�g�O����
	fg* histo_window = (fg*)_aligned_malloc(sizeof(fg) * Imax, MEMORY_ALIGNMENT);


	//
	int x_add_dim1 = radius - 1;//�폜������
	int x_rem_dim1 = -radius - 2;//�폜�����s
	int x_dim1 = 0;

	//�����J�n
	for (; x_dim1 < size_dim1; x_dim1++)
	{
		x_add_dim1++;
		x_rem_dim1++;

		bool hasRem_dim1 = x_rem_dim1 >= 0;
		bool hasAdd_dim1 = x_add_dim1 < size_dim1;

		//������

		//Window�֌W
		//�a
		gSumData gSum_window = { 0, 0 };
		//index�ȉ�
		int sumUpToIndex_window_f = 0;
		int sumUpToIndex_window_g = 0;

		//�E�B���h�E�q�X�g�O����������
		memset(histo_window, 0, sizeof(fg) * Imax);
		//�E�B���h�E����f��
		int pixel_sum_window = 0;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 0.0f;
		//1�񓖂���̉�f��
		//int pixel_sum_dim1;
		//���C���E�B���h�E���C���f�b�N�X
		int index;


		//�Ή������f�ւ̃|�C���^
		//��̍폜��f
		int* I_rem = NULL;
		int* G_rem = NULL;
		//��̒ǉ���f
		int* I_add = NULL;
		int* G_add = NULL;
		//�������̉�f
		int* G_center = G.ptr<int>(x_dim1);
		int* med_center = result.ptr<int>(x_dim1);



		if (hasRem_dim1) {
			I_rem = I.ptr<int>(x_rem_dim1);
			G_rem = G.ptr<int>(x_rem_dim1);
		}
		if (hasAdd_dim1) {
			I_add = I.ptr<int>(x_add_dim1);
			G_add = G.ptr<int>(x_add_dim1);
		}


		//1�񓖂���̉�f��
		int pixel_sum_dim1 = 2 * radius + 1;
		if (x_dim1 < radius)
			pixel_sum_dim1 = x_dim1 + radius + 1;
		else if ((x_dim1 + radius) >= size_dim1)
			pixel_sum_dim1 = size_dim1 - x_dim1 + radius;


		if (x_dim1 > 0)
			index = *result.ptr<int>(x_dim1 - 1);
		else
			index = *I.ptr<int>(0);



		int x_add_dim0 = -1;// �ǉ�������
		int x_rem_dim0 = -2 * radius - 2;// �폜������
		int x_dim0 = -radius;

		for (; x_dim0 < size_dim0; x_dim0++)
		{
			x_add_dim0++;
			x_rem_dim0++;


			//�ǉ����鎟�̗񂪂��邩
			bool hasAdd_dim0 = x_add_dim0 < size_dim0;
			//�폜����O�̗񂪂��邩
			bool hasRem_dim0 = x_rem_dim0 >= 0;

			//window����f���̍X�V
			if (hasAdd_dim0 && !hasRem_dim0) {
				//�ǉ��̂�
				pixel_sum_window += pixel_sum_dim1;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}
			else if (!hasAdd_dim0 && hasRem_dim0)
			{
				//�폜�̂�
				pixel_sum_window -= pixel_sum_dim1;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}

			//��q�X�g�O�����̍X�V

			//���̗񂪂���Ȃ�X�V
			if (hasAdd_dim0) {
				if (x_dim1 > 0) {

					//�O�̍s������Ȃ�q�X�g�O��������폜
					if (hasRem_dim1)
					{
						//�O�̍s��g��f�擾
						//�q�X�g�O�������猸�Z
						int fidx = I_rem[x_add_dim0];//�s�͍폜�s�A��͒ǉ��s�Ȃ̂ł����Ȃ�
						int gval = G_rem[x_add_dim0];
						histo_win1[x_add_dim0][fidx].f -= 1;
						histo_win1[x_add_dim0][fidx].g -= gval;

						//SIMD���\(���ʂ��邩������Ȃ�����)
						//g_sum_col, gg_sum_col���猸�Z
						sum_win1[x_add_dim0].g_sum -= gval;
						sum_win1[x_add_dim0].gg_sum -= gval * gval;
						if (fidx <= sum_win1[x_add_dim0].index) {
							sum_win1[x_add_dim0].f_sum_upto_index -= 1;
							sum_win1[x_add_dim0].g_sum_upto_index -= gval;
						}
					}
					//���̍s������Ȃ�X�V
					if (hasAdd_dim1) {
						//���̍s���q�X�g�O�����ɒǉ�
						//���̍s��g��f�擾
						//�q�X�g�O�����ɉ��Z
						int fidx = I_add[x_add_dim0];
						int gval = G_add[x_add_dim0];
						histo_win1[x_add_dim0][fidx].f += 1;
						histo_win1[x_add_dim0][fidx].g += gval;

						//SIMD���\(���ʂ��邩������Ȃ�����)
						//g_sum_col, gg_sum_col�ɉ��Z
						sum_win1[x_add_dim0].g_sum += gval;
						sum_win1[x_add_dim0].gg_sum += gval * gval;
						//�ǉ��v�f��f������index�ȉ����ǂ����𔻒肵�A�ȉ��Ȃ�ǉ����邽�߂ɑ����A���傫���Ȃ�sum�Ɋ܂܂�Ȃ��̂ŉ������Ȃ�
						if (fidx <= sum_win1[x_add_dim0].index) {
							sum_win1[x_add_dim0].f_sum_upto_index += 1;
							sum_win1[x_add_dim0].g_sum_upto_index += gval;
						}
					}
				}
				else
				{
					//1�s��(y=0)�p�����@�񂷂ׂĒǉ�
					for (int yy = 0; yy <= radius; yy++)
					{
						//g��f�擾
						//�q�X�g�O�����ɉ��Z
						int fidx = I.ptr<int>(yy)[x_add_dim0];
						int gval = G.ptr<int>(yy)[x_add_dim0];
						histo_win1[x_add_dim0][fidx].f += 1;
						histo_win1[x_add_dim0][fidx].g += gval;
						//g_sum_col, gg_sum_col�ɉ��Z
						sum_win1[x_add_dim0].g_sum += gval;
						sum_win1[x_add_dim0].gg_sum += gval * gval;
					}

				}
				//window�X�V
				//sum_window�ɒǉ�������Z
				gSum_window.g_sum += sum_win1[x_add_dim0].g_sum;
				gSum_window.gg_sum += sum_win1[x_add_dim0].gg_sum;
				//�q�X�g�O�����ɒǉ�
				addHistogram(Imax, histo_window, histo_win1[x_add_dim0]);

				int u_idx = sum_win1[x_add_dim0].index;
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//����SIMD���\
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//�ǉ�
					sum_win1[x_add_dim0].f_sum_upto_index += histo_win1[x_add_dim0][i].f * sign1;
					sum_win1[x_add_dim0].g_sum_upto_index += histo_win1[x_add_dim0][i].g * sign1;
				}

				//under index�̍X�V
				sum_win1[x_add_dim0].index = index;

				//�E�B���h�E��f_sum, g_sum�̍X�V
				sumUpToIndex_window_f += sum_win1[x_add_dim0].f_sum_upto_index;
				sumUpToIndex_window_g += sum_win1[x_add_dim0].g_sum_upto_index;

			}

			//�폜�񂪂���Ȃ�
			if (hasRem_dim0)
			{
				//window�X�V
				//�q�X�g�O��������폜
				remHistogram(Imax, histo_window, histo_win1[x_rem_dim0]);


				//sum_window����폜������Z
				gSum_window.g_sum -= sum_win1[x_rem_dim0].g_sum;
				gSum_window.gg_sum -= sum_win1[x_rem_dim0].gg_sum;
				//�E�B���h�E��f_sum, g_sum�̍X�V
				sumUpToIndex_window_f -= sum_win1[x_rem_dim0].f_sum_upto_index;
				sumUpToIndex_window_g -= sum_win1[x_rem_dim0].g_sum_upto_index;


				int u_idx = sum_win1[x_rem_dim0].index;
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//����SIMD���\
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//�ǉ�
					sumUpToIndex_window_f -= histo_win1[x_rem_dim0][i].f * sign1;
					sumUpToIndex_window_g -= histo_win1[x_rem_dim0][i].g * sign1;
				}
			}

			//�����l�̌v�Z
			if (x_dim0 >= 0)
			{
				float g_ave = gSum_window.g_sum * pixel_sum_window_inv;
				float gg_ave = gSum_window.gg_sum * pixel_sum_window_inv;
				float vx = gg_ave - g_ave * g_ave + eps2;
				float tmp = G_center[x_dim0] - g_ave;
				float cx = tmp * pixel_sum_window_inv / vx;
				float dx = pixel_sum_window_inv - g_ave * cx;
				float h = cx * sumUpToIndex_window_g + dx * sumUpToIndex_window_f;

				int flag1 = h < half;
				int flag2 = flag1 - 1;
				int sign1 = flag1 * 2 - 1;
				while (true)
				{
					index += flag1;
					int fnum = histo_window[index].f;
					if (fnum != 0)
					{
						//����Ȃ�X�V����h�`�F�b�N
						sumUpToIndex_window_f += fnum * sign1;
						sumUpToIndex_window_g += histo_window[index].g * sign1;
						h = cx * sumUpToIndex_window_g + dx * sumUpToIndex_window_f;
						if ((h - half) * sign1 >= 0)
						{
							//�������̂ł���index��median
							med_center[x_dim0] = index;
							index += flag2;
							break;
						}
					}
					index += flag2;
				}
			}

		}

	}



	//�������J��
	for (int i = 0; i < size_dim0; i++)
	{
		_aligned_free(histo_win1[i]);
	}
	delete[] histo_win1;
	_aligned_free(histo_window);
	delete[] sum_win1;


	return result;
}




//�Â�
Mat FGMF::filterOld(Mat& I, Mat& G, int r, float eps2, int Imax)
{
	//check validation
	//�ǂ����int�^�A�`�����l����1
	assert(I.depth() == CV_32S && I.channels() == 1);
	assert(G.depth() == CV_32S && G.channels() == 1);
#if defined(USE_AVX2)
	assert(Imax % 4 == 0);
#endif

	//�����l
	const float half = 0.5f;

	//������
	int width_ = I.cols;
	int height_ = I.rows;

	//���ʕۑ�
	Mat med_idx = Mat(height_, width_, CV_32S);


	//�e��p�q�X�g�O�����i�[�ϐ�
	fg** histo_col = new fg *[width_];
	for (int i = 0; i < width_; i++)
	{
		histo_col[i] = (fg*)_aligned_malloc(sizeof(fg) * Imax, 32);
		memset(histo_col[i], 0, sizeof(fg) * Imax);
	}


	//mean(g), mean(gg)�����v�Z�p�ϐ��i�a�ŕۑ��j
	//under_i_index�Ŏ������l�ȉ��̗�q�X�g�O�����a
	struct colDataStruct {
		int g_sum;
		int gg_sum;
		int under_i_f;
		int under_i_g;

	};
	colDataStruct* colData = new colDataStruct[width_];
	memset(colData, 0, sizeof(colDataStruct) * width_);

	//under_i_index�Ŏ������l�ȉ� �̂Ƃ���index
	int* under_i_col_index = new int[width_];
	memset(under_i_col_index, 0, sizeof(int) * width_);

	//Window���q�X�g�O����
	fg* histo_window = (fg*)_aligned_malloc(sizeof(fg) * Imax, 32);


	//
	int y_add = r - 1;//�폜������
	int y_rem = -r - 2;//�폜�����s


	//�����J�n
	for (int y = 0; y < height_; y++)
	{
		//cout << "y:" << y << endl;

		y_add++;
		y_rem++;

		bool hasPreviousRow = y_rem >= 0;
		bool hasNestRow = y_add < height_;

		//������

		//Window�֌W
		//�a
		int g_sum_window = 0;
		int gg_sum_window = 0;
		int pixel_sum_window = 0;
		//�E�B���h�E�q�X�g�O����������
		memset(histo_window, 0, sizeof(fg) * Imax);

		//index�ȉ�
		int f_sum_under_i_window = 0;
		int g_sum_under_i_window = 0;



		//1�񓖂���̉�f��
		int pixel_sum_col = 2 * r + 1;
		if (y < r)
			pixel_sum_col = y + r + 1;
		else if ((y + r) >= height_)
			pixel_sum_col = height_ - y + r;


		//��������index y = 0�Ȃ���͂ŏ������Ay > 0�Ȃ�[y - 1, 0]�̒l�ŏ�����
		int index;
		if (y > 0)
			index = *med_idx.ptr<int>(y - 1);
		else
			index = *I.ptr<int>(0);

		int x_add = -1;// �ǉ�������
		int x_rem = -2 * r - 2;// �폜������
		//�Ή������f�ւ̃|�C���^
		//��̍폜��f
		int* I_rem = NULL;
		int* G_rem = NULL;
		if (hasPreviousRow) {
			I_rem = I.ptr<int>(y_rem);
			G_rem = G.ptr<int>(y_rem);
		}
		//��̒ǉ���f
		int* I_add = NULL;
		int* G_add = NULL;
		if (hasNestRow) {
			I_add = I.ptr<int>(y_add);
			G_add = G.ptr<int>(y_add);
		}
		//�������̉�f
		int* G_center = G.ptr<int>(y);
		int* med_center = med_idx.ptr<int>(y);

		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 0.0f;

		for (int x = -r; x < width_; x++)
		{
			x_add++;
			x_rem++;


			//�ǉ����鎟�̗񂪂��邩
			bool hasNextColumn = x_add < width_;
			//�폜����O�̗񂪂��邩
			bool hasPreviousColumn = x_rem >= 0;

			//window����f���̍X�V
			if (hasNextColumn && !hasPreviousColumn) {
				//�ǉ��̂�
				pixel_sum_window += pixel_sum_col;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}
			else if (!hasNextColumn && hasPreviousColumn)
			{
				//�폜�̂�
				pixel_sum_window -= pixel_sum_col;
				pixel_sum_window_inv = 1.0f / pixel_sum_window;
			}

			//��q�X�g�O�����̍X�V

			//���̗񂪂���Ȃ�X�V
			if (hasNextColumn) {
				if (y > 0) {

					//�O�̍s������Ȃ�q�X�g�O��������폜
					if (hasPreviousRow)
					{
						//�O�̍s��g��f�擾
						//�q�X�g�O�������猸�Z
						int fidx = I_rem[x_add];//�s�͍폜�s�A��͒ǉ��s�Ȃ̂ł����Ȃ�
						int gval = G_rem[x_add];
						histo_col[x_add][fidx].f -= 1;
						histo_col[x_add][fidx].g -= gval;

						//SIMD���\(���ʂ��邩������Ȃ�����)
						//g_sum_col, gg_sum_col���猸�Z
						colData[x_add].g_sum -= gval;
						colData[x_add].gg_sum -= gval * gval;
						if (fidx <= under_i_col_index[x_add]) {
							colData[x_add].under_i_f -= 1;
							colData[x_add].under_i_g -= gval;
						}

						/*
						__m128i a128 = _mm_load_si128((__m128i *)&colData[x_add]);
						int m1 = fidx <= under_i_col_index[x_add];
						__m128i m128 = _mm_set_epi32(gval * m1, m1, gval * gval, gval);
						__m128i c128 = _mm_sub_epi32(a128, m128);
						_mm_store_si128((__m128i *)&colData[x_add], c128);
						*/
					}
					//���̍s������Ȃ�X�V
					if (hasNestRow) {
						//���̍s���q�X�g�O�����ɒǉ�
						//���̍s��g��f�擾
						//�q�X�g�O�����ɉ��Z
						int fidx = I_add[x_add];
						int gval = G_add[x_add];
						histo_col[x_add][fidx].f += 1;
						histo_col[x_add][fidx].g += gval;

						//SIMD���\(���ʂ��邩������Ȃ�����)
						//g_sum_col, gg_sum_col�ɉ��Z
						colData[x_add].g_sum += gval;
						colData[x_add].gg_sum += gval * gval;
						//�ǉ��v�f��f������index�ȉ����ǂ����𔻒肵�A�ȉ��Ȃ�ǉ����邽�߂ɑ����A���傫���Ȃ�sum�Ɋ܂܂�Ȃ��̂ŉ������Ȃ�
						if (fidx <= under_i_col_index[x_add]) {
							colData[x_add].under_i_f += 1;
							colData[x_add].under_i_g += gval;
						}

						/*
						__m128i a128 = _mm_load_si128((__m128i *)&colData[x_add]);
						int m1 = fidx <= under_i_col_index[x_add];
						__m128i m128 = _mm_set_epi32(gval * m1, m1, gval * gval, gval);
						__m128i c128 = _mm_add_epi32(a128, m128);
						_mm_store_si128((__m128i *)&colData[x_add], c128);
						*/
					}
				}
				else
				{
					//1�s��(y=0)�p�����@�񂷂ׂĒǉ�
					for (int yy = 0; yy <= r; yy++)
					{
						//g��f�擾
						//�q�X�g�O�����ɉ��Z
						int fidx = I.ptr<int>(yy)[x_add];
						int gval = G.ptr<int>(yy)[x_add];
						histo_col[x_add][fidx].f += 1;
						histo_col[x_add][fidx].g += gval;
						//g_sum_col, gg_sum_col�ɉ��Z
						colData[x_add].g_sum += gval;
						colData[x_add].gg_sum += gval * gval;
					}

				}
				//window�X�V
				//sum_window�ɒǉ�������Z
				g_sum_window += colData[x_add].g_sum;
				gg_sum_window += colData[x_add].gg_sum;
				//�q�X�g�O�����ɒǉ�
				addHistogram(Imax, histo_window, histo_col[x_add]);

				int u_idx = under_i_col_index[x_add];
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//����SIMD���\
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//�ǉ�
					colData[x_add].under_i_f += histo_col[x_add][i].f * sign1;
					colData[x_add].under_i_g += histo_col[x_add][i].g * sign1;
				}

				//under index�̍X�V
				under_i_col_index[x_add] = index;

				//�E�B���h�E��f_sum, g_sum�̍X�V
				f_sum_under_i_window += colData[x_add].under_i_f;
				g_sum_under_i_window += colData[x_add].under_i_g;

			}

			//�폜�񂪂���Ȃ�
			if (hasPreviousColumn)
			{
				//window�X�V
				//�q�X�g�O��������폜
				remHistogram(Imax, histo_window, histo_col[x_rem]);


				//sum_window����폜������Z
				g_sum_window -= colData[x_rem].g_sum;
				gg_sum_window -= colData[x_rem].gg_sum;
				//�E�B���h�E��f_sum, g_sum�̍X�V
				f_sum_under_i_window -= colData[x_rem].under_i_f;
				g_sum_under_i_window -= colData[x_rem].under_i_g;


				int u_idx = under_i_col_index[x_rem];
				int flag1 = u_idx < index;
				int sign1 = flag1 * 2 - 1;
				int startIdx = u_idx + flag1;
				int numIdx = (index - u_idx) * sign1;
				//����SIMD���\
				for (int i = startIdx, j = 0; j < numIdx; i += sign1, j++)
				{
					//�ǉ�
					f_sum_under_i_window -= histo_col[x_rem][i].f * sign1;
					g_sum_under_i_window -= histo_col[x_rem][i].g * sign1;
				}
			}

			//�����l�̌v�Z
			if (x >= 0)
			{
				float g_ave = g_sum_window * pixel_sum_window_inv;
				float gg_ave = gg_sum_window * pixel_sum_window_inv;
				float vx = gg_ave - g_ave * g_ave + eps2;
				float tmp = G_center[x] - g_ave;
				float cx = tmp * pixel_sum_window_inv / vx;
				float dx = pixel_sum_window_inv - g_ave * cx;
				float h = cx * g_sum_under_i_window + dx * f_sum_under_i_window;

				int flag1 = h < half;
				int flag2 = flag1 - 1;
				int sign1 = flag1 * 2 - 1;
				while (true)
				{
					index += flag1;
					//if (histo_window[index].f)
					{
						//����Ȃ�X�V����h�`�F�b�N
						f_sum_under_i_window += histo_window[index].f * sign1;
						g_sum_under_i_window += histo_window[index].g * sign1;
						h = cx * g_sum_under_i_window + dx * f_sum_under_i_window;
						if ((h >= half) == flag1)
						{
							//�������̂ł���index��median
							med_center[x] = index;
							index += flag2;
							break;
						}
					}
					index += flag2;
				}
			}

		}

	}



	//�������J��
	for (int i = 0; i < width_; i++)
	{
		_aligned_free(histo_col[i]);
	}
	delete[] histo_col;
	_aligned_free(histo_window);
	delete[] colData;


	return med_idx;
}



//2D�@����1�`�����l�� ���������p
template<typename GSum, typename FGSumUpToIndex, typename FG, typename GTYPE, typename CTYPE>
Mat FGMF::filter2DWindow(Mat& I, Mat& G, int radius, float eps2, int Imax)
{
	//check validation
	//�ǂ����int�^�A�`�����l����1
	assert(I.depth() == CV_32S && I.channels() == 1);
	//assert(G.depth() == CV_32S && G.channels() == 1);
	//assert(I.isContinuous() && G.isContinuous());
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
	const vector<int> r_dim{ radius,radius };

	//�E�B���h�E�A�N�Z�X�p
	//size_dim = {dim0, dim1, dim3, dim4}�̂Ƃ�
	// = {dim0, dim0*dim1, dim0*dim1*dim3}
	//�Ƃ��邱�ƂŁA�Ⴆ�� [x1,x2,x3,x4]�ɃA�N�Z�X����Ƃ��A�E�B���h�E��������ł͂��ꂪ1�����ŕ���ł���̂�
	// x4*dim0*dim1*dim3 + x3*dim0*dim1 + x2*dim0 + x1
	const vector<int> size_prod{ size_dim[0] };

	//�������m�ہE������
	//���ʕۑ�
	Mat result = Mat(I.rows, I.cols, CV_32S);


	//
	// (gSum, sumUpToIndex, histo)
	//

	//W0:��f�Ȃ̂ŋL�^�̕K�v�Ȃ��iI,G�j

	//W1
	Window_vector<GSum, FGSumUpToIndex, FG> W1 = Window_vector<GSum, FGSumUpToIndex, FG>(Imax, size_dim, 1, true);

	//W2(Wmain)
	Window_single<GSum, FGSumUpToIndex, FG> Wmain = Window_single<GSum, FGSumUpToIndex, FG>(Imax);


	//��f��
	vector<int> pixel_sum(DIM);
	//�ʒu
	vector<Pos> x(DIM);
	//�X�e�[�^�X
	vector<DimStatus> status(DIM);

	//�Ή������f�ւ̃|�C���^
	//�������̉�f
	int* result_center = result.ptr<int>(0);
	GTYPE* G_center = G.ptr<GTYPE>(0);
	int* W0_rem_f = I.ptr<int>(0);
	GTYPE* W0_rem_g = G.ptr<GTYPE>(0);
	int* W0_add_f = I.ptr<int>(0);
	GTYPE* W0_add_g = G.ptr<GTYPE>(0);


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
		Wmain.initialize();

		//�E�B���h�E����f��
		int pixel_sum_window = 0;
		//�E�B���h�E����f���̋t��
		float pixel_sum_window_inv = 0.0f;



		/*
		�e�K�w�̃E�B���h�E�ɑ΂���|�C���^�����ƍl����΁A��f�����łȂ�win�n�������悤�ɏ�����̂ł�

		dim0�����ɃX���C�h (W0�͉�f)
		W1[x[1].add, x[0].add] = W1[x[1].add, x[0].add] + W0[x[2].add, x[1].add, x[0].add] - W0[x[2].rem, x[1].add, x[0].add]
		W2[x[0].add] = W2[x[0].add] + W1[x[1].add, x[0].add] - W1[x[1].rem, x[0].add]
		W3 = W3 + W2[x[0].add] - W2[x[0].rem] (W3��window)
		*/

		//W2


		//���̊K�w�̏����ʒu�Z�b�g
		setPos(x[0], r_dim[0]);
		for (; x[0].center < size_dim[0]; x[0].center++, x[0].add++, x[0].rem++)
		{
			//�X�e�[�^�X�Z�b�g
			setStatus(x[0], size_dim[0], status[1].isInside_image, status[0]);


			//window����f���̍X�V
			calculatePixelNumAtIntermostLoop(pixel_sum_window, pixel_sum_window_inv, pixel_sum[1], status[0]);

			//W2�̍X�V
			if (status[0].hasAdd)
			{
				//W1[x[0].add] �̍X�V
				if (status[1].hasAdd)//��f�ǉ�	(W1[x[0].add]) + W0[x[1].add, x[0].add]
				{
					addPixelToWindow_gSum(W1, x[0].add, *W0_add_f, *W0_add_g);
					W0_add_f++;
					W0_add_g++;
				}
				if (status[1].hasRem)//��f�폜	(W1[x[0].add]) - W0[x[1].rem, x[0].add]
				{
					removePixelFromWindow_gSum(W1, x[0].add, *W0_rem_f, *W0_rem_g);
					W0_rem_f++;
					W0_rem_g++;
				}
				//W1�ǉ�	(W2) + W1[x[0].add]
				updateSubWindowAndAddToWindow_gSum(Imax, Wmain, W1, x[0].add);
			}
			if (status[0].hasRem)//W1�폜	(W2) - W1[x[0].rem]
				removeSubWindowFromWindow_gSum(Imax, Wmain, W1, x[0].rem);


			//�����l�̌v�Z
			if (status[0].isInside_image)
			{
				CTYPE cx;
				float dx;
				calculateCxDx(Wmain.gsum, pixel_sum_window_inv, eps2, *G_center, cx, dx);
				//findMedian(cx, dx, half, Wmain.sumUpToIndex.index, *result_center, Wmain);
				findMedian(cx, dx, half, Wmain, *result_center);
				G_center++;
				result_center++;
			}
		}
	}


	return result;
}



#endif