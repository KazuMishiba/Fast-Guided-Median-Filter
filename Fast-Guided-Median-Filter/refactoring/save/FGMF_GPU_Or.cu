
namespace FGMF_GPU_Or
{

/*
	__device__ inline int
		de_calculateHcum(const int& W_f_cum, const int* W_g_cum, const float* c_x, const float& d_x, const int channelNum_g)
	{
		float sum = d_x * W_f_cum;
		for (int i = 0; i < channelNum_g; i++)
			sum += c_x[i] * W_g_cum[i];
		return sum;
	}

	__device__ inline int
		searchWeightedMedian(const int* W_F, const int** W_G, int& W_f_cum, int** W_g_cum, int W_k, float** c_x, float& d_x, int channelNum_g)
	{
		float h = de_calculateHcum(W_f_cum, W_g_cum, c_x, d_x, channelNum_g);

		const int flagA = h < 0.5f;
		const int flag2 = flagA - 1;
		const int sign = flagA * 2 - 1;

		while (true)
		{
			W_k += flagA;
			//if(histogram[W_k].x)//�����̓R�����g�O���Ă���������
			{
				W_f_cum += W_F[W_k] * sign;
				for (int i = 0; i < channelNum_g; i++)
					W_g_cum[i] += W_G[W_k][i] * sign;

				h = de_calculateHcum(W_f_cum, W_g_cum, c_x, d_x, channelNum_g);

				if ((h >= 0.5f) == flagA)
				{
					int result_center = W_k;
					W_k += flag2;
					return result_center;
				}
			}
			W_k += flag2;
		}
	}

	__device__ inline void
	de_addPixelToWindow(const int& f_x, const int* g_x, int* W_F, int** W_G, const int& W_f_cum, const int** W_g_cum, const int W_k, const int channelNum_g)
	{
		atomicAdd(&W_F[f_x], 1);
		for (int i = 0; i < channelNum_g; i++)
			atomicAdd(&W_G[f_x][i], g_x[i]);
		if (f_x <= W_k)
		{
			atomicAdd(&W_f_cum, 1);
			for (int i = 0; i < channelNum_g; i++)
				atomicAdd(&W_g_cum[i], g_x[i]);
		}
	}

	__device__ inline void
		de_removePixelFromWindow(const int& f_x, const int* g_x, int* W_F, int** W_G, const int channelNum_g)
	{

	}

	//gX
	__global__ void
		de_filter2D(int width, int height, int radius, int Imax, int* result_center, cudaTextureObject_t texF, cudaTextureObject_t* texGX, float** CxDx, size_t pitchI1, size_t pitchF1, int channelNum_g)
	{
		int s_center = blockIdx.x;
		if (s_center >= width || threadIdx.x >= radius * 2 + 1)
			return;
		int s = blockIdx.x + threadIdx.x - radius;


		__shared__ int W_k;
		__shared__ int W_f_cum;
		extern __shared__ int sdata[];//sizeof(int) * (1 + Imax) * channelNum_g
		int* W_F = &sdata[0];//sizeof(int) * Imax
		int* W_G = &sdata[Imax];//sizeof(int) * Imax * channelNum_g
		int* W_g_cum = &sdata[Imax * (1 + channelNum_g)];//sizeof(int) * channelNum_g

		int f;
		int* g = new int[channelNum_g];
		float* cx = new int[channelNum_g];
		float dx;

		//���S�X���b�h�̂ݎ��s
		if (threadIdx.x == radius)
		{
			//Initialize W
			//�q�X�g�O����������
			for (int i = 0; i < (1 + Imax) * channelNum_g; i++)
				sdata[i] = 0;

			W_k = tex2D<int>(texF, s, 0);//current index
		}
		//thread����
		__syncthreads();

		//1�ڃq�X�g�O�����`��
		for (int yy = -radius; yy <= radius; yy++)
		{
			//x�����̃q�X�g�O�����`���͊e�X���b�h���S������
			{
				f = tex2D<int>(texF, s, yy);
				for (int i = 0; i < channelNum_g; i++)
					g[i] = tex2D<int>(texGX[i], s, yy);
				for (int i = 0; i < channelNum_g; i++)
					atomicAdd(&histogramX[f * k + i], g[i]);
				atomicAdd(&histogramX[f * k + n], 1);
				if (f <= W_k)
				{
					for (int i = 0; i < channelNum_g; i++)
						atomicAdd(&fgXSumUpToIndex[i], g[i]);
					atomicAdd(&fgXSumUpToIndex[n], 1);
				}

			}
		}

		//thread����
		__syncthreads();


		//1�s�ڂ̒����l�v�Z
		//���S�X���b�h�̂ݎ��s
		if (threadIdx.x == radius)
		{
			for (int i = 0; i < channelNum_g; i++)
//				cxdx[i] = *((float*)((char*)CxDx[i]) + x);
//			*((int*)((char*)result_center) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, W_k, n);
		}
		//thread����
		__syncthreads();

		//2�s�ڈȍ~�̏���
		for (int t = 1; t < height; t++)
		{
			int tp = t + radius;
			int tm = t - radius - 1;
			//Add pixel at(s, t+) to W(2)
			f = tex2D<int>(texF, s, tp);
			for (int i = 0; i < channelNum_g; i++)
				g[i] = tex2D<int>(texGX[i], s, tp);
			de_addPixelToWindow(f, g, W_F, W_G, W_f_cum, W_g_cum, W_k, channelNum_g);

			//Remove pixel at(s, t-) from W(2)
			f = tex2D<int>(texF, s, tm);
			for (int i = 0; i < channelNum_g; i++)
				g[i] = tex2D<int>(texGX[i], s, tm);
			de_removePixelFromWindow(f, g, W_F, W_G, channelNum_g);

			//thread����
			__syncthreads();

			
			//���S�X���b�h�̂ݎ��s
			if (threadIdx.x == radius)
			{
				for (int i = 0; i < k; i++)
					cxdx[i] = *((float*)((char*)CxDx[i] + t * pitchF1) + x);
				//�����l�v�Z
				*((int*)((char*)result_center + t * pitchI1) + x) = de_findMedian(cxdx, histogramX, fgXSumUpToIndex, W_k, n);
			}
			
			//thread����
			__syncthreads();
		}


		delete g;
		if (threadIdx.x == radius)
		{
			//delete cxdx;
		}
		__syncthreads();
	}
	*/

}