#include "ConstantTimeWMF.cuh"

/*
事前計算可能なものは
meanG, corrG, varG + eps2
*/

__global__ void
de_checkForDebug(int width, int height, float* I, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	if (x == 30 && y == 30) {
		printf("Checked:%f\n", *((float*)((char*)I + y * pitch) + x));
	}
}
void cu_checkForDebug(float* I, SizeInfo& sizeInfo)
{
	de_checkForDebug << <sizeInfo.gridSize, sizeInfo.blockSize, 0, NULL >> > (sizeInfo.width, sizeInfo.height, I, sizeInfo.pitch<float>());
}


//GG = G*G, GI = G*I
__global__ void
de_GGGI(int width, int height, float* G, float* I, float* GG, float* GI, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float g = *((float*)((char*)G + y * pitch) + x);
	float i = *((float*)((char*)I + y * pitch) + x);
	*((float*)((char*)GG + y * pitch) + x) = g * g;
	*((float*)((char*)GI + y * pitch) + x) = g * i;
}
//varG = corrG - meanG * meanG, covGI = corrGI - meanG * meanI
__global__ void
de_varGcovGI(int width, int height, float* corrG, float* meanG, float* corrGI, float* meanI, float* varG, float* covGI, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float corrg = *((float*)((char*)corrG + y * pitch) + x);
	float meang = *((float*)((char*)meanG + y * pitch) + x);
	float corrgi = *((float*)((char*)corrGI + y * pitch) + x);
	float meani = *((float*)((char*)meanI + y * pitch) + x);
	*((float*)((char*)varG + y * pitch) + x) = corrg - meang * meang;
	*((float*)((char*)covGI + y * pitch) + x) = corrgi - meang * meani;
}
//a = covGI / (varG + e), b = meanI - a * meanG
__global__ void
de_ab(int width, int height, float* covGI, float* varG, float* meanI, float* meanG, float* a, float* b, float e, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float covgi = *((float*)((char*)covGI + y * pitch) + x);
	float varg = *((float*)((char*)varG + y * pitch) + x) + e;
	float meani = *((float*)((char*)meanI + y * pitch) + x);
	float meang = *((float*)((char*)meanG + y * pitch) + x);
	float a_ = covgi / varg;
	*((float*)((char*)a + y * pitch) + x) = a_;
	*((float*)((char*)b + y * pitch) + x) = meani - a_ * meang;
}
//dst = mean_a * G + mean_b
__global__ void
de_agb(int width, int height, float* mean_a, float* G, float* mean_b, float* dst, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float meana = *((float*)((char*)mean_a + y * pitch) + x);
	float meanb = *((float*)((char*)mean_b + y * pitch) + x);
	float g = *((float*)((char*)G + y * pitch) + x);
	*((float*)((char*)dst + y * pitch) + x) = meana * g + meanb;
}

//a = inv(varG + e) * covGI, b = meanI - a * meanG
__global__ void
de_ab(int width, int height, float* covGI_1, float* covGI_2, float* covGI_3, float* varG_11, float* varG_12, float* varG_13, float* varG_22, float* varG_23, float* varG_33, float* meanI, float* meanG_1, float* meanG_2, float* meanG_3, float* a_1, float* a_2, float* a_3, float* b, float e, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	const float v11 = *((float*)((char*)varG_11 + y * pitch) + x) + e;
	const float v12 = *((float*)((char*)varG_12 + y * pitch) + x);
	const float v13 = *((float*)((char*)varG_13 + y * pitch) + x);
	const float v22 = *((float*)((char*)varG_22 + y * pitch) + x) + e;
	const float v23 = *((float*)((char*)varG_23 + y * pitch) + x);
	const float v33 = *((float*)((char*)varG_33 + y * pitch) + x) + e;
	const float delta =
		v11 * v22 * v33 +
		v12 * v23 * v13 * 2 -
		v13 * v13 * v22 -
		v12 * v12 * v33 -
		v11 * v23 * v23;
	const float vinv11 = (v22 * v33 - v23 * v23);
	const float vinv12 = (v13 * v23 - v12 * v33);
	const float vinv13 = (v12 * v23 - v13 * v22);
	const float vinv22 = (v11 * v33 - v13 * v13);
	const float vinv23 = (v13 * v12 - v11 * v23);
	const float vinv33 = (v11 * v22 - v12 * v12);

	float covgi_1 = *((float*)((char*)covGI_1 + y * pitch) + x);
	float covgi_2 = *((float*)((char*)covGI_2 + y * pitch) + x);
	float covgi_3 = *((float*)((char*)covGI_3 + y * pitch) + x);
	float meani = *((float*)((char*)meanI + y * pitch) + x);
	float meang_1 = *((float*)((char*)meanG_1 + y * pitch) + x);
	float meang_2 = *((float*)((char*)meanG_2 + y * pitch) + x);
	float meang_3 = *((float*)((char*)meanG_3 + y * pitch) + x);

	float a1 = (vinv11 * covgi_1 + vinv12 * covgi_2 + vinv13 * covgi_3) / delta;
	float a2 = (vinv12 * covgi_1 + vinv22 * covgi_2 + vinv23 * covgi_3) / delta;
	float a3 = (vinv13 * covgi_1 + vinv23 * covgi_2 + vinv33 * covgi_3) / delta;
	*((float*)((char*)a_1 + y * pitch) + x) = a1;
	*((float*)((char*)a_2 + y * pitch) + x) = a2;
	*((float*)((char*)a_3 + y * pitch) + x) = a3;
	*((float*)((char*)b + y * pitch) + x) = meani - (a1 * meang_1 + a2 * meang_2 + a3 * meang_3);

}
//dst = mean_a * G + mean_b
__global__ void
de_agb(int width, int height, float* mean_a1, float* mean_a2, float* mean_a3, float* G1, float* G2, float* G3, float* mean_b, float* dst, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float meana1 = *((float*)((char*)mean_a1 + y * pitch) + x);
	float meana2 = *((float*)((char*)mean_a2 + y * pitch) + x);
	float meana3 = *((float*)((char*)mean_a3 + y * pitch) + x);
	float meanb = *((float*)((char*)mean_b + y * pitch) + x);
	float g1 = *((float*)((char*)G1 + y * pitch) + x);
	float g2 = *((float*)((char*)G2 + y * pitch) + x);
	float g3 = *((float*)((char*)G3 + y * pitch) + x);
	*((float*)((char*)dst + y * pitch) + x) = meana1 * g1 + meana2 * g2 + meana3 * g3 + meanb;

}

//covGI = corrGI - meanG * meanI
__global__ void
de_covGI(int width, int height, float* corrGI, float* meanG, float* meanI, float* covGI, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float meang = *((float*)((char*)meanG + y * pitch) + x);
	float corrgi = *((float*)((char*)corrGI + y * pitch) + x);
	float meani = *((float*)((char*)meanI + y * pitch) + x);
	*((float*)((char*)covGI + y * pitch) + x) = corrgi - meang * meani;
}

//varG = corrG - meanG * meanG
__global__ void
de_varG(int width, int height, float* corrG, float* meanG, float* varG, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float corrg = *((float*)((char*)corrG + y * pitch) + x);
	float meang = *((float*)((char*)meanG + y * pitch) + x);
	*((float*)((char*)varG + y * pitch) + x) = corrg - meang * meang;
}

//varG xy = corrG xy - meanG x * meanG y
__global__ void
de_varG(int width, int height, float* corrG_11, float* corrG_12, float* corrG_13, float* corrG_22, float* corrG_23, float* corrG_33, float* meanG_1, float* meanG_2, float* meanG_3, float* varG_11, float* varG_12, float* varG_13, float* varG_22, float* varG_23, float* varG_33, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float corrg11 = *((float*)((char*)corrG_11 + y * pitch) + x);
	float corrg12 = *((float*)((char*)corrG_12 + y * pitch) + x);
	float corrg13 = *((float*)((char*)corrG_13 + y * pitch) + x);
	float corrg22 = *((float*)((char*)corrG_22 + y * pitch) + x);
	float corrg23 = *((float*)((char*)corrG_23 + y * pitch) + x);
	float corrg33 = *((float*)((char*)corrG_33 + y * pitch) + x);
	float meang1 = *((float*)((char*)meanG_1 + y * pitch) + x);
	float meang2 = *((float*)((char*)meanG_2 + y * pitch) + x);
	float meang3 = *((float*)((char*)meanG_3 + y * pitch) + x);
	*((float*)((char*)varG_11 + y * pitch) + x) = corrg11 - meang1 * meang1;
	*((float*)((char*)varG_12 + y * pitch) + x) = corrg12 - meang1 * meang2;
	*((float*)((char*)varG_13 + y * pitch) + x) = corrg13 - meang1 * meang3;
	*((float*)((char*)varG_22 + y * pitch) + x) = corrg22 - meang2 * meang2;
	*((float*)((char*)varG_23 + y * pitch) + x) = corrg23 - meang2 * meang3;
	*((float*)((char*)varG_33 + y * pitch) + x) = corrg33 - meang3 * meang3;
}

//Guided Filtering (temps = float*[8])
void cu_guidedFiltering(float* I, float* G, float* dst, int radius, float e, float** temps, SizeInfo& sizeInfo, cudaStream_t stream)
{
	cudaTextureObject_t tex;
	cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
	//float* srcTex;
	//temps[7]はテクスチャフェッチ？用ソースを格納(コピー)する。これはボックスフィルタリング中に書き換わるので、毎回ソースのコピーを用意する。
	//Utility::allocateDeviceMemory(temps[7], sizeInfo);
	UtilityForCUDA::copyDeviceMemory(G, temps[7], sizeInfo, stream);
	UtilityForCUDA::setLinearArrayToTexture(temps[7], tex, sizeInfo, filterMode);

	int blockSize = 32;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);

	//stream = NULL;
	//Utility::showDevice(temps[7], sizeInfo, "I", false, 1.0 / 255.0f);

	//meanG ([0]) = mean(G) 
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[7], temps[0], temps[6], radius, stream, tex, sizeInfo.pitch<float>());
	//Utility::showDevice(temps[0], sizeInfo, "meanG", false, 1.0f/255.0f);

	//meanI ([1]) = mean(I) 
	//cu_meanFiltering(blockSize, gridSize, stream, width, height, I, temps[1], temps[6], radius, pitch);
	UtilityForCUDA::copyDeviceMemory(I, temps[7], sizeInfo, stream);
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[7], temps[1], temps[6], radius, stream, tex, sizeInfo.pitch<float>());

	//GG ([2]) = G*G, GI ([3]) = G*I
	de_GGGI << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, I, temps[2], temps[3], sizeInfo.pitch<float>());

	//corrG ([4]) = mean(GG)
	UtilityForCUDA::copyDeviceMemory(temps[2], temps[7], sizeInfo, stream);
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[7], temps[4], temps[6], radius, stream, tex, sizeInfo.pitch<float>());

	//corrGI ([5]) = mean(GI)
	UtilityForCUDA::copyDeviceMemory(temps[3], temps[7], sizeInfo, stream);
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[7], temps[5], temps[6], radius, stream, tex, sizeInfo.pitch<float>());

	//varG ([2]) = corrG - meanG * meanG, covGI ([3]) = corrGI - meanG * meanI
	de_varGcovGI << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[4], temps[0], temps[5], temps[1], temps[2], temps[3], sizeInfo.pitch<float>());

	//a ([4]) = covGI / (varG + e), b ([5]) = meanI - a * meanG
	de_ab << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[3], temps[2], temps[1], temps[0], temps[4], temps[5], e, sizeInfo.pitch<float>());


	//mean_a ([0]) = mean(a)
	UtilityForCUDA::copyDeviceMemory(temps[4], temps[7], sizeInfo, stream);
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[7], temps[0], temps[6], radius, stream, tex, sizeInfo.pitch<float>());

	//mean_b ([1]) = mean(b)
	UtilityForCUDA::copyDeviceMemory(temps[5], temps[7], sizeInfo, stream);
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[7], temps[1], temps[6], radius, stream, tex, sizeInfo.pitch<float>());

	//dst = mean_a * G + mean_b
	de_agb << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[0], G, temps[1], dst, sizeInfo.pitch<float>());

	//dst = a * G + b
	//de_agb << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[4], G, temps[5], dst, sizeInfo.pitch<float>());

	//Utility::showDevice(dst, sizeInfo, "dst");

	cudaDestroyTextureObject(tex);
}





//二つの要素の掛け算
__global__ void
de_Mult(int width, int height, float* src1, float* src2, float* dst, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	*((float*)((char*)dst + y * pitch) + x) = *((float*)((char*)src1 + y * pitch) + x) * *((float*)((char*)src2 + y * pitch) + x);
}
//2乗
__global__ void
de_Pow2(int width, int height, float* src, float* dst, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float val = *((float*)((char*)src + y * pitch) + x);
	*((float*)((char*)dst + y * pitch) + x) = val * val;
}

//Guided Filtering事前計算要素使用 (temps = float*[5])
void cu_GuidedFilteringWithPrecalculation(float* I, float* G, float* meanG, float* varG, int radius, float e, float** temps, float* result, SizeInfo& sizeInfo, cudaStream_t stream)
{
	int blockSize = 32;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);

	//meanI ([0]) = mean(I)
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, I, temps[0], temps[1], temps[2], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

	//corrGI = mean(G*I)
	//GI ([1]) = G*I
	de_Mult << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, I, temps[1], sizeInfo.pitch<float>());
	//corrGI ([2]) = mean(GI)
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[1], temps[2], temps[3], temps[4], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

	//covGI ([3]) = corrGI - meanG * meanI
	de_covGI << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[2], meanG, temps[0], temps[3], sizeInfo.pitch<float>());

	//a([1]) = covGI / (varG + e)
	//b([2]) = meanI - a * meanG
	de_ab << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[3], varG, temps[0], meanG, temps[1], temps[2], e, sizeInfo.pitch<float>());

	//mean_a ([0]) = mean(a)
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[1], temps[0], temps[3], temps[4], radius, stream, sizeInfo.pitch<float>(), sizeInfo);
	//mean_b ([1]) = mean(b)
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[2], temps[1], temps[3], temps[4], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

	//result = mean_a * G + mean_b
	de_agb << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[0], G, temps[1], result, sizeInfo.pitch<float>());

}

//Guided Filtering高速計算事前計算要素 (temps = float*[4])
//meanG, varG + eps2
void cu_precalculationForGuidedFiltering(float* G, int radius, float e, float** temps, float* meanG, float* varG, SizeInfo& sizeInfo, cudaStream_t stream)
{
	int blockSize = 32;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);


	//meanG = mean(G) 
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, G, meanG, temps[0], temps[1], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

	//corrG (temps4) = mean(G*G)
	//GG (temps[0]) = G*G
	de_Pow2 << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G, temps[0], sizeInfo.pitch<float>());
	//corrG = mean(GG)
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[0], temps[3], temps[1], temps[2], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

	//varG
	de_varG << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[3], meanG, varG, sizeInfo.pitch<float>());
}


//ガイドカラー用
//Guided Filtering事前計算要素使用 (temps = float*[9]) varGは3*3行列の上三角分の要素をスキャンライン順
void cu_GuidedFilteringWithPrecalculation(float* I, float** G, float** meanG, float** varG, int radius, float e, float** temps, float* result, SizeInfo& sizeInfo, cudaStream_t stream)
{
	int blockSize = 32;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);

	//meanI ([0]) = mean(I)
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, I, temps[0], temps[1], temps[2], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

	//corrGI[x] = mean(G[x]*I)
	//covGI ([3～5]) = corrGI[x] - meanG[x] * meanI
	for (int i = 0; i < 3; i++)
	{
		//GI ([1]) = G[x]*I
		de_Mult << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G[i], I, temps[1], sizeInfo.pitch<float>());
		//corrGI ([2]) = mean(GI)
		cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[1], temps[2], temps[6], temps[7], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

		//covGI ([3～5]) = corrGI[x] - meanG[x] * meanI
		de_covGI << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[2], meanG[i], temps[0], temps[3+i], sizeInfo.pitch<float>());
	}

	//a([6～8]) = inv(varG + e) * covGI
	//b([1]) = meanI - a * meanG
	de_ab << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[3], temps[4], temps[5], varG[0], varG[1], varG[2], varG[3], varG[4], varG[5], temps[0], meanG[0], meanG[1], meanG[2], temps[6], temps[7], temps[8], temps[1], e, sizeInfo.pitch<float>());


	//mean_a ([2～4]) = mean(a)
	for (int i = 0; i < 3; i++)
	{
		cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[6+i], temps[2+i], temps[0], temps[5], radius, stream, sizeInfo.pitch<float>(), sizeInfo);
	}
	//mean_b ([0]) = mean(b)
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[1], temps[0], temps[5], temps[6], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

	//result = mean_a * G + mean_b
	de_agb << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[2], temps[3], temps[4], G[0], G[1], G[2], temps[0], result, sizeInfo.pitch<float>());

}
//ガイドカラー用
//Guided Filtering高速計算事前計算要素 (temps = float*[9])
//meanG, varG ベクトル、行列
void cu_precalculationForGuidedFiltering(float** G, int radius, float e, float** temps, float** meanG, float** varG, SizeInfo& sizeInfo, cudaStream_t stream)
{
	int blockSize = 32;
	int gridSizeY = ceil(sizeInfo.height / (float)blockSize);
	int gridSizeX = ceil(sizeInfo.width / (float)blockSize);

	//meanG = mean(G) 
	for (int i = 0; i < 3; i++)
	{
		cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, G[i], meanG[i], temps[0], temps[1], radius, stream, sizeInfo.pitch<float>(), sizeInfo);
	}
	//varG =[v11, v12, v13; v12, v22, v23, v13, v23, v33];


	//corrG xy() = mean(Gx*Gy)
	//GxGy  = Gx*Gy
	//corrG xy (temps[0～5]) = mean(GxGy)
	//11
	de_Pow2 << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G[0], temps[8], sizeInfo.pitch<float>());
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[8], temps[0], temps[6], temps[7], radius, stream, sizeInfo.pitch<float>(), sizeInfo);
	//12
	de_Mult << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G[0], G[1], temps[8], sizeInfo.pitch<float>());
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[8], temps[1], temps[6], temps[7], radius, stream, sizeInfo.pitch<float>(), sizeInfo);
	//13
	de_Mult << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G[0], G[2], temps[8], sizeInfo.pitch<float>());
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[8], temps[2], temps[6], temps[7], radius, stream, sizeInfo.pitch<float>(), sizeInfo);
	//22
	de_Pow2 << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G[1], temps[8], sizeInfo.pitch<float>());
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[8], temps[3], temps[6], temps[7], radius, stream, sizeInfo.pitch<float>(), sizeInfo);
	//23
	de_Mult << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G[1], G[2], temps[8], sizeInfo.pitch<float>());
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[8], temps[4], temps[6], temps[7], radius, stream, sizeInfo.pitch<float>(), sizeInfo);
	//33
	de_Pow2 << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, G[2], temps[8], sizeInfo.pitch<float>());
	cu_meanFiltering(blockSize, gridSizeX, gridSizeY, sizeInfo.width, sizeInfo.height, temps[8], temps[5], temps[6], temps[7], radius, stream, sizeInfo.pitch<float>(), sizeInfo);

	//varG xy = corrG xy - meanG x * meanG y
	de_varG << <sizeInfo.gridSize, sizeInfo.blockSize, 0, stream >> > (sizeInfo.width, sizeInfo.height, temps[0], temps[1], temps[2], temps[3], temps[4], temps[5], meanG[0], meanG[1], meanG[2], varG[0], varG[1], varG[2], varG[3], varG[4], varG[5], sizeInfo.pitch<float>());
}



//IがtargetLevel以下の時に、uptoI=1, uptoG=g, そうでないときはuptoI=0, uptoG=0を格納する
__global__ void
de_UpToTargetLevel(int width, int height, int targetLevel, int* I, float* G, float* uptoI, float* uptoG, size_t pitchI1, size_t pitchF1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int i = *((int*)((char*)I + y * pitchI1) + x);
	float g = *((float*)((char*)G + y * pitchF1) + x);

	*((float*)((char*)uptoI + y * pitchF1) + x) = (float)(targetLevel <= i);
	*((float*)((char*)uptoG + y * pitchF1) + x) = (float)((targetLevel <= i) * g);
}

__global__ void
de_setValue(int width, int height, int value, int* dst, size_t pitchI1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	*((int*)((char*)dst + y * pitchI1) + x) = value;
}


// グリッドにスライス（に相当するデータ）をコピー (これ単にコピー関数でできるはず)
__global__ void
de_copyToSlice(int width, int height, int targetLevel, float* src, cudaPitchedPtr grid, size_t pitchF1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	//grid
	size_t gridPitch = grid.pitch;
	size_t pitchSlice = gridPitch * height;
	char* d_ptr = static_cast<char*>(grid.ptr) + y * gridPitch;
	//src
	float srcVal = *((float*)((char*)src + y * pitchF1) + x);

	d_ptr += (pitchSlice * targetLevel);

	*((float*)(d_ptr)+x) = srcVal;
}

// ３次元ボリュームに入力を0,1データとして格納
__global__ void
de_storeDataIntoVolume(int width, int height, int range, int* src, cudaPitchedPtr volume, size_t pitchI1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	//grid
	size_t gridPitch = volume.pitch;
	size_t pitchSlice = gridPitch * height;
	char* d_ptr = static_cast<char*>(volume.ptr) + y * gridPitch;
	//src
	int srcVal = *((int*)((char*)src + y * pitchI1) + x);

	//srcの値と一致するスライスに1、一致しない場合は0を格納する
	for (int i = 0; i < range; i++)
	{
		if (srcVal == i)
			*((float*)(d_ptr)+x) = 1.0f;
		else
			*((float*)(d_ptr)+x) = 0.0f;

		d_ptr += pitchSlice;
	}
}


// グリッド深さ方向に対するmedian計算
__global__ void
de_calculateMedian(int width, int height, int range, cudaPitchedPtr grid, int* dst, size_t pitchI1)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	//grid
	size_t gridPitch = grid.pitch;
	size_t pitchSlice = gridPitch * height;
	char* d_ptr = static_cast<char*>(grid.ptr) + y * gridPitch;


	//合計の計算
	float sum = 0.0f;
	for (int i = 0; i < range; i++)
	{
		float weight = *((float*)(d_ptr)+x);
		sum += weight;
		d_ptr += pitchSlice;
	}
	//中央値の計算
	//float half = 0.5f;
	float half = sum / 2;



	//中央値の探索
	d_ptr = static_cast<char*>(grid.ptr) + y * gridPitch;
	sum = 0.0f;
	for (int i = 0; i < range; i++)
	{
		sum += *((float*)(d_ptr)+x);
		if (sum >= half) {
			//dst
			*((int*)((char*)dst + y * pitchI1) + x) = i;
			break;
		}
		d_ptr += pitchSlice;
	}


}


void cu_storeDataIntoVolume(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int range, int* src, cudaPitchedPtr volume, size_t pitchI1)
{
	de_storeDataIntoVolume << <gridSize, blockSize, 0, stream >> > (width, height, range, src, volume, pitchI1);
}

void cu_copyToSlice(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int targetLevel, float* src, cudaPitchedPtr grid, size_t pitchF1)
{
	de_copyToSlice << <gridSize, blockSize, 0, stream >> > (width, height, targetLevel, src, grid, pitchF1);
}


void cu_calculateMedian(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int range, cudaPitchedPtr grid, int* dst, size_t pitchI1)
{
	de_calculateMedian << <gridSize, blockSize, 0, stream >> > (width, height, range, grid, dst, pitchI1);
}





//I1G1
void cu_ConstantTimeWMF(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, float* g, float eps2)
{
	//メモリ確保
	int tempNum = 5;
	float** temps;
	temps = new float*[tempNum];
	for (int i = 0; i < tempNum; i++)
		UtilityForCUDA::allocateDeviceMemory(temps[i], sizeInfo);

	float* meanG;
	float* varG;
	UtilityForCUDA::allocateDeviceMemory(meanG, sizeInfo);
	UtilityForCUDA::allocateDeviceMemory(varG, sizeInfo);
	float* dst;//スライスに対するGF結果格納用
	UtilityForCUDA::allocateDeviceMemory(dst, sizeInfo);

	//ガイデッドフィルタの事前要素計算
	cu_precalculationForGuidedFiltering(g, radius, eps2, temps, meanG, varG, sizeInfo, stream);

	//3次元メモリ確保
	cudaPitchedPtr volume;
	cudaExtent volumeSizeBytes = make_cudaExtent(sizeof(float) * sizeInfo.width, sizeInfo.height, Imax);
	gpuErrchk(cudaMalloc3D(&volume, volumeSizeBytes));
	size_t gridPitch = volume.pitch;
	size_t pitchSlice = gridPitch * sizeInfo.height;

	//入力画像を0,1としてボリュームにセット
	cu_storeDataIntoVolume(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, sizeInfo.range, f, volume, sizeInfo.pitch<int>());

	//各スライスに対してguided filter
	char* d_ptr = static_cast<char*>(volume.ptr);
	for (int i = 0; i < sizeInfo.range; i++)
	{
		float* ptr = (float*)(d_ptr);

		cu_GuidedFilteringWithPrecalculation(ptr, g, meanG, varG, radius, eps2, temps, dst, sizeInfo, stream);

		cu_copyToSlice(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, i, dst, volume, sizeInfo.pitch<float>());
		d_ptr += pitchSlice;
	}

	//中央値計算
	cu_calculateMedian(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, sizeInfo.range, volume, result_center, sizeInfo.pitch<int>());

	//メモリ開放
	cudaFree(dst);
	cudaFree(meanG);
	cudaFree(varG);
	for (int i = 0; i < tempNum; i++)
		cudaFree(temps[i]);
	delete temps;
	cudaFree(volume.ptr);
}


//I3G1
void cu_ConstantTimeWMF(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, float* g, float eps2)
{
	//メモリ確保
	int tempNum = 5;
	float** temps;
	temps = new float*[tempNum];
	for (int i = 0; i < tempNum; i++)
		UtilityForCUDA::allocateDeviceMemory(temps[i], sizeInfo);

	float* meanG;
	float* varG;
	UtilityForCUDA::allocateDeviceMemory(meanG, sizeInfo);
	UtilityForCUDA::allocateDeviceMemory(varG, sizeInfo);

	float* dst;//スライスに対するGF結果格納用
	UtilityForCUDA::allocateDeviceMemory(dst, sizeInfo);


	//ガイデッドフィルタの事前要素計算
	cu_precalculationForGuidedFiltering(g, radius, eps2, temps, meanG, varG, sizeInfo, stream);

	//3次元メモリ確保
	cudaPitchedPtr volume;
	cudaExtent volumeSizeBytes = make_cudaExtent(sizeof(float) * sizeInfo.width, sizeInfo.height, Imax);
	gpuErrchk(cudaMalloc3D(&volume, volumeSizeBytes));
	size_t gridPitch = volume.pitch;
	size_t pitchSlice = gridPitch * sizeInfo.height;

	//各入力チャンネルに対して処理
	for (int c = 0; c < 3; c++)
	{
		//入力画像を0,1としてボリュームにセット
		cu_storeDataIntoVolume(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, sizeInfo.range, f->host[c], volume, sizeInfo.pitch<int>());

		//各スライスに対してguided filter

		char* d_ptr = static_cast<char*>(volume.ptr);
		for (int i = 0; i < sizeInfo.range; i++)
		{
			float* ptr = (float*)(d_ptr);

			cu_GuidedFilteringWithPrecalculation(ptr, g, meanG, varG, radius, eps2, temps, dst, sizeInfo, stream);

			cu_copyToSlice(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, i, dst, volume, sizeInfo.pitch<float>());
			d_ptr += pitchSlice;
		}
		//中央値計算
		cu_calculateMedian(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, sizeInfo.range, volume, result_center->host[c], sizeInfo.pitch<int>());
	}

	//メモリ開放
	cudaFree(dst);
	cudaFree(meanG);
	cudaFree(varG);
	for (int i = 0; i < tempNum; i++)
		cudaFree(temps[i]);
	delete temps;
	cudaFree(volume.ptr);
}


//I1G3
void cu_ConstantTimeWMF(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, int* result_center, int* f, DeviceArray<float>* g, float eps2)
{
	//メモリ確保
	int tempNum = 9;
	float** temps;
	temps = new float*[tempNum];
	for (int i = 0; i < tempNum; i++)
		UtilityForCUDA::allocateDeviceMemory(temps[i], sizeInfo);

	float** meanG;
	float** varG;
	meanG = new float*[3];
	varG = new float*[6];
	for (int i = 0; i < 3; i++)
		UtilityForCUDA::allocateDeviceMemory(meanG[i], sizeInfo);
	for (int i = 0; i < 6; i++)
		UtilityForCUDA::allocateDeviceMemory(varG[i], sizeInfo);
	float* dst;//スライスに対するGF結果格納用
	UtilityForCUDA::allocateDeviceMemory(dst, sizeInfo);


	//ガイデッドフィルタの事前要素計算(3チャンネル)
	cu_precalculationForGuidedFiltering(g->host, radius, eps2, temps, meanG, varG, sizeInfo, stream);

	//3次元メモリ確保
	cudaPitchedPtr volume;
	cudaExtent volumeSizeBytes = make_cudaExtent(sizeof(float) * sizeInfo.width, sizeInfo.height, Imax);
	gpuErrchk(cudaMalloc3D(&volume, volumeSizeBytes));
	size_t gridPitch = volume.pitch;
	size_t pitchSlice = gridPitch * sizeInfo.height;

	//入力画像を0,1としてボリュームにセット
	cu_storeDataIntoVolume(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, sizeInfo.range, f, volume, sizeInfo.pitch<int>());

	//各スライスに対してguided filter
	char* d_ptr = static_cast<char*>(volume.ptr);
	for (int i = 0; i < sizeInfo.range; i++)
	{
		float* ptr = (float*)(d_ptr);

		cu_GuidedFilteringWithPrecalculation(ptr, g->host, meanG, varG, radius, eps2, temps, dst, sizeInfo, stream);

		cu_copyToSlice(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, i, dst, volume, sizeInfo.pitch<float>());
		d_ptr += pitchSlice;
	}

	//中央値計算
	cu_calculateMedian(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, sizeInfo.range, volume, result_center, sizeInfo.pitch<int>());

	//メモリ開放
	cudaFree(dst);
	for (int i = 0; i < 3; i++)
		cudaFree(meanG[i]);
	delete meanG;
	for (int i = 0; i < 6; i++)
		cudaFree(varG[i]);
	for (int i = 0; i < tempNum; i++)
		cudaFree(temps[i]);
	delete(temps);
	cudaFree(volume.ptr);
}



//I3G3
void cu_ConstantTimeWMF(SizeInfo& sizeInfo, cudaStream_t stream, int radius, int Imax, DeviceArray<int>* result_center, DeviceArray<int>* f, DeviceArray<float>* g, float eps2)
{
	//メモリ確保
	int tempNum = 9;
	float** temps;
	temps = new float*[tempNum];
	for (int i = 0; i < tempNum; i++)
		UtilityForCUDA::allocateDeviceMemory(temps[i], sizeInfo);

	float** meanG;
	float** varG;
	meanG = new float*[3];
	varG = new float*[6];
	for (int i = 0; i < 3; i++)
		UtilityForCUDA::allocateDeviceMemory(meanG[i], sizeInfo);
	for (int i = 0; i < 6; i++)
		UtilityForCUDA::allocateDeviceMemory(varG[i], sizeInfo);
	float* dst;//スライスに対するGF結果格納用
	UtilityForCUDA::allocateDeviceMemory(dst, sizeInfo);


	//ガイデッドフィルタの事前要素計算(3チャンネル)
	cu_precalculationForGuidedFiltering(g->host, radius, eps2, temps, meanG, varG, sizeInfo, stream);

	//3次元メモリ確保
	cudaPitchedPtr volume;
	cudaExtent volumeSizeBytes = make_cudaExtent(sizeof(float) * sizeInfo.width, sizeInfo.height, Imax);
	gpuErrchk(cudaMalloc3D(&volume, volumeSizeBytes));
	size_t gridPitch = volume.pitch;
	size_t pitchSlice = gridPitch * sizeInfo.height;

	//各入力チャンネルに対して処理
	for (int c = 0; c < 3; c++)
	{
		//入力画像を0,1としてボリュームにセット
		cu_storeDataIntoVolume(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, sizeInfo.range, f->host[c], volume, sizeInfo.pitch<int>());

		//各スライスに対してguided filter
		char* d_ptr = static_cast<char*>(volume.ptr);
		for (int i = 0; i < sizeInfo.range; i++)
		{
			float* ptr = (float*)(d_ptr);

			cu_GuidedFilteringWithPrecalculation(ptr, g->host, meanG, varG, radius, eps2, temps, dst, sizeInfo, stream);

			cu_copyToSlice(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, i, dst, volume, sizeInfo.pitch<float>());
			d_ptr += pitchSlice;
		}

		//中央値計算
		cu_calculateMedian(sizeInfo.blockSize, sizeInfo.gridSize, NULL, sizeInfo.width, sizeInfo.height, sizeInfo.range, volume, result_center->host[c], sizeInfo.pitch<int>());
	}

	//メモリ開放
	cudaFree(dst);
	for (int i = 0; i < 3; i++)
		cudaFree(meanG[i]);
	delete meanG;
	for (int i = 0; i < 6; i++)
		cudaFree(varG[i]);
	delete varG;
	for (int i = 0; i < tempNum; i++)
		cudaFree(temps[i]);
	delete temps;
	cudaFree(volume.ptr);
}
